"""
regime.features.build
=====================
Phase 2 feature engineering orchestrator.

Builds 40+ strictly causal features from master.parquet, runs automated
no-leakage checks, generates breakpoint visualisations (2008, 2013, 2018,
2020, 2024), and writes versioned features.parquet + lineage JSON.

Usage
-----
    python -m regime.features.build --config configs/features.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from regime.data.labels import label_regimes
from regime.data.versioning import (
    attach_lineage,
    build_lineage_record,
    hash_config,
    hash_dataframe,
    save_lineage_json,
)
from regime.features.breadth import compute_breadth_features
from regime.features.macro_flows import compute_macro_flow_features
from regime.features.returns import compute_return_features
from regime.features.tda_gnn import compute_gnn_features, compute_tda_features
from regime.features.trend import compute_trend_features
from regime.features.volatility import compute_volatility_features

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

BREAKPOINTS = [
    ("2008-10-15", "2008 GFC"),
    ("2013-08-15", "2013 Taper"),
    ("2018-09-21", "2018 IL&FS"),
    ("2020-03-23", "2020 COVID"),
    ("2024-06-04", "2024 Election"),
]


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_feature_matrix(
    master: pd.DataFrame,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Compute all configured feature blocks from master DataFrame.
    Returns (features_df, proxy_notes).
    """
    trend_cfg = cfg.get("trend", {})
    vol_cfg = cfg.get("volatility", {})
    cap_cfg = cfg.get("cap_segment", {})

    ret_df = compute_return_features(master)
    trend_df = compute_trend_features(
        master,
        dma_windows=trend_cfg.get("dma_windows", [20, 50, 100, 200]),
        rsi_window=trend_cfg.get("rsi_window", 14),
    )
    vol_df = compute_volatility_features(
        master,
        realised_windows=vol_cfg.get("realised_windows", [5, 21, 63]),
    )
    breadth_df = compute_breadth_features(
        master,
        rolling_window=cap_cfg.get("rolling_window", 63),
    )
    macro_df, proxies = compute_macro_flow_features(master, cfg)
    tda_df = compute_tda_features(master, cfg)
    gnn_df = compute_gnn_features(master, cfg)

    parts = [ret_df, trend_df, vol_df, breadth_df, macro_df, tda_df, gnn_df]
    features = pd.concat([p for p in parts if not p.empty], axis=1)
    features = features.replace([np.inf, -np.inf], np.nan)
    return features, proxies


def check_future_correlation(
    features: pd.DataFrame,
    ref_col: str = "nifty50_ret_1d",
    max_corr: float = 0.05,
) -> pd.DataFrame:
    """
    Check |corr(feature_t, ref_return_{t+1})| for every numeric feature.
    Returns DataFrame of flagged features (if any).
    """
    if ref_col not in features.columns:
        logger.warning("Reference return %s not in features; skipping corr check", ref_col)
        return pd.DataFrame()

    next_ret = features[ref_col].shift(-1)
    rows = []
    for col in features.select_dtypes(include=[np.number]).columns:
        valid = features[col].notna() & next_ret.notna()
        if valid.sum() < 100:
            continue
        r = float(features.loc[valid, col].corr(next_ret.loc[valid]))
        if abs(r) > max_corr:
            rows.append({"feature": col, "corr_next_ret": round(r, 4)})
    return pd.DataFrame(rows)


def check_causal_perturbation(
    master: pd.DataFrame,
    cfg: dict[str, Any],
    cutoff_date: str = "2022-01-03",
) -> bool:
    """
    Perturb master data strictly after `cutoff_date` and verify all
    features at or before `cutoff_date` remain identical.
    """
    base_feats, _ = build_feature_matrix(master, cfg)
    perturbed = master.copy()
    mask_after = perturbed.index > pd.Timestamp(cutoff_date)
    if not mask_after.any():
        return True
    perturbed.loc[mask_after] = perturbed.loc[mask_after] * 3.5 + 100.0
    pert_feats, _ = build_feature_matrix(perturbed, cfg)

    left = base_feats.loc[:cutoff_date]
    right = pert_feats.loc[:cutoff_date]
    pd.testing.assert_frame_equal(left, right, check_exact=False, rtol=1e-12, atol=1e-12)
    return True


def plot_feature_breakpoints(
    features: pd.DataFrame,
    out_path: str | Path = "reports/figures/phase2_features_breakpoints.png",
) -> None:
    """Plot key regime features with historical crisis/event breakpoints."""
    panels = [
        ("nifty50_vs_dma200", "Nifty 50 vs 200-DMA"),
        ("india_vix", "India VIX Level"),
        ("nifty50_ret_vol21", "21d Realised Vol (%)"),
        ("midcap_vs_nifty_z63", "Midcap/Nifty Rel Strength (63d z)"),
        ("inr_orth_dxy", "INR Orthogonalised to DXY"),
    ]
    available = [(col, title) for col, title in panels if col in features.columns]
    if not available:
        return

    fig, axes = plt.subplots(len(available), 1, figsize=(15, 2.6 * len(available)), sharex=True)
    if len(available) == 1:
        axes = [axes]

    for ax, (col, title) in zip(axes, available):
        s = features[col].dropna()
        ax.plot(s.index, s.values, linewidth=0.9, color="#1f4e79")
        ax.axhline(0.0, color="grey", linewidth=0.6, linestyle="--", alpha=0.6)
        ax.set_ylabel(title, fontsize=9)
        ax.grid(alpha=0.25)
        for dt_str, label in BREAKPOINTS:
            dt = pd.Timestamp(dt_str)
            if s.index.min() <= dt <= s.index.max():
                ax.axvline(dt, color="#c0392b", linestyle="--", linewidth=1.0, alpha=0.8)
                if ax is axes[0]:
                    ax.text(
                        dt,
                        ax.get_ylim()[1] * 0.92,
                        f" {label}",
                        color="#c0392b",
                        fontsize=8,
                        rotation=0,
                        va="top",
                    )

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle(
        "Phase 2 — Key Regime Features & Historical Breakpoints (2008, 2013, 2018, 2020, 2024)",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Breakpoint figure saved: %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2: Feature Engineering")
    parser.add_argument("--config", default="configs/features.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    master_path = Path(cfg["input_parquet"])
    if not master_path.exists():
        raise FileNotFoundError(f"Master parquet missing at {master_path}; run make phase-1 first")

    master = pd.read_parquet(master_path)
    data_hash = hash_dataframe(master)
    cfg_hash = hash_config(cfg)

    features, proxies = build_feature_matrix(master, cfg)
    feature_cols = list(features.columns)
    logger.info("Built %d features over %d rows", len(feature_cols), len(features))

    # Leakage checks
    leak_cfg = cfg.get("leakage", {})
    flagged = check_future_correlation(
        features,
        ref_col=leak_cfg.get("reference_return", "nifty50_ret_1d"),
        max_corr=float(leak_cfg.get("max_future_correlation", 0.05)),
    )
    if not flagged.empty:
        logger.warning("Future-correlation check flagged:\n%s", flagged.to_string(index=False))
    else:
        logger.info("Future-correlation leakage check PASSED (0 flagged)")

    check_causal_perturbation(master, cfg)
    logger.info("Causal perturbation leakage test PASSED")

    # Attach weak labels and lineage
    features["regime_label"] = label_regimes(master)
    feat_hash = hash_dataframe(features[feature_cols])
    lineage = build_lineage_record(
        phase=2,
        data_hash=data_hash,
        feature_hash=feat_hash,
        config_hash=cfg_hash,
        extra={
            "n_features": len(feature_cols),
            "feature_columns": feature_cols,
            "proxies": proxies,
            "future_corr_flagged": flagged.to_dict(orient="records") if not flagged.empty else [],
        },
    )
    features_out = attach_lineage(features.copy(), lineage)

    out_path = Path(cfg["output_parquet"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features_out.to_parquet(out_path)
    logger.info("Features parquet saved: %s (%d rows × %d cols)", out_path, *features_out.shape)

    save_lineage_json(lineage, cfg["lineage_json"])
    plot_feature_breakpoints(features)
    logger.info("Phase 2 complete.")


if __name__ == "__main__":
    main()
