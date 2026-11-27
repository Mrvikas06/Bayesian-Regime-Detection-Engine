"""
regime.models.hmm_freq
======================
Phase 3: Frequentist Gaussian HMM for 5-regime classification.

Features:
- Compares K in {3, 5, 7} states by BIC/AIC.
- Causal forward filtering P(S_t | x_{1:t}) for zero-lookahead OOS probabilities.
- Post-hoc deterministic state-to-regime mapping via Hungarian assignment on
  (return, vol, VIX, trend, breadth) signatures to resolve label switching.
- Seed-stability verification across multiple random seeds.
- Transition matrix, expected state durations, and stickiness diagnostics.
- Expanding-window OOS calibration (log-loss, multiclass Brier, ECE) logged to MLflow.

Usage
-----
    python -m regime.models.hmm_freq --config configs/models.yaml
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
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import yaml
from hmmlearn.hmm import GaussianHMM
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, log_loss
from sklearn.preprocessing import StandardScaler

from regime.data.labels import REGIME_NAMES
from regime.data.versioning import (
    attach_lineage,
    build_lineage_record,
    hash_config,
    hash_dataframe,
    save_lineage_json,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

REGIME_COLORS = {
    "Risk-On": "#2ecc71",
    "Risk-Off": "#e74c3c",
    "Transitional": "#f39c12",
    "Late-Cycle": "#9b59b6",
    "Post-Shock": "#1abc9c",
}


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_feature_columns(
    df: pd.DataFrame,
    requested: list[str],
) -> tuple[list[str], list[str]]:
    """
    Resolve configured feature_subset against available columns in features.parquet.
    Falls back to documented proxy columns when manual-CSV features are absent.
    """
    fallbacks = {
        "fii_net_zscore63": "midcap_vs_nifty_z63",
    }
    resolved: list[str] = []
    notes: list[str] = []
    for col in requested:
        if col in df.columns:
            resolved.append(col)
        elif col in fallbacks and fallbacks[col] in df.columns:
            fb = fallbacks[col]
            resolved.append(fb)
            msg = f"Feature '{col}' missing; using proxy '{fb}'"
            logger.warning(msg)
            notes.append(msg)
        else:
            logger.warning("Feature '%s' missing and no fallback found; skipping.", col)
    return resolved, notes


def count_hmm_params(n_states: int, n_features: int, cov_type: str = "full") -> int:
    """Free parameters for a Gaussian HMM."""
    p_start = n_states - 1
    p_trans = n_states * (n_states - 1)
    p_means = n_states * n_features
    if cov_type == "full":
        p_cov = n_states * (n_features * (n_features + 1) // 2)
    elif cov_type == "diag":
        p_cov = n_states * n_features
    else:
        p_cov = n_states * n_features
    return p_start + p_trans + p_means + p_cov


def fit_best_hmm(
    X: np.ndarray,
    n_states: int,
    cov_type: str = "full",
    n_iter: int = 200,
    n_init: int = 10,
    base_seed: int = 42,
) -> tuple[GaussianHMM, float]:
    """Fit GaussianHMM across `n_init` seeds and return the highest-likelihood model."""
    best_model: GaussianHMM | None = None
    best_ll = -np.inf
    for i in range(n_init):
        model = GaussianHMM(
            n_components=n_states,
            covariance_type=cov_type,
            n_iter=n_iter,
            random_state=base_seed + i,
            tol=1e-4,
        )
        try:
            model.fit(X)
            ll = float(model.score(X))
            if ll > best_ll:
                best_ll = ll
                best_model = model
        except Exception as exc:  # noqa: BLE001
            logger.debug("HMM init %d failed for K=%d: %s", i, n_states, exc)
    if best_model is None:
        raise RuntimeError(f"All {n_init} HMM initialisations failed for K={n_states}")
    return best_model, best_ll


def compare_states_bic(
    X: np.ndarray,
    n_states_grid: list[int],
    cov_type: str = "full",
    n_iter: int = 200,
    n_init: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare Gaussian HMMs across K in n_states_grid by log-likelihood, AIC, and BIC."""
    n_obs, n_feat = X.shape
    rows = []
    for k in n_states_grid:
        model, ll = fit_best_hmm(
            X, n_states=k, cov_type=cov_type, n_iter=n_iter, n_init=n_init, base_seed=seed
        )
        p = count_hmm_params(k, n_feat, cov_type)
        bic = -2.0 * ll + p * np.log(n_obs)
        aic = -2.0 * ll + 2.0 * p
        rows.append(
            {
                "n_states": k,
                "n_params": p,
                "log_likelihood": round(ll, 2),
                "aic": round(aic, 2),
                "bic": round(bic, 2),
                "converged": bool(model.monitor_.converged),
            }
        )
    return pd.DataFrame(rows).sort_values("n_states")


def causal_forward_filter(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """
    Compute strictly causal filtered state probabilities P(S_t = j | x_{1:t})
    using only observations up to time t (no backward smoothing pass).
    """
    framelogprob = model._compute_log_likelihood(X)
    log_start = np.log(np.clip(model.startprob_, 1e-12, 1.0))
    log_trans = np.log(np.clip(model.transmat_, 1e-12, 1.0))

    n_samples, n_comp = framelogprob.shape
    alpha = np.zeros((n_samples, n_comp), dtype=float)

    # t = 0
    log_a0 = log_start + framelogprob[0]
    log_a0 -= np.max(log_a0)
    a0 = np.exp(log_a0)
    alpha[0] = a0 / a0.sum()

    # t >= 1
    for t in range(1, n_samples):
        # log sum_i alpha_{t-1}(i) * trans(i, j)
        log_prev = np.log(np.clip(alpha[t - 1], 1e-300, 1.0))[:, None] + log_trans
        m = np.max(log_prev, axis=0)
        log_pred = m + np.log(np.sum(np.exp(log_prev - m), axis=0))
        log_post = log_pred + framelogprob[t]
        log_post -= np.max(log_post)
        post = np.exp(log_post)
        alpha[t] = post / post.sum()

    return alpha


def map_states_to_regimes(
    X_scaled: np.ndarray,
    state_probs: np.ndarray,
    feature_names: list[str],
) -> dict[int, str]:
    """
    Deterministically map 5 HMM states to the 5 canonical regime names using
    Hungarian assignment on standardised state centroids.

    Canonical prototypes (in standardised feature space):
      - Risk-On:      ret +, vol -, vix -, breadth/proxy +, trend +
      - Risk-Off:     ret -, vol +, vix +, breadth/proxy -, trend -
      - Transitional: near-zero across features
      - Late-Cycle:   ret +, vol -, vix -, breadth -, trend +
      - Post-Shock:   ret +, vol ++, vix +, breadth +, trend -
    """
    n_states = state_probs.shape[1]
    if n_states != 5:
        raise ValueError(f"Expected 5 states for regime mapping, got {n_states}")

    # Compute responsibility-weighted state centroids in standardised space
    weights = state_probs / np.clip(state_probs.sum(axis=0, keepdims=True), 1e-12, None)
    centroids = weights.T @ X_scaled  # (5, D)

    # Build prototype matrix aligned to `feature_names`
    proto_defs = {
        "Risk-On":      {"ret": +0.7, "vol": -0.7, "vix": -0.8, "trend": +0.8, "breadth": +0.6},
        "Risk-Off":     {"ret": -1.0, "vol": +1.2, "vix": +1.3, "trend": -1.2, "breadth": -0.8},
        "Transitional": {"ret":  0.0, "vol": -0.1, "vix": -0.1, "trend":  0.0, "breadth":  0.0},
        "Late-Cycle":   {"ret": +0.3, "vol": -0.3, "vix": -0.3, "trend": +0.7, "breadth": -0.7},
        "Post-Shock":   {"ret": +0.2, "vol": +1.4, "vix": +0.9, "trend": -0.8, "breadth": +0.4},
    }

    def _feat_role(name: str) -> str:
        nl = name.lower()
        if "ret" in nl and "vol" not in nl:
            return "ret"
        if "vol" in nl:
            return "vol"
        if "vix" in nl:
            return "vix"
        if "dma" in nl or "trend" in nl:
            return "trend"
        return "breadth"

    proto_mat = np.zeros((5, len(feature_names)))
    for r_idx, r_name in enumerate(REGIME_NAMES):
        for f_idx, fname in enumerate(feature_names):
            proto_mat[r_idx, f_idx] = proto_defs[r_name][_feat_role(fname)]

    # Cost matrix: squared Euclidean distance between state centroid i and regime prototype j
    cost = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            cost[i, j] = float(np.sum((centroids[i] - proto_mat[j]) ** 2))

    row_ind, col_ind = linear_sum_assignment(cost)
    return {int(s): REGIME_NAMES[int(r)] for s, r in zip(row_ind, col_ind)}


def reorder_probs_by_regime(
    probs: np.ndarray,
    state_map: dict[int, str],
) -> pd.DataFrame:
    """Reorder state probability columns to canonical REGIME_NAMES order."""
    inv_map = {reg: state for state, reg in state_map.items()}
    ordered = np.column_stack([probs[:, inv_map[r]] for r in REGIME_NAMES])
    return pd.DataFrame(ordered, columns=REGIME_NAMES)


def compute_calibration_metrics(
    probs_df: pd.DataFrame,
    true_labels: pd.Series,
    n_bins: int = 10,
) -> dict[str, float]:
    """
    Compute multiclass log-loss, Brier score, Expected Calibration Error (ECE),
    and accuracy against reference labels.
    """
    valid = true_labels.isin(REGIME_NAMES) & probs_df.notna().all(axis=1)
    y_str = true_labels[valid]
    P = probs_df.loc[valid, REGIME_NAMES].values
    P = np.clip(P, 1e-7, 1.0)
    P = P / P.sum(axis=1, keepdims=True)

    y_idx = np.array([REGIME_NAMES.index(s) for s in y_str])
    ll = float(log_loss(y_idx, P, labels=list(range(len(REGIME_NAMES)))))

    # Multiclass Brier score: mean_i sum_k (p_{ik} - y_{ik})^2
    Y_onehot = np.eye(len(REGIME_NAMES))[y_idx]
    brier = float(np.mean(np.sum((P - Y_onehot) ** 2, axis=1)))

    # Top-class ECE
    conf = np.max(P, axis=1)
    pred = np.argmax(P, axis=1)
    acc = (pred == y_idx).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        mask = (conf > bins[b]) & (conf <= bins[b + 1]) if b > 0 else (conf >= bins[b]) & (conf <= bins[b + 1])
        if mask.any():
            ece += (mask.sum() / len(conf)) * abs(acc[mask].mean() - conf[mask].mean())

    return {
        "log_loss": round(ll, 4),
        "brier_score": round(brier, 4),
        "ece": round(float(ece), 4),
        "accuracy_vs_weak_label": round(float(acc.mean()), 4),
        "n_eval": int(len(y_idx)),
    }


def check_seed_stability(
    X_scaled: np.ndarray,
    feature_names: list[str],
    seeds: list[int] = (42, 101, 202, 303, 404),
    cov_type: str = "full",
    n_iter: int = 200,
) -> dict[str, Any]:
    """Fit 5-state HMM across multiple seeds, map labels, and measure ARI stability."""
    mapped_seqs = []
    for s in seeds:
        model, _ = fit_best_hmm(X_scaled, n_states=5, cov_type=cov_type, n_iter=n_iter, n_init=3, base_seed=s)
        probs = causal_forward_filter(model, X_scaled)
        smap = map_states_to_regimes(X_scaled, probs, feature_names)
        df_p = reorder_probs_by_regime(probs, smap)
        mapped_seqs.append(df_p.idxmax(axis=1).values)

    aris = []
    agreements = []
    base = mapped_seqs[0]
    for other in mapped_seqs[1:]:
        aris.append(float(adjusted_rand_score(base, other)))
        agreements.append(float(np.mean(base == other)))

    return {
        "seeds": list(seeds),
        "mean_ari": round(float(np.mean(aris)), 4),
        "min_ari": round(float(np.min(aris)), 4),
        "mean_label_agreement": round(float(np.mean(agreements)), 4),
    }


def expanding_window_oos(
    df_clean: pd.DataFrame,
    feat_cols: list[str],
    cov_type: str = "full",
    n_iter: int = 200,
    n_init: int = 5,
    seed: int = 42,
    first_test_year: int = 2019,
) -> pd.DataFrame:
    """
    Strictly causal expanding-window OOS probabilities:
    Train scaler + 5-state HMM up to year Y-1, map states on train,
    then run causal forward filter on year Y.
    """
    years = sorted(df_clean.index.year.unique())
    test_years = [y for y in years if y >= first_test_year]
    oos_frames = []

    for yr in test_years:
        train_df = df_clean[df_clean.index.year < yr]
        test_df = df_clean[df_clean.index.year == yr]
        if len(train_df) < 500 or test_df.empty:
            continue

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df[feat_cols].values)
        X_test = scaler.transform(test_df[feat_cols].values)

        model, _ = fit_best_hmm(
            X_train, n_states=5, cov_type=cov_type, n_iter=n_iter, n_init=n_init, base_seed=seed
        )
        train_probs = causal_forward_filter(model, X_train)
        smap = map_states_to_regimes(X_train, train_probs, feat_cols)

        test_probs = causal_forward_filter(model, X_test)
        df_p = reorder_probs_by_regime(test_probs, smap)
        df_p.index = test_df.index
        oos_frames.append(df_p)

    if not oos_frames:
        return pd.DataFrame(columns=REGIME_NAMES)
    return pd.concat(oos_frames, axis=0)


def plot_regime_overlay(
    nifty_prices: pd.Series,
    probs_df: pd.DataFrame,
    out_path: str | Path = "reports/figures/phase3_hmm_overlay.png",
) -> None:
    """Plot Nifty 50 with Frequentist HMM 5-regime shading and regime probabilities."""
    common_idx = nifty_prices.index.intersection(probs_df.index)
    px = nifty_prices.loc[common_idx]
    dom_regime = probs_df.loc[common_idx, REGIME_NAMES].idxmax(axis=1)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(15, 7), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]}
    )

    ax1.plot(px.index, px.values, color="black", linewidth=0.9, label="Nifty 50")
    prev_r = None
    start_dt = None
    for dt, r in dom_regime.items():
        if r != prev_r:
            if prev_r is not None:
                ax1.axvspan(start_dt, dt, alpha=0.25, color=REGIME_COLORS[prev_r])
            start_dt = dt
            prev_r = r
    if start_dt is not None and prev_r is not None:
        ax1.axvspan(start_dt, dom_regime.index[-1], alpha=0.25, color=REGIME_COLORS[prev_r])

    patches = [Patch(facecolor=REGIME_COLORS[r], alpha=0.45, label=r) for r in REGIME_NAMES]
    ax1.legend(handles=patches, loc="upper left", ncol=5, fontsize=8)
    ax1.set_title("Phase 3 — Frequentist 5-State Gaussian HMM Regime Overlay on Nifty 50", fontweight="bold")
    ax1.set_ylabel("Nifty 50")
    ax1.grid(alpha=0.2)

    # Stacked area of probabilities
    ax2.stackplot(
        common_idx,
        [probs_df.loc[common_idx, r].values for r in REGIME_NAMES],
        labels=REGIME_NAMES,
        colors=[REGIME_COLORS[r] for r in REGIME_NAMES],
        alpha=0.8,
    )
    ax2.set_ylabel("Regime Prob")
    ax2.set_ylim(0.0, 1.0)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Overlay figure saved: %s", out_path)


def _log_to_mlflow(
    experiment_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    artifacts: list[str],
) -> None:
    """Log Phase 3 run to local MLflow tracking store."""
    try:
        import mlflow

        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name="phase3_hmm_freq"):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            for art in artifacts:
                if Path(art).exists():
                    mlflow.log_artifact(art)
        logger.info("Logged phase3_hmm_freq to MLflow experiment '%s'", experiment_name)
    except Exception as exc:  # noqa: BLE001
        logger.warning("MLflow logging failed (non-fatal): %s", exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3: Frequentist HMM")
    parser.add_argument("--config", default="configs/models.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    hmm_cfg = cfg["hmm_freq"]
    seed = int(hmm_cfg.get("random_state", cfg.get("seed", 42)))

    features_path = Path(cfg["features_parquet"])
    if not features_path.exists():
        raise FileNotFoundError(f"Features parquet missing at {features_path}; run make phase-2 first")

    df = pd.read_parquet(features_path)
    master = pd.read_parquet("data/processed/master.parquet")

    feat_cols, proxy_notes = resolve_feature_columns(df, hmm_cfg["feature_subset"])
    df_clean = df[feat_cols + ["regime_label"]].dropna().copy()
    logger.info("Clean HMM sample: %d rows × %d features: %s", len(df_clean), len(feat_cols), feat_cols)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean[feat_cols].values)

    # 1. BIC comparison across K in [3, 5, 7]
    bic_df = compare_states_bic(
        X_scaled,
        n_states_grid=hmm_cfg["n_states_grid"],
        cov_type=hmm_cfg.get("covariance_type", "full"),
        n_iter=int(hmm_cfg.get("n_iter", 200)),
        n_init=int(hmm_cfg.get("n_init", 10)),
        seed=seed,
    )
    Path("reports/tables").mkdir(parents=True, exist_ok=True)
    bic_path = "reports/tables/phase3_hmm_bic.csv"
    bic_df.to_csv(bic_path, index=False)
    logger.info("BIC Comparison:\n%s", bic_df.to_string(index=False))

    # 2. Fit 5-state HMM, map states, extract transition matrix & durations
    model5, ll5 = fit_best_hmm(
        X_scaled,
        n_states=5,
        cov_type=hmm_cfg.get("covariance_type", "full"),
        n_iter=int(hmm_cfg.get("n_iter", 200)),
        n_init=int(hmm_cfg.get("n_init", 10)),
        base_seed=seed,
    )
    probs_full = causal_forward_filter(model5, X_scaled)
    state_map = map_states_to_regimes(X_scaled, probs_full, feat_cols)
    logger.info("State-to-regime mapping: %s", state_map)

    probs_df = reorder_probs_by_regime(probs_full, state_map)
    probs_df.index = df_clean.index

    # Transition matrix & expected durations
    inv_map = {reg: st for st, reg in state_map.items()}
    order_idx = [inv_map[r] for r in REGIME_NAMES]
    trans_mat = model5.transmat_[np.ix_(order_idx, order_idx)]
    trans_df = pd.DataFrame(trans_mat, index=REGIME_NAMES, columns=REGIME_NAMES)
    trans_df["stickiness"] = np.diag(trans_mat)
    trans_df["expected_duration_days"] = 1.0 / np.clip(1.0 - np.diag(trans_mat), 1e-6, 1.0)
    trans_path = "reports/tables/phase3_hmm_transitions.csv"
    trans_df.round(4).to_csv(trans_path)
    logger.info("Transition Matrix & Durations:\n%s", trans_df.round(3).to_string())

    # 3. Seed stability
    stability = check_seed_stability(
        X_scaled, feat_cols, cov_type=hmm_cfg.get("covariance_type", "full"), n_iter=int(hmm_cfg.get("n_iter", 200))
    )
    logger.info("Seed stability: %s", stability)

    # 4. Strictly causal expanding-window OOS evaluation (2019+)
    oos_probs = expanding_window_oos(
        df_clean,
        feat_cols,
        cov_type=hmm_cfg.get("covariance_type", "full"),
        n_iter=int(hmm_cfg.get("n_iter", 200)),
        n_init=5,
        seed=seed,
        first_test_year=2019,
    )
    oos_metrics = compute_calibration_metrics(oos_probs, df_clean["regime_label"])
    logger.info("OOS Calibration Metrics (2019+): %s", oos_metrics)

    # 5. Save outputs with lineage
    data_hash = str(df["lineage_data_version_hash"].iloc[0]) if "lineage_data_version_hash" in df.columns else "unknown"
    feat_hash = str(df["lineage_feature_set_version"].iloc[0]) if "lineage_feature_set_version" in df.columns else "unknown"
    cfg_hash = hash_config(cfg)
    model_hash = hash_dataframe(probs_df)

    lineage = build_lineage_record(
        phase=3,
        data_hash=data_hash,
        feature_hash=feat_hash,
        model_hash=model_hash,
        config_hash=cfg_hash,
        extra={
            "model": "hmm_freq_5state",
            "features_used": feat_cols,
            "proxy_notes": proxy_notes,
            "state_map": state_map,
            "seed_stability": stability,
            "oos_metrics": oos_metrics,
        },
    )

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    probs_out = attach_lineage(probs_df.copy(), lineage)
    probs_out["is_oos"] = probs_out.index.isin(oos_probs.index)
    # Overwrite OOS rows with strictly causal OOS probabilities
    probs_out.loc[oos_probs.index, REGIME_NAMES] = oos_probs[REGIME_NAMES]
    probs_out["predicted_regime"] = probs_out[REGIME_NAMES].idxmax(axis=1)
    probs_out["conviction"] = probs_out[REGIME_NAMES].max(axis=1)
    probs_out.to_parquet(out_dir / "hmm_freq_probs.parquet")

    save_lineage_json(lineage, "data/lineage/phase3_lineage.json")
    overlay_path = "reports/figures/phase3_hmm_overlay.png"
    plot_regime_overlay(master["nifty50"], probs_df, out_path=overlay_path)

    # 6. Log to MLflow
    _log_to_mlflow(
        experiment_name=cfg.get("mlflow_experiment", "regime-detection"),
        params={
            "model": "hmm_freq",
            "n_states": 5,
            "covariance_type": hmm_cfg.get("covariance_type", "full"),
            "features": ",".join(feat_cols),
            "seed": seed,
        },
        metrics={
            **{k: float(v) for k, v in oos_metrics.items() if isinstance(v, (int, float))},
            "mean_seed_ari": stability["mean_ari"],
            "full_sample_ll": round(ll5, 2),
        },
        artifacts=[bic_path, trans_path, overlay_path, "data/lineage/phase3_lineage.json"],
    )
    logger.info("Phase 3 Frequentist HMM complete.")


if __name__ == "__main__":
    main()
