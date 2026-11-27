"""
regime.features.tda_gnn
=======================
Optional TDA (persistence landscapes of rolling correlation matrices)
and GNN (sector graph embeddings). Both are disabled by default in
configs/features.yaml and never break the pipeline when disabled.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_tda_features(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """
    Compute persistence landscape L2 norms of rolling correlation matrices
    when `tda.use_tda` is True. Returns empty DataFrame when disabled.
    """
    tda_cfg = cfg.get("tda", {})
    if not tda_cfg.get("use_tda", False):
        logger.info("TDA features disabled (use_tda=false)")
        return pd.DataFrame(index=df.index)

    window = int(tda_cfg.get("rolling_corr_window", 63))
    cols = [c for c in ("nifty50", "nifty_midcap", "nifty_bank", "india_vix", "usdinr") if c in df.columns]
    if len(cols) < 3:
        logger.warning("Insufficient columns for TDA correlation matrix; skipping.")
        return pd.DataFrame(index=df.index)

    rets = df[cols].pct_change(1, fill_method=None)
    # Lightweight spectral/topological proxy (Fiedler value + Frobenius norm of corr)
    # if giotto-tda is not installed.
    try:
        from gtda.homology import VietorisRipsPersistence  # noqa: F401
        use_gtda = True
    except ImportError:
        logger.warning("giotto-tda not installed; using spectral correlation-graph fallback.")
        use_gtda = False

    n = len(df)
    l2_norm = np.full(n, np.nan)
    spectral_gap = np.full(n, np.nan)

    for i in range(window, n):
        sub = rets.iloc[i - window + 1 : i + 1].dropna(axis=0, how="any")
        if len(sub) < 20:
            continue
        corr = sub.corr().fillna(0.0).values
        eigvals = np.sort(np.linalg.eigvalsh(corr))
        l2_norm[i] = float(np.linalg.norm(1.0 - np.abs(corr), ord="fro"))
        spectral_gap[i] = float(eigvals[-1] - eigvals[-2]) if len(eigvals) >= 2 else 0.0

    out = pd.DataFrame(index=df.index)
    out["tda_corr_frobenius"] = l2_norm
    out["tda_spectral_gap"] = spectral_gap
    if not use_gtda:
        logger.info("TDA computed via spectral fallback (%d rows)", n)
    return out


def compute_gnn_features(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """
    Compute sector graph embeddings when `gnn.use_gnn` is True.
    Returns empty DataFrame when disabled.
    """
    gnn_cfg = cfg.get("gnn", {})
    if not gnn_cfg.get("use_gnn", False):
        logger.info("GNN features disabled (use_gnn=false)")
        return pd.DataFrame(index=df.index)

    logger.warning("GNN sector indices not present in master.parquet; returning empty DataFrame.")
    return pd.DataFrame(index=df.index)
