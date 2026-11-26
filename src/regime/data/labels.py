"""
regime.data.labels
==================
Rule-based reference label series for the 5 regimes.

**This is a weak label, not ground truth.** It is used for:
- Accuracy scoring alongside log-likelihood
- Labelling HMM states post-hoc
- Investment Committee interpretability

Design: apply rules sequentially with strict priority order.
If multiple rules fire, the first matching rule wins.
Ambiguous days where no rule has clear majority → Transitional.

See docs/decisions.md D-002 for rationale.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REGIME_NAMES = ["Risk-On", "Risk-Off", "Transitional", "Late-Cycle", "Post-Shock"]
REGIME_INT = {name: i for i, name in enumerate(REGIME_NAMES)}

# ---------------------------------------------------------------------------
# Individual signal functions (all return pd.Series of bool/float on same idx)
# ---------------------------------------------------------------------------


def _rolling_mean(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=max(1, window // 2)).mean()


def compute_rule_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute intermediate boolean signals used by the labelling rules.

    Parameters
    ----------
    df : pd.DataFrame
        Master DataFrame with columns including nifty50, india_vix, usdinr, etc.

    Returns
    -------
    pd.DataFrame
        Boolean/float signal columns aligned on df.index.
    """
    sigs = pd.DataFrame(index=df.index)

    # ---- Nifty trend signals ----
    if "nifty50" in df.columns:
        px = df["nifty50"]
        sigs["dma50"]  = _rolling_mean(px, 50)
        sigs["dma200"] = _rolling_mean(px, 200)
        sigs["nifty_above_dma200"]  = px > sigs["dma200"]
        sigs["nifty_below_dma200"]  = px < sigs["dma200"]
        sigs["golden_cross"]        = sigs["dma50"] > sigs["dma200"]
        # 21-day rolling drawdown from 252-day high
        sigs["drawdown_from_high"]  = px / px.rolling(252, min_periods=63).max() - 1

    # ---- VIX signals ----
    if "india_vix" in df.columns:
        vix = df["india_vix"]
        sigs["vix_risk_on"]   = vix < 14.0
        sigs["vix_risk_off"]  = vix > 22.0
        sigs["vix_mid"]       = (vix >= 14.0) & (vix <= 22.0)
        # 21-day rolling realised vol of VIX (rising vs falling)
        sigs["vix_momentum"]  = vix.diff(21)

    # ---- FII flow signals (if available) ----
    fii_col = next((c for c in df.columns if "fii_net" in c.lower()), None)
    if fii_col:
        fii = df[fii_col]
        sigs["fii_net_21d"]    = fii.rolling(21, min_periods=5).sum()
        sigs["fii_inflow"]     = sigs["fii_net_21d"] > 0
        sigs["fii_outflow"]    = sigs["fii_net_21d"] < 0
        sigs["fii_persistent_outflow"] = fii.rolling(63, min_periods=10).sum() < 0
    else:
        sigs["fii_inflow"]             = pd.Series(np.nan, index=df.index)
        sigs["fii_outflow"]            = pd.Series(np.nan, index=df.index)
        sigs["fii_persistent_outflow"] = pd.Series(np.nan, index=df.index)

    # ---- INR signals ----
    if "usdinr" in df.columns:
        inr = df["usdinr"]
        sigs["inr_depreciation_63d"] = inr.pct_change(63, fill_method=None) > 0.03   # > 3% depreciation

    # ---- Gilt yield signals ----
    if "gilt_10y" in df.columns:
        gilt = df["gilt_10y"]
        sigs["gilt_rising"] = gilt.diff(63) > 0.25   # yield risen > 25bps in 63 days

    # ---- Breadth proxy: Midcap relative to Largecap ----
    if "nifty_midcap" in df.columns and "nifty50" in df.columns:
        rel = df["nifty_midcap"] / df["nifty50"]
        sigs["breadth_strong"]  = rel / _rolling_mean(rel, 63) > 1.0   # midcap outperforming
        sigs["breadth_weak"]    = rel / _rolling_mean(rel, 63) < 0.97
    else:
        sigs["breadth_strong"] = pd.Series(np.nan, index=df.index)
        sigs["breadth_weak"]   = pd.Series(np.nan, index=df.index)

    # ---- Volatility decay: Post-Shock ----
    if "nifty50" in df.columns:
        # Realised vol 5-day vs 21-day — vol compressing
        ret = df["nifty50"].pct_change()
        rv5  = ret.rolling(5).std() * np.sqrt(252)
        rv21 = ret.rolling(21).std() * np.sqrt(252)
        sigs["vol_compressing"]   = rv5 < rv21
        sigs["deep_drawdown_63d"] = sigs.get("drawdown_from_high", pd.Series(0, index=df.index)) < -0.10

    return sigs


def label_regimes(df: pd.DataFrame) -> pd.Series:
    """
    Apply rule hierarchy to produce a regime label series.

    Priority (first match wins):
      1. Post-Shock  — recent deep drawdown + vol compressing
      2. Risk-Off    — Nifty below 200DMA + VIX > 22 + FII outflow
      3. Late-Cycle  — golden cross but breadth weak (midcap lagging)
      4. Risk-On     — Nifty above 200DMA + golden cross + VIX < 14 + FII inflow
      5. Transitional — everything else

    Returns
    -------
    pd.Series of str, same index as df.
    """
    sigs = compute_rule_signals(df)
    n = len(df)
    labels = pd.Series(["Transitional"] * n, index=df.index, name="regime_label")

    def _safe(col: str, default: bool = False) -> pd.Series:
        """Return signal column or a constant Series if unavailable."""
        if col in sigs.columns:
            s = sigs[col]
            # Cast to bool-compatible type before filling to avoid FutureWarning
            if s.dtype == object:
                s = s.astype("boolean")
            return s.fillna(default)
        return pd.Series(default, index=df.index)

    # ---- 4. Risk-On (set first so lower-priority rules can override) ----
    risk_on_mask = (
        _safe("nifty_above_dma200", False)
        & _safe("golden_cross", False)
        & _safe("vix_risk_on", False)
        & _safe("fii_inflow", True)   # if FII data missing, don't penalise
    )
    labels[risk_on_mask] = "Risk-On"

    # ---- Late-Cycle (overrides Risk-On when breadth weak) ----
    late_cycle_mask = (
        _safe("nifty_above_dma200", False)
        & _safe("golden_cross", False)
        & _safe("breadth_weak", False)
    )
    labels[late_cycle_mask] = "Late-Cycle"

    # ---- Risk-Off (overrides Late-Cycle) ----
    risk_off_mask = (
        _safe("nifty_below_dma200", False)
        & _safe("vix_risk_off", False)
        & _safe("fii_outflow", True)
    )
    labels[risk_off_mask] = "Risk-Off"

    # ---- Post-Shock (highest priority: recovery from crash) ----
    post_shock_mask = (
        _safe("deep_drawdown_63d", False)
        & _safe("vol_compressing", False)
        & ~_safe("vix_risk_off", False)   # VIX is moderating, not spiking
    )
    labels[post_shock_mask] = "Post-Shock"

    # Log distribution
    dist = labels.value_counts(normalize=True).round(3)
    logger.info("Reference label distribution:\n%s", dist.to_string())

    return labels


def label_to_int(labels: pd.Series) -> pd.Series:
    """Map string labels to integer codes 0–4."""
    return labels.map(REGIME_INT).astype("Int64")
