"""
regime.features.macro_flows
===========================
Causal macro, FX, rate, flow, and valuation features.
Missing manual series (FII/DII, SIP, India 10Y Gilt, Nifty P/E) are handled
gracefully with documented proxies and tracked in lineage.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _rolling_ols_residual(y: pd.Series, x: pd.Series, window: int = 63) -> pd.Series:
    """
    Strictly causal rolling OLS residual: y_t - (alpha_t + beta_t * x_t)
    where alpha_t, beta_t are estimated over [t - window + 1, t].
    """
    cov_xy = y.rolling(window, min_periods=21).cov(x)
    var_x = x.rolling(window, min_periods=21).var().replace(0.0, np.nan)
    beta = cov_xy / var_x
    alpha = y.rolling(window, min_periods=21).mean() - beta * x.rolling(window, min_periods=21).mean()
    return y - (alpha + beta * x)


def compute_macro_flow_features(
    df: pd.DataFrame,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Compute macro, flow, and valuation features.

    Returns
    -------
    (features_df, proxy_notes)
    """
    out = pd.DataFrame(index=df.index)
    proxies: list[str] = []

    # ---- FX & DXY ----
    if "usdinr" in df.columns:
        inr = df["usdinr"]
        out["usdinr_ret_1d"] = inr.pct_change(1, fill_method=None)
        out["usdinr_ret_21d"] = inr.pct_change(21, fill_method=None)
        out["usdinr_vol_21d"] = out["usdinr_ret_1d"].rolling(21, min_periods=10).std() * np.sqrt(252.0)

    if "dxy" in df.columns:
        dxy = df["dxy"]
        out["dxy_ret_21d"] = dxy.pct_change(21, fill_method=None)

    if "usdinr_ret_21d" in out.columns and "dxy_ret_21d" in out.columns:
        out["inr_orth_dxy"] = _rolling_ols_residual(
            out["usdinr_ret_21d"], out["dxy_ret_21d"], window=63
        )

    # ---- Rates & Policy Stance ----
    rate_col = "gilt_10y" if "gilt_10y" in df.columns else ("us_10y" if "us_10y" in df.columns else None)
    if rate_col is not None:
        if rate_col == "us_10y":
            proxies.append("real_rate_proxy & policy_stance_proxy use us_10y (india_10y_rbi.csv missing)")
        y10 = df[rate_col]
        out["us_10y_chg_21d"] = y10.diff(21)
        out["us_10y_chg_63d"] = y10.diff(63)
        if "usdinr" in df.columns:
            inr_dep_12m = df["usdinr"].pct_change(252, fill_method=None) * 100.0
            out["real_rate_proxy"] = y10 - inr_dep_12m
        out["policy_stance_proxy"] = np.where(y10.diff(63) < 0, 1.0, -1.0)
        out.loc[y10.diff(63).isna(), "policy_stance_proxy"] = np.nan

    # ---- FII/DII Flows (if present) ----
    fii_col = next((c for c in df.columns if "fii_net" in c.lower()), None)
    dii_col = next((c for c in df.columns if "dii_net" in c.lower()), None)
    fii_w = cfg.get("flows", {}).get("fii_zscore_window", 63)
    if fii_col is not None:
        fii = df[fii_col]
        out["fii_net_zscore63"] = (
            fii - fii.rolling(fii_w, min_periods=10).mean()
        ) / fii.rolling(fii_w, min_periods=10).std().replace(0.0, np.nan)
    else:
        proxies.append("fii_net_zscore63 absent (fii_dii_flows.csv missing)")

    if dii_col is not None:
        dii = df[dii_col]
        out["dii_net_zscore63"] = (
            dii - dii.rolling(fii_w, min_periods=10).mean()
        ) / dii.rolling(fii_w, min_periods=10).std().replace(0.0, np.nan)

    # ---- Valuation (P/E if available, else 252d price-stretch z-score proxy) ----
    pe_csv = cfg.get("valuation", {}).get("nifty_pe_csv", "")
    pe_win = cfg.get("valuation", {}).get("pe_zscore_window", 252)
    pe_loaded = False
    if pe_csv and Path(pe_csv).exists():
        pe_df = pd.read_csv(pe_csv, parse_dates=["date"], index_col="date")
        if not pe_df.empty and "pe_ratio" in pe_df.columns:
            pe_s = pe_df["pe_ratio"].reindex(df.index).ffill(limit=5)
            out["pe_zscore_252d"] = (
                pe_s - pe_s.rolling(pe_win, min_periods=63).mean()
            ) / pe_s.rolling(pe_win, min_periods=63).std().replace(0.0, np.nan)
            pe_loaded = True

    if not pe_loaded and "nifty50" in df.columns:
        px = df["nifty50"]
        stretch = px / px.rolling(pe_win, min_periods=63).mean() - 1.0
        out["valuation_stretch_z252"] = (
            stretch - stretch.rolling(pe_win, min_periods=63).mean()
        ) / stretch.rolling(pe_win, min_periods=63).std().replace(0.0, np.nan)
        proxies.append("valuation_stretch_z252 used as proxy for pe_zscore_252d (nifty_pe.csv empty)")

    return out, proxies
