"""
regime.features.volatility
==========================
Causal volatility features: realised vol across horizons, vol ratios,
India VIX level/changes/z-score, and VIX-to-realised-vol term proxies.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def compute_volatility_features(
    df: pd.DataFrame,
    realised_windows: Sequence[int] = (5, 21, 63),
) -> pd.DataFrame:
    """Compute annualised realised volatility and VIX-derived features."""
    out = pd.DataFrame(index=df.index)

    if "nifty50" in df.columns:
        ret = df["nifty50"].pct_change(1, fill_method=None)
        rv_map: dict[int, pd.Series] = {}
        for w in set(realised_windows) | {252}:
            rv = ret.rolling(w, min_periods=max(3, w // 2)).std() * np.sqrt(252.0) * 100.0
            rv_map[w] = rv
            out[f"nifty50_ret_vol{w}"] = rv

        if 5 in rv_map and 21 in rv_map:
            out["vol_ratio_5_21"] = rv_map[5] / rv_map[21].replace(0.0, np.nan)
        if 21 in rv_map and 63 in rv_map:
            out["vol_ratio_21_63"] = rv_map[21] / rv_map[63].replace(0.0, np.nan)

    if "india_vix" in df.columns:
        vix = df["india_vix"]
        out["india_vix"] = vix
        out["vix_chg_5d"] = vix.diff(5)
        out["vix_chg_21d"] = vix.diff(21)
        vix_mean = vix.rolling(252, min_periods=63).mean()
        vix_std = vix.rolling(252, min_periods=63).std().replace(0.0, np.nan)
        out["vix_zscore_252d"] = (vix - vix_mean) / vix_std

        if "nifty50_ret_vol252" in out.columns:
            out["vix_rv252_ratio"] = vix / out["nifty50_ret_vol252"].replace(0.0, np.nan)
        if "nifty50_ret_vol21" in out.columns:
            out["vix_rv21_ratio"] = vix / out["nifty50_ret_vol21"].replace(0.0, np.nan)

    return out
