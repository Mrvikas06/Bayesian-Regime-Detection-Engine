"""
regime.features.breadth
=======================
Cap-segment relative strength and market breadth proxy features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_breadth_features(
    df: pd.DataFrame,
    rolling_window: int = 63,
) -> pd.DataFrame:
    """
    Compute cap-segment relative strength and breadth proxies from
    Nifty 50, Nifty Midcap, and Nifty Bank.
    """
    out = pd.DataFrame(index=df.index)

    if "nifty_midcap" in df.columns and "nifty50" in df.columns:
        ratio = df["nifty_midcap"] / df["nifty50"].replace(0.0, np.nan)
        out["midcap_vs_nifty_ratio"] = ratio
        rm = ratio.rolling(rolling_window, min_periods=21).mean()
        rs = ratio.rolling(rolling_window, min_periods=21).std().replace(0.0, np.nan)
        out["midcap_vs_nifty_z63"] = (ratio - rm) / rs

        mc_ret21 = df["nifty_midcap"].pct_change(21, fill_method=None)
        n50_ret21 = df["nifty50"].pct_change(21, fill_method=None)
        out["cap_spread_ret_21d"] = mc_ret21 - n50_ret21

    if "nifty_bank" in df.columns and "nifty50" in df.columns:
        b_ratio = df["nifty_bank"] / df["nifty50"].replace(0.0, np.nan)
        out["bank_vs_nifty_ratio"] = b_ratio
        bm = b_ratio.rolling(rolling_window, min_periods=21).mean()
        bs = b_ratio.rolling(rolling_window, min_periods=21).std().replace(0.0, np.nan)
        out["bank_vs_nifty_z63"] = (b_ratio - bm) / bs

    # Breadth proxy: fraction of available equity indices above their N-DMA
    idx_cols = [c for c in ("nifty50", "nifty_midcap", "nifty_bank") if c in df.columns]
    if idx_cols:
        for w in (50, 200):
            above = pd.concat(
                [
                    (df[c] > df[c].rolling(w, min_periods=max(1, w // 2)).mean()).astype(float)
                    for c in idx_cols
                ],
                axis=1,
            )
            out[f"pct_indices_above_dma{w}"] = above.mean(axis=1)

    return out
