"""
regime.features.returns
=======================
Strictly causal return features across equity indices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simple and multi-horizon returns for available equity indices.
    All calculations at time t depend only on prices <= t.
    """
    out = pd.DataFrame(index=df.index)

    if "nifty50" in df.columns:
        px = df["nifty50"]
        out["nifty50_ret_1d"] = px.pct_change(1, fill_method=None)
        out["nifty50_ret_5d"] = px.pct_change(5, fill_method=None)
        out["nifty50_ret_21d"] = px.pct_change(21, fill_method=None)
        out["nifty50_ret_63d"] = px.pct_change(63, fill_method=None)
        out["nifty50_log_ret_1d"] = np.log(px / px.shift(1))

    if "nifty_midcap" in df.columns:
        mc = df["nifty_midcap"]
        out["nifty_midcap_ret_1d"] = mc.pct_change(1, fill_method=None)
        out["nifty_midcap_ret_21d"] = mc.pct_change(21, fill_method=None)

    if "nifty_bank" in df.columns:
        bk = df["nifty_bank"]
        out["nifty_bank_ret_1d"] = bk.pct_change(1, fill_method=None)
        out["nifty_bank_ret_21d"] = bk.pct_change(21, fill_method=None)

    return out
