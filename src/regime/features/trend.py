"""
regime.features.trend
=====================
Causal trend features: DMA ratios, golden cross, RSI, and rolling drawdown.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def _compute_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Wilder-style EMA RSI, strictly causal."""
    delta = prices.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def compute_trend_features(
    df: pd.DataFrame,
    dma_windows: Sequence[int] = (20, 50, 100, 200),
    rsi_window: int = 14,
) -> pd.DataFrame:
    """Compute DMA ratios, golden cross, RSI, and 252d drawdown."""
    out = pd.DataFrame(index=df.index)

    if "nifty50" in df.columns:
        px = df["nifty50"]
        dmas: dict[int, pd.Series] = {}
        for w in dma_windows:
            dma = px.rolling(w, min_periods=max(1, w // 2)).mean()
            dmas[w] = dma
            out[f"nifty50_vs_dma{w}"] = px / dma - 1.0

        if 50 in dmas and 200 in dmas:
            out["golden_cross_50_200"] = (dmas[50] > dmas[200]).astype(float)
            out["dma50_vs_dma200"] = dmas[50] / dmas[200] - 1.0

        out[f"nifty50_rsi{rsi_window}"] = _compute_rsi(px, window=rsi_window)
        rolling_max = px.rolling(252, min_periods=63).max()
        out["nifty50_drawdown_252d"] = px / rolling_max - 1.0

    if "nifty_midcap" in df.columns:
        mc = df["nifty_midcap"]
        for w in (50, 200):
            dma = mc.rolling(w, min_periods=max(1, w // 2)).mean()
            out[f"nifty_midcap_vs_dma{w}"] = mc / dma - 1.0

    if "nifty_bank" in df.columns:
        bk = df["nifty_bank"]
        for w in (50, 200):
            dma = bk.rolling(w, min_periods=max(1, w // 2)).mean()
            out[f"nifty_bank_vs_dma{w}"] = bk / dma - 1.0

    return out
