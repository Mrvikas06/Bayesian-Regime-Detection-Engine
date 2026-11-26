"""
regime.data.ingest
==================
Phase 1 data ingestion module.

Downloads index, macro, and flow data; aligns calendars; saves a versioned
master parquet with a lineage JSON record.

Usage
-----
    python -m regime.data.ingest --config configs/data.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _hash_df(df: pd.DataFrame) -> str:
    """SHA-256 of the parquet bytes — stable across runs given same data."""
    buf = df.to_parquet(index=True)
    return hashlib.sha256(buf).hexdigest()[:16]


def _safe_yf_download(
    ticker: str,
    start: str,
    end: str,
    max_retries: int = 3,
    delay: float = 2.0,
) -> pd.DataFrame:
    """
    Download OHLCV from yfinance with retry logic.

    Returns a flat-column DataFrame with standard OHLCV columns,
    compatible with yfinance ≥ 1.8 (which uses MultiIndex columns).
    """
    import yfinance as yf  # local import — not needed if disabled

    for attempt in range(1, max_retries + 1):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                multi_level_index=False,  # yfinance ≥1.8: flatten to single-level
            )
            if df.empty:
                logger.warning("Empty response for %s (attempt %d/%d)", ticker, attempt, max_retries)
            else:
                # yfinance ≥ 1.8 may still return MultiIndex; flatten if needed
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                return df
        except TypeError:
            # Older yfinance does not accept multi_level_index; retry without it
            try:
                df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
                if not df.empty:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    return df
            except Exception as exc2:  # noqa: BLE001
                logger.warning("Fallback download error for %s: %s", ticker, exc2)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Download error for %s attempt %d: %s", ticker, attempt, exc)
        if attempt < max_retries:
            time.sleep(delay)
    logger.error("All %d attempts failed for %s — returning empty DataFrame", max_retries, ticker)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Per-series loaders
# ---------------------------------------------------------------------------

def _load_index_series(cfg: dict[str, Any], start: str, end: str) -> dict[str, pd.Series]:
    """Download close prices for all configured index tickers."""
    series: dict[str, pd.Series] = {}
    for key, spec in cfg.get("indices", {}).items():
        ticker = spec["ticker"]
        logger.info("Downloading %s (%s) …", key, ticker)
        raw = _safe_yf_download(ticker, start, end)
        if raw.empty:
            logger.error("MISSING: %s (%s) — no data returned", key, ticker)
            series[key] = pd.Series(dtype=float, name=key)
            continue
        # yfinance returns MultiIndex columns when downloading multiple tickers;
        # here we download one at a time so columns are flat.
        close_col = "Close" if "Close" in raw.columns else raw.columns[0]
        s = raw[close_col].rename(key)
        if spec.get("is_proxy"):
            logger.warning(
                "PROXY: %s — %s", key, spec.get("proxy_note", "see configs/data.yaml")
            )
        series[key] = s
    return series


def _load_macro_series(cfg: dict[str, Any], start: str, end: str) -> dict[str, pd.Series]:
    """Download macro / fixed-income tickers."""
    series: dict[str, pd.Series] = {}
    for key, spec in cfg.get("macro", {}).items():
        ticker = spec["ticker"]
        fallback_csv = spec.get("fallback_csv")
        logger.info("Downloading macro: %s (%s) …", key, ticker)
        raw = _safe_yf_download(ticker, start, end)
        if raw.empty and fallback_csv and Path(fallback_csv).exists():
            logger.warning(
                "yfinance failed for %s — loading from fallback CSV: %s", key, fallback_csv
            )
            df_fb = pd.read_csv(fallback_csv, parse_dates=["date"], index_col="date")
            s = df_fb.iloc[:, 0].rename(key)
        elif raw.empty:
            logger.error(
                "MISSING: %s — yfinance empty and no fallback CSV at %s", key, fallback_csv
            )
            s = pd.Series(dtype=float, name=key)
        else:
            close_col = "Close" if "Close" in raw.columns else raw.columns[0]
            s = raw[close_col].rename(key)
        series[key] = s
    return series


def _load_flow_series(cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Load FII/DII and SIP flows from manual CSVs."""
    result: dict[str, pd.DataFrame] = {}
    flows_cfg = cfg.get("flows", {})

    # FII / DII
    fii_cfg = flows_cfg.get("fii_dii", {})
    fii_path = fii_cfg.get("csv_path", "")
    if Path(fii_path).exists():
        df = pd.read_csv(fii_path, parse_dates=["date"], index_col="date")
        df.index = pd.to_datetime(df.index)
        result["fii_dii"] = df
        logger.info("Loaded FII/DII flows: %d rows from %s", len(df), fii_path)
    else:
        logger.warning(
            "FII/DII CSV not found at %s — %s", fii_path, fii_cfg.get("fallback_note", "")
        )
        result["fii_dii"] = pd.DataFrame()

    # SIP
    sip_cfg = flows_cfg.get("sip", {})
    sip_path = sip_cfg.get("csv_path", "")
    if Path(sip_path).exists():
        df = pd.read_csv(sip_path, parse_dates=["date"], index_col="date")
        df.index = pd.to_datetime(df.index)
        result["sip"] = df
        logger.info("Loaded SIP data: %d rows from %s", len(df), sip_path)
    else:
        logger.warning(
            "SIP CSV not found at %s — %s", sip_path, sip_cfg.get("fallback_note", "")
        )
        result["sip"] = pd.DataFrame()

    return result


def _load_credit_spread(cfg: dict[str, Any]) -> pd.Series:
    """Load AAA-Gilt credit spread from manual CSV, or return empty Series."""
    cs_cfg = cfg.get("credit_spread", {})
    csv_path = cs_cfg.get("csv_path", "")
    if Path(csv_path).exists():
        df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
        s = df.iloc[:, 0].rename("credit_spread_aaa_gilt")
        logger.info("Loaded credit spread: %d rows", len(s))
        return s
    logger.warning(
        "Credit spread CSV not found at %s — %s", csv_path, cs_cfg.get("note", "")
    )
    return pd.Series(dtype=float, name="credit_spread_aaa_gilt")


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build_master(cfg: dict[str, Any]) -> pd.DataFrame:
    """
    Download all configured series, align on NSE trading calendar,
    forward-fill up to 5 days, and assemble into a single wide DataFrame.
    """
    start = cfg["start_date"]
    end = cfg["end_date"]

    # --- Download ---
    idx_series = _load_index_series(cfg, start, end)
    macro_series = _load_macro_series(cfg, start, end)
    flow_data = _load_flow_series(cfg)
    cs_series = _load_credit_spread(cfg)

    # --- Combine into wide frame ---
    all_series: list[pd.Series] = []
    for s in idx_series.values():
        if not s.empty:
            all_series.append(s)
    for s in macro_series.values():
        if not s.empty:
            all_series.append(s)

    if all_series:
        master = pd.concat(all_series, axis=1)
    else:
        master = pd.DataFrame()

    # Merge FII/DII flows
    if not flow_data["fii_dii"].empty:
        # Rename columns to avoid collisions
        fii_df = flow_data["fii_dii"].add_prefix("fii_dii_")
        master = master.merge(fii_df, left_index=True, right_index=True, how="left")

    # Merge SIP (monthly → forward fill to daily)
    if not flow_data["sip"].empty:
        sip_df = flow_data["sip"].add_prefix("sip_")
        master = master.merge(sip_df, left_index=True, right_index=True, how="left")

    # Merge credit spread
    if not cs_series.empty:
        master = master.merge(cs_series, left_index=True, right_index=True, how="left")

    if master.empty:
        logger.error("Master DataFrame is empty — check network / manual CSV placement.")
        return master

    # --- Calendar alignment: keep only weekdays, sort index ---
    master.index = pd.to_datetime(master.index)
    master = master[master.index.dayofweek < 5].sort_index()
    master = master.loc[start:end]

    # --- Forward-fill gaps up to 5 trading days (holiday fill) ---
    master = master.ffill(limit=5)

    logger.info(
        "Master DataFrame assembled: %d rows × %d cols  (%s to %s)",
        len(master),
        master.shape[1],
        master.index.min().date(),
        master.index.max().date(),
    )
    return master


# ---------------------------------------------------------------------------
# Lineage
# ---------------------------------------------------------------------------

def write_lineage(
    cfg: dict[str, Any],
    master: pd.DataFrame,
    output_path: str,
    lineage_path: str,
) -> None:
    """Write a JSON lineage record for the master dataset."""
    lineage: dict[str, Any] = {
        "phase": 1,
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "data_version_hash": _hash_df(master) if not master.empty else "EMPTY",
        "config": {
            "start_date": cfg["start_date"],
            "end_date": cfg["end_date"],
        },
        "output_path": output_path,
        "shape": {"rows": len(master), "cols": master.shape[1]},
        "columns": list(master.columns),
        "proxies": [],
        "missing_series": [],
    }

    # Document proxies
    for key, spec in cfg.get("indices", {}).items():
        if spec.get("is_proxy"):
            lineage["proxies"].append({"series": key, "note": spec.get("proxy_note", "")})
    for key, spec in cfg.get("macro", {}).items():
        if spec.get("is_proxy"):
            lineage["proxies"].append({"series": key, "note": spec.get("proxy_note", "")})

    # Document missing (all-NaN) columns
    if not master.empty:
        for col in master.columns:
            if master[col].isna().all():
                lineage["missing_series"].append(col)

    Path(lineage_path).parent.mkdir(parents=True, exist_ok=True)
    with open(lineage_path, "w") as f:
        json.dump(lineage, f, indent=2)
    logger.info("Lineage written to %s", lineage_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1: Data Ingestion")
    parser.add_argument("--config", default="configs/data.yaml", help="Path to data config YAML")
    args = parser.parse_args()

    cfg = _load_config(args.config)

    # Ensure output directories exist
    Path(cfg["processed_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["lineage_dir"]).mkdir(parents=True, exist_ok=True)
    Path("data/raw/manual").mkdir(parents=True, exist_ok=True)

    master = build_master(cfg)

    if master.empty:
        logger.error("No data downloaded. Check network connection and manual CSV placement.")
        return

    out_path = cfg["output"]["master_parquet"]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    master.to_parquet(out_path)
    logger.info("Master parquet saved: %s", out_path)

    write_lineage(
        cfg,
        master,
        output_path=out_path,
        lineage_path=cfg["output"]["lineage_json"],
    )
    logger.info("Phase 1 ingestion complete.")


if __name__ == "__main__":
    main()
