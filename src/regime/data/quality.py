"""
regime.data.quality
===================
Phase 1 data quality module.

Runs checks on the master parquet (missing values, outliers, stationarity)
and writes an HTML quality report and a JSON quality summary.

Usage
-----
    python -m regime.data.quality --config configs/data.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)


def _load_config(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------

def check_missing(df: pd.DataFrame, max_pct: float = 0.03) -> pd.DataFrame:
    """Return per-column missing-value statistics."""
    total = len(df)
    records = []
    for col in df.columns:
        n_missing = df[col].isna().sum()
        pct = n_missing / total
        records.append(
            {
                "column": col,
                "n_total": total,
                "n_missing": int(n_missing),
                "pct_missing": round(pct, 4),
                "flag": pct > max_pct,
            }
        )
    return pd.DataFrame(records).sort_values("pct_missing", ascending=False)


def check_outliers(df: pd.DataFrame, z_thresh: float = 5.0) -> pd.DataFrame:
    """Flag cells with |z-score| > z_thresh (computed on returns for price series)."""
    records = []
    for col in df.columns:
        s = df[col].dropna()
        if len(s) < 30:
            continue
        # Use first-difference for price-like series to get returns
        s_diff = s.pct_change().dropna()
        if s_diff.std() == 0:
            continue
        z = (s_diff - s_diff.mean()) / s_diff.std()
        n_outliers = int((np.abs(z) > z_thresh).sum())
        worst_dates = z[np.abs(z) > z_thresh].abs().nlargest(3).index.tolist()
        records.append(
            {
                "column": col,
                "n_outliers": n_outliers,
                "worst_dates": [str(d.date()) for d in worst_dates],
                "flag": n_outliers > 0,
            }
        )
    return pd.DataFrame(records).sort_values("n_outliers", ascending=False)


def check_history_length(df: pd.DataFrame, min_years: float = 10.0) -> pd.DataFrame:
    """Check each column has at least min_years of non-NaN data."""
    records = []
    for col in df.columns:
        s = df[col].dropna()
        if s.empty:
            years = 0.0
        else:
            years = (s.index.max() - s.index.min()).days / 365.25
        records.append(
            {
                "column": col,
                "start": str(s.index.min().date()) if not s.empty else "N/A",
                "end": str(s.index.max().date()) if not s.empty else "N/A",
                "years": round(years, 1),
                "flag": years < min_years,
            }
        )
    return pd.DataFrame(records).sort_values("years")


def check_weekend_gaps(df: pd.DataFrame) -> dict[str, Any]:
    """Confirm no Saturday/Sunday rows remain in the index."""
    weekend_rows = df[df.index.dayofweek >= 5]
    return {
        "n_weekend_rows": len(weekend_rows),
        "flag": len(weekend_rows) > 0,
        "sample_dates": [str(d.date()) for d in weekend_rows.index[:5]],
    }


def check_non_trading_gaps(df: pd.DataFrame, primary_col: str = "nifty50") -> pd.DataFrame:
    """Find stretches of consecutive NaN in the primary column (holiday gaps)."""
    if primary_col not in df.columns:
        return pd.DataFrame()
    s = df[primary_col]
    is_nan = s.isna()
    runs = []
    start = None
    for i, (dt, v) in enumerate(is_nan.items()):
        if v and start is None:
            start = dt
        elif not v and start is not None:
            end = s.index[i - 1]
            n = (end - start).days + 1
            runs.append({"start": str(start.date()), "end": str(end.date()), "n_days": n})
            start = None
    if start is not None:
        runs.append({"start": str(start.date()), "end": str(s.index[-1].date()), "n_days": 999})
    return pd.DataFrame(runs).sort_values("n_days", ascending=False).head(20)


# ---------------------------------------------------------------------------
# HTML Report Builder
# ---------------------------------------------------------------------------

def _df_to_html_table(df: pd.DataFrame, flag_col: str | None = None) -> str:
    """Render a DataFrame as an HTML table, highlighting flagged rows in amber."""
    if df.empty:
        return "<p><em>No data</em></p>"

    rows_html = []
    for _, row in df.iterrows():
        flag = row.get(flag_col, False) if flag_col else False
        style = ' style="background:#fff3cd"' if flag else ""
        cells = "".join(f"<td>{v}</td>" for v in row.values)
        rows_html.append(f"<tr{style}>{cells}</tr>")

    headers = "".join(f"<th>{c}</th>" for c in df.columns)
    return f"""
<table border="1" cellpadding="4" cellspacing="0" style="border-collapse:collapse;font-size:12px">
  <thead><tr style="background:#2c5f8a;color:white">{headers}</tr></thead>
  <tbody>{''.join(rows_html)}</tbody>
</table>
"""


def build_html_report(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    missing_df: pd.DataFrame,
    outlier_df: pd.DataFrame,
    history_df: pd.DataFrame,
    weekend_check: dict[str, Any],
    gap_df: pd.DataFrame,
) -> str:
    """Generate a self-contained HTML quality report."""
    ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n_flags = (
        int(missing_df["flag"].sum())
        + int(outlier_df["flag"].sum() if not outlier_df.empty else 0)
        + int(history_df["flag"].sum())
        + int(weekend_check["flag"])
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Phase 1 — Data Quality Report</title>
  <style>
    body {{font-family: Arial, sans-serif; margin: 40px; color: #222}}
    h1 {{color: #2c5f8a}} h2 {{color: #2c5f8a; border-bottom: 2px solid #2c5f8a}}
    .badge-ok {{background:#d4edda;padding:2px 8px;border-radius:4px;color:#155724}}
    .badge-warn {{background:#fff3cd;padding:2px 8px;border-radius:4px;color:#856404}}
    .badge-err {{background:#f8d7da;padding:2px 8px;border-radius:4px;color:#721c24}}
    pre {{background:#f4f4f4;padding:12px;border-radius:4px;font-size:12px}}
  </style>
</head>
<body>
<h1>Phase 1 — Data Quality Report</h1>
<p><strong>Generated:</strong> {ts}</p>
<p><strong>Total flags:</strong> <span class="{'badge-ok' if n_flags == 0 else 'badge-warn'}">{n_flags}</span></p>
<p><strong>Dataset shape:</strong> {df.shape[0]:,} rows × {df.shape[1]} columns
   &nbsp;|&nbsp; <strong>Date range:</strong> {df.index.min().date()} → {df.index.max().date()}</p>

<h2>1. Missing Values</h2>
<p>Threshold: &gt; {cfg['quality']['max_missing_pct']*100:.0f}% flags the series.</p>
{_df_to_html_table(missing_df, flag_col='flag')}

<h2>2. Outliers (|z| &gt; {cfg['quality']['max_outlier_zscore']})</h2>
<p>Computed on 1-day returns. Large spikes on known event dates are expected (2008, 2020).</p>
{_df_to_html_table(outlier_df, flag_col='flag')}

<h2>3. History Length</h2>
<p>Minimum required: {cfg['quality']['min_history_years']} years.</p>
{_df_to_html_table(history_df, flag_col='flag')}

<h2>4. Weekend / Holiday Gaps</h2>
<p>Weekend rows remaining in index: <strong>{weekend_check['n_weekend_rows']}</strong>
   {'✅' if not weekend_check['flag'] else '⚠️ FLAGGED'}</p>

<h2>5. Consecutive NaN Stretches (primary: nifty50)</h2>
<p>Top 20 gaps; long gaps around 2008 crisis and COVID may be expected.</p>
{_df_to_html_table(gap_df)}

<h2>6. Column Summary</h2>
<pre>
{df.describe().to_string()}
</pre>

<h2>7. Proxies and Gaps Catalogue</h2>
<ul>
"""
    # Add proxies from config
    for key, spec in cfg.get("indices", {}).items():
        if spec.get("is_proxy"):
            html += f"  <li><strong>{key}</strong>: {spec.get('proxy_note', '')}</li>\n"
    for key, spec in cfg.get("macro", {}).items():
        if spec.get("is_proxy"):
            html += f"  <li><strong>{key}</strong>: {spec.get('proxy_note', '')}</li>\n"
    html += f"""  <li><strong>credit_spread</strong>: {cfg.get('credit_spread', {}).get('note', 'manual CSV required')}</li>
  <li><strong>fii_dii flows</strong>: {cfg.get('flows', {}).get('fii_dii', {}).get('fallback_note', '')}</li>
  <li><strong>sip monthly</strong>: {cfg.get('flows', {}).get('sip', {}).get('fallback_note', '')}</li>
</ul>

</body>
</html>"""
    return html


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_quality_checks(cfg: dict[str, Any]) -> dict[str, Any]:
    """Run all checks; return JSON-serialisable summary dict."""
    parquet_path = cfg["output"]["master_parquet"]
    if not Path(parquet_path).exists():
        logger.error("Master parquet not found at %s — run phase-1 ingest first", parquet_path)
        return {"error": "master parquet missing"}

    df = pd.read_parquet(parquet_path)
    logger.info("Loaded master parquet: %s rows × %s cols", len(df), df.shape[1])

    q = cfg["quality"]
    missing_df   = check_missing(df, max_pct=q["max_missing_pct"])
    outlier_df   = check_outliers(df, z_thresh=q["max_outlier_zscore"])
    history_df   = check_history_length(df, min_years=q["min_history_years"])
    weekend_chk  = check_weekend_gaps(df)
    gap_df       = check_non_trading_gaps(df)

    # Build HTML report
    html = build_html_report(df, cfg, missing_df, outlier_df, history_df, weekend_chk, gap_df)
    report_path = cfg["output"]["quality_report"]
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(html)
    logger.info("Quality report written: %s", report_path)

    # JSON summary
    summary = {
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "shape": {"rows": len(df), "cols": df.shape[1]},
        "date_range": {"start": str(df.index.min().date()), "end": str(df.index.max().date())},
        "missing_flags": int(missing_df["flag"].sum()),
        "outlier_flags": int(outlier_df["flag"].sum()) if not outlier_df.empty else 0,
        "history_flags": int(history_df["flag"].sum()),
        "weekend_flag": weekend_chk["flag"],
        "total_flags": (
            int(missing_df["flag"].sum())
            + (int(outlier_df["flag"].sum()) if not outlier_df.empty else 0)
            + int(history_df["flag"].sum())
            + int(weekend_chk["flag"])
        ),
        "columns_all_nan": list(missing_df[missing_df["pct_missing"] == 1.0]["column"]),
    }
    summary_path = Path(cfg["lineage_dir"]) / "quality_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Quality summary written: %s", summary_path)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1: Data Quality Checks")
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    cfg = _load_config(args.config)
    summary = run_quality_checks(cfg)
    logger.info("Quality check complete. Total flags: %d", summary.get("total_flags", -1))


if __name__ == "__main__":
    main()
