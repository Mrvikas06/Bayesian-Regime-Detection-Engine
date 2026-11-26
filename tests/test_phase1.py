"""
Phase 1 pytest tests: data ingestion, quality, versioning, labels.

Run with: pytest tests/test_phase1.py -v
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cfg():
    cfg_path = Path(__file__).parent.parent / "configs" / "data.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def master_df(cfg):
    """Load master parquet if it exists, else skip dependent tests."""
    path = Path(cfg["output"]["master_parquet"])
    if not path.exists():
        pytest.skip(f"master parquet not found at {path}; run `make phase-1` first")
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Config sanity
# ---------------------------------------------------------------------------

class TestConfig:
    def test_start_date_before_end(self, cfg):
        assert cfg["start_date"] < cfg["end_date"]

    def test_min_history_years_reasonable(self, cfg):
        assert 5 <= cfg["quality"]["min_history_years"] <= 20

    def test_required_tickers_present(self, cfg):
        assert "nifty50" in cfg["indices"]
        assert "india_vix" in cfg["indices"]

    def test_output_paths_defined(self, cfg):
        assert "master_parquet" in cfg["output"]
        assert "quality_report" in cfg["output"]


# ---------------------------------------------------------------------------
# Master parquet structure
# ---------------------------------------------------------------------------

class TestMasterParquet:
    def test_parquet_exists(self, cfg):
        assert Path(cfg["output"]["master_parquet"]).exists(), (
            "Master parquet missing — run `make phase-1`"
        )

    def test_no_duplicate_index(self, master_df):
        assert not master_df.index.duplicated().any(), "Duplicate dates in master index"

    def test_index_sorted(self, master_df):
        assert master_df.index.is_monotonic_increasing, "Index not sorted"

    def test_no_weekend_rows(self, master_df):
        dow = master_df.index.dayofweek
        assert (dow < 5).all(), f"Weekend rows found: {master_df.index[dow >= 5][:5].tolist()}"

    def test_minimum_rows(self, master_df):
        """At least 10 years of daily data = ~2500 rows."""
        assert len(master_df) >= 2500, f"Only {len(master_df)} rows — expected ≥ 2500"

    def test_nifty50_mostly_present(self, master_df):
        if "nifty50" in master_df.columns:
            pct_missing = master_df["nifty50"].isna().mean()
            assert pct_missing < 0.05, f"nifty50 missing {pct_missing:.1%} of rows"

    def test_no_negative_prices(self, master_df):
        price_cols = [c for c in master_df.columns
                      if any(x in c for x in ["nifty", "india_vix"])]
        for col in price_cols:
            s = master_df[col].dropna()
            assert (s > 0).all(), f"Non-positive values in {col}"

    def test_vix_in_reasonable_range(self, master_df):
        if "india_vix" in master_df.columns:
            vix = master_df["india_vix"].dropna()
            assert vix.max() < 100, "VIX > 100 — likely a data error"
            assert vix.min() > 0, "VIX <= 0 — data error"


# ---------------------------------------------------------------------------
# Lineage
# ---------------------------------------------------------------------------

class TestLineage:
    def test_lineage_json_exists(self, cfg):
        path = Path(cfg["output"]["lineage_json"])
        assert path.exists(), f"Lineage JSON missing at {path}"

    def test_lineage_has_required_fields(self, cfg):
        with open(cfg["output"]["lineage_json"]) as f:
            lineage = json.load(f)
        for field in ["phase", "timestamp_utc", "data_version_hash", "shape"]:
            assert field in lineage, f"Missing lineage field: {field}"

    def test_lineage_shape_matches_parquet(self, cfg, master_df):
        with open(cfg["output"]["lineage_json"]) as f:
            lineage = json.load(f)
        assert lineage["shape"]["rows"] == len(master_df)
        assert lineage["shape"]["cols"] == master_df.shape[1]


# ---------------------------------------------------------------------------
# Reference Labels
# ---------------------------------------------------------------------------

class TestReferenceLabels:
    def test_labels_cover_all_regimes(self, master_df):
        from regime.data.labels import label_regimes, REGIME_NAMES  # noqa: PLC0415
        labels = label_regimes(master_df)
        # At least 3 of the 5 regimes should appear in a 15-year history
        present = set(labels.unique())
        assert len(present) >= 3, f"Only {present} regimes appeared"

    def test_labels_no_nan(self, master_df):
        from regime.data.labels import label_regimes  # noqa: PLC0415
        labels = label_regimes(master_df)
        assert labels.notna().all(), "NaN in reference labels"

    def test_labels_valid_strings(self, master_df):
        from regime.data.labels import label_regimes, REGIME_NAMES  # noqa: PLC0415
        labels = label_regimes(master_df)
        assert set(labels.unique()).issubset(set(REGIME_NAMES)), (
            f"Unknown regime names: {set(labels.unique()) - set(REGIME_NAMES)}"
        )


# ---------------------------------------------------------------------------
# Versioning utilities
# ---------------------------------------------------------------------------

class TestVersioning:
    def test_hash_dataframe_deterministic(self, master_df):
        from regime.data.versioning import hash_dataframe  # noqa: PLC0415
        h1 = hash_dataframe(master_df)
        h2 = hash_dataframe(master_df)
        assert h1 == h2

    def test_hash_config_deterministic(self, cfg):
        from regime.data.versioning import hash_config  # noqa: PLC0415
        h1 = hash_config(cfg)
        h2 = hash_config(cfg)
        assert h1 == h2

    def test_hash_length(self, master_df):
        from regime.data.versioning import hash_dataframe  # noqa: PLC0415
        h = hash_dataframe(master_df)
        assert len(h) == 16
