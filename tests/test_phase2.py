"""
Phase 2 pytest tests: 30+ causal features, no-leakage, optional TDA/GNN, lineage.

Run with: pytest tests/test_phase2.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from regime.features.build import (
    build_feature_matrix,
    check_causal_perturbation,
    check_future_correlation,
    load_config,
)


@pytest.fixture(scope="module")
def feat_cfg():
    return load_config(Path(__file__).parent.parent / "configs" / "features.yaml")


@pytest.fixture(scope="module")
def master_df():
    path = Path("data/processed/master.parquet")
    if not path.exists():
        pytest.skip("master.parquet missing — run `make phase-1` first")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def features_df(feat_cfg):
    path = Path(feat_cfg["output_parquet"])
    if not path.exists():
        pytest.skip("features.parquet missing — run `make phase-2` first")
    return pd.read_parquet(path)


class TestFeatureMatrix:
    def test_feature_count_at_least_30(self, features_df):
        non_meta = [
            c for c in features_df.columns
            if c != "regime_label" and not c.startswith("lineage_")
        ]
        assert len(non_meta) >= 30, f"Expected >= 30 features, got {len(non_meta)}"

    def test_no_infinite_values(self, features_df):
        num = features_df.select_dtypes(include=[np.number])
        assert np.isfinite(num.dropna()).all().all(), "Infinite values found in features"

    def test_index_aligned_with_master(self, master_df, features_df):
        assert len(features_df) == len(master_df)
        assert (features_df.index == master_df.index).all()

    def test_lineage_columns_present(self, features_df):
        for col in ("lineage_data_version_hash", "lineage_feature_set_version", "lineage_config_hash"):
            assert col in features_df.columns, f"Missing lineage column: {col}"


class TestNoLeakage:
    def test_causal_perturbation(self, master_df, feat_cfg):
        """Perturbing future data (> cutoff) must not alter features <= cutoff."""
        assert check_causal_perturbation(master_df, feat_cfg, cutoff_date="2022-01-03")

    def test_future_return_correlation(self, features_df, feat_cfg):
        leak_cfg = feat_cfg.get("leakage", {})
        flagged = check_future_correlation(
            features_df,
            ref_col=leak_cfg.get("reference_return", "nifty50_ret_1d"),
            max_corr=float(leak_cfg.get("max_future_correlation", 0.05)),
        )
        assert flagged.empty, f"Features correlated with t+1 return: {flagged.to_dict('records')}"


class TestOptionalModules:
    def test_tda_disabled_does_not_break(self, master_df, feat_cfg):
        cfg_copy = dict(feat_cfg)
        cfg_copy["tda"] = {"use_tda": False}
        feats, _ = build_feature_matrix(master_df, cfg_copy)
        assert not feats.empty

    def test_tda_enabled_runs_cleanly(self, master_df, feat_cfg):
        cfg_copy = dict(feat_cfg)
        cfg_copy["tda"] = {"use_tda": True, "rolling_corr_window": 63}
        feats, _ = build_feature_matrix(master_df.iloc[-300:], cfg_copy)
        assert "tda_corr_frobenius" in feats.columns
        assert "tda_spectral_gap" in feats.columns


class TestPhase2Artefacts:
    def test_lineage_json_exists(self, feat_cfg):
        path = Path(feat_cfg["lineage_json"])
        assert path.exists()
        with open(path) as f:
            lin = json.load(f)
        assert lin["phase"] == 2
        assert lin["n_features"] >= 30

    def test_breakpoint_plot_exists(self):
        assert Path("reports/figures/phase2_features_breakpoints.png").exists()
