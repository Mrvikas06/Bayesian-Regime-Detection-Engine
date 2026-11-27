"""
Phase 3 pytest tests: Frequentist HMM, BIC comparison, label stability, OOS calibration.

Run with: pytest tests/test_phase3.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from regime.data.labels import REGIME_NAMES
from regime.models.hmm_freq import load_config


@pytest.fixture(scope="module")
def models_cfg():
    return load_config(Path(__file__).parent.parent / "configs" / "models.yaml")


@pytest.fixture(scope="module")
def hmm_probs(models_cfg):
    path = Path(models_cfg["output_dir"]) / "hmm_freq_probs.parquet"
    if not path.exists():
        pytest.skip("hmm_freq_probs.parquet missing — run `make phase-3` first")
    return pd.read_parquet(path)


class TestHMMOutputs:
    def test_probs_sum_to_one(self, hmm_probs):
        s = hmm_probs[REGIME_NAMES].sum(axis=1)
        assert np.allclose(s.values, 1.0, atol=1e-5)

    def test_probs_non_negative(self, hmm_probs):
        assert (hmm_probs[REGIME_NAMES].values >= 0.0).all()

    def test_predicted_regime_valid(self, hmm_probs):
        assert set(hmm_probs["predicted_regime"].unique()).issubset(set(REGIME_NAMES))

    def test_lineage_attached(self, hmm_probs):
        for col in ("lineage_data_version_hash", "lineage_model_version", "lineage_config_hash"):
            assert col in hmm_probs.columns


class TestPhase3Artefacts:
    def test_bic_table_has_3_5_7(self):
        df = pd.read_csv("reports/tables/phase3_hmm_bic.csv")
        assert list(df["n_states"]) == [3, 5, 7]
        assert df["bic"].notna().all()

    def test_transition_matrix_rows_sum_to_one(self):
        df = pd.read_csv("reports/tables/phase3_hmm_transitions.csv", index_col=0)
        row_sums = df[REGIME_NAMES].sum(axis=1)
        assert np.allclose(row_sums.values, 1.0, atol=1e-3)
        assert (df["expected_duration_days"] > 1.0).all()

    def test_seed_stability_passes(self):
        with open("data/lineage/phase3_lineage.json") as f:
            lin = json.load(f)
        assert lin["seed_stability"]["mean_ari"] >= 0.65
        assert lin["oos_metrics"]["n_eval"] > 500

    def test_overlay_figure_exists(self):
        assert Path("reports/figures/phase3_hmm_overlay.png").exists()
