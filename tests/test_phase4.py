"""
Phase 4 pytest tests: Bayesian HMM PyTensor Op gradients, MCMC diagnostics (Rule 4),
credible intervals, and Phase 3 comparison artefacts.

Run with: pytest tests/test_phase4.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytensor
import pytensor.tensor as pt
import pytest
from scipy.optimize import check_grad

from regime.data.labels import REGIME_NAMES
from regime.models.hmm_bayes import hmm_forward_op
from regime.models.hmm_freq import load_config


@pytest.fixture(scope="module")
def models_cfg():
    return load_config(Path(__file__).parent.parent / "configs" / "models.yaml")


@pytest.fixture(scope="module")
def bayes_probs(models_cfg):
    path = Path(models_cfg["output_dir"]) / "hmm_bayes_probs.parquet"
    if not path.exists():
        pytest.skip("hmm_bayes_probs.parquet missing — run `make phase-4` first")
    return pd.read_parquet(path)


class TestHMMForwardOpGradient:
    def test_analytical_gradient_matches_finite_diff(self):
        rng = np.random.default_rng(42)
        T, K = 15, 4
        log_emit_val = rng.normal(size=(T, K))
        P_raw = rng.dirichlet(np.ones(K) * 3.0, size=K)
        log_P_val = np.log(P_raw)
        log_pi0_val = np.log(rng.dirichlet(np.ones(K)))

        le_sym = pt.dmatrix("le")
        lp_sym = pt.dmatrix("lp")
        lpi_sym = pt.dvector("lpi")
        out = hmm_forward_op(le_sym, lp_sym, lpi_sym)
        g_le, g_lp, g_lpi = pt.grad(out, [le_sym, lp_sym, lpi_sym])
        fn = pytensor.function([le_sym, lp_sym, lpi_sym], [out, g_le, g_lp, g_lpi])

        def f_le(x):
            return float(fn(x.reshape(T, K), log_P_val, log_pi0_val)[0])

        def grad_le(x):
            return fn(x.reshape(T, K), log_P_val, log_pi0_val)[1].ravel()

        err_le = check_grad(f_le, grad_le, log_emit_val.ravel(), epsilon=1e-6)
        assert err_le < 1e-4, f"log_emit grad mismatch: {err_le}"

        def f_lp(x):
            return float(fn(log_emit_val, x.reshape(K, K), log_pi0_val)[0])

        def grad_lp(x):
            return fn(log_emit_val, x.reshape(K, K), log_pi0_val)[2].ravel()

        err_lp = check_grad(f_lp, grad_lp, log_P_val.ravel(), epsilon=1e-6)
        assert err_lp < 1e-4, f"log_P grad mismatch: {err_lp}"


class TestMCMCDiagnosticsRule4:
    def test_rhat_and_divergences_pass(self):
        path = Path("data/lineage/phase4_lineage.json")
        if not path.exists():
            pytest.skip("phase4_lineage.json missing — run `make phase-4` first")
        with open(path) as f:
            lin = json.load(f)
        diag = lin["mcmc_diagnostics"]
        assert diag["max_rhat"] <= 1.01, f"R-hat too high: {diag['max_rhat']}"
        assert diag["n_divergences"] == 0, f"Divergences > 0: {diag['n_divergences']}"
        assert diag["min_ess_bulk"] >= 400, f"Low bulk ESS: {diag['min_ess_bulk']}"


class TestBayesianHMMOutputs:
    def test_probs_sum_to_one(self, bayes_probs):
        s = bayes_probs[REGIME_NAMES].sum(axis=1)
        assert np.allclose(s.values, 1.0, atol=1e-4)

    def test_credible_intervals_ordered(self, bayes_probs):
        for r in REGIME_NAMES:
            assert (bayes_probs[f"{r}_q05"] <= bayes_probs[f"{r}_q95"] + 1e-9).all()
            assert (bayes_probs[f"{r}_q05"] >= -1e-9).all()
            assert (bayes_probs[f"{r}_q95"] <= 1.0 + 1e-9).all()

    def test_artefacts_exist(self):
        assert Path("reports/tables/phase4_mcmc_diagnostics.csv").exists()
        assert Path("reports/tables/phase4_transitions_ci.csv").exists()
        assert Path("reports/figures/phase4_trace_plots.png").exists()
        assert Path("reports/figures/phase4_vs_phase3_comparison.png").exists()
