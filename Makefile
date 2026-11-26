# ===========================================================================
#  Bayesian Regime Detection Engine — Makefile
#  Every phase is reproducible with a single `make phase-N`
# ===========================================================================

PYTHON  := python
PYTEST  := pytest
RUFF    := ruff
BLACK   := black
MLFLOW  := mlflow

.PHONY: all env lint test clean \
        phase-1 phase-2 phase-3 phase-4 phase-5 phase-6 phase-7 \
        phase-8 phase-9 phase-10 phase-11 phase-12 phase-13 phase-14 phase-15

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
env:
	conda env create -f environment.yml --force
	conda run -n regime pip install -e .

# ---------------------------------------------------------------------------
# Code quality
# ---------------------------------------------------------------------------
lint:
	$(RUFF) check src/ tests/
	$(BLACK) --check src/ tests/

format:
	$(BLACK) src/ tests/
	$(RUFF) check --fix src/ tests/

test:
	$(PYTEST) tests/ -v

# ---------------------------------------------------------------------------
# Phase targets
# ---------------------------------------------------------------------------
phase-1:
	@echo "=== Phase 1: Data Ingestion & Quality ==="
	$(PYTHON) -m regime.data.ingest --config configs/data.yaml
	$(PYTHON) -m regime.data.quality --config configs/data.yaml
	$(PYTEST) tests/test_phase1.py -v
	@echo "Phase 1 DONE. Check data/processed/ and reports/"

phase-2:
	@echo "=== Phase 2: Feature Engineering ==="
	$(PYTHON) -m regime.features.build --config configs/features.yaml
	$(PYTEST) tests/test_phase2.py -v
	@echo "Phase 2 DONE."

phase-3:
	@echo "=== Phase 3: Frequentist HMM ==="
	$(PYTHON) -m regime.models.hmm_freq --config configs/models.yaml
	$(PYTEST) tests/test_phase3.py -v

phase-4:
	@echo "=== Phase 4: Bayesian HMM ==="
	$(PYTHON) -m regime.models.hmm_bayes --config configs/models.yaml
	$(PYTEST) tests/test_phase4.py -v

phase-5:
	@echo "=== Phase 5: Regime-Switching VAR ==="
	$(PYTHON) -m regime.models.rs_var --config configs/models.yaml
	$(PYTEST) tests/test_phase5.py -v

phase-6:
	@echo "=== Phase 6: Bayesian Deep Learning ==="
	$(PYTHON) -m regime.models.bdl --config configs/models.yaml
	$(PYTEST) tests/test_phase6.py -v

phase-7:
	@echo "=== Phase 7: Foundation Model Hybrids ==="
	$(PYTHON) -m regime.models.foundation --config configs/models.yaml
	$(PYTEST) tests/test_phase7.py -v

phase-8:
	@echo "=== Phase 8: Ensemble ==="
	$(PYTHON) -m regime.ensemble.build --config configs/models.yaml
	$(PYTEST) tests/test_phase8.py -v

phase-9:
	@echo "=== Phase 9: Online Inference ==="
	$(PYTHON) -m regime.online.stream --config configs/online.yaml
	$(PYTEST) tests/test_phase9.py -v

phase-10:
	@echo "=== Phase 10: Conformal & Calibration ==="
	$(PYTHON) -m regime.calibration.run --config configs/conformal.yaml
	$(PYTEST) tests/test_phase10.py -v

phase-11:
	@echo "=== Phase 11: Case Studies ==="
	$(PYTHON) -m regime.backtest.case_studies --config configs/backtest.yaml

phase-12:
	@echo "=== Phase 12: Backtest & Monte Carlo ==="
	$(PYTHON) -m regime.montecarlo.run --config configs/backtest.yaml
	$(PYTHON) -m regime.backtest.run --config configs/backtest.yaml
	$(PYTEST) tests/test_phase12.py -v

phase-13:
	@echo "=== Phase 13: R Cross-checks ==="
	Rscript r/run_all.R

phase-14:
	@echo "=== Phase 14: Reports & Model Card ==="
	$(PYTHON) -m regime.audit.report --config configs/data.yaml
	$(PYTHON) reports/build_report.py

phase-15:
	@echo "=== Phase 15: Final Validation ==="
	$(PYTEST) tests/ -v --tb=long
	$(PYTHON) -m regime.audit.validate_all

# ---------------------------------------------------------------------------
# Full rebuild
# ---------------------------------------------------------------------------
all: phase-1 phase-2 phase-3 phase-4 phase-5 phase-6 phase-7 \
     phase-8 phase-9 phase-10 phase-11 phase-12 phase-13 phase-14 phase-15

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	rm -rf .pytest_cache .ruff_cache .mypy_cache
