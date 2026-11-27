# Experiment Log

> All model runs are also logged to MLflow (`mlflow ui --port 5000`).
> This file captures human-readable summaries and key findings.

---

## Phase 1 — Data Ingestion

| Date | Action | Result |
|------|--------|--------|
| 2026-11-26 | Initial data pull (yfinance + manual CSVs) | 5,190 rows × 7 cols; quality report in `reports/data_quality_report.html` |

---

## Phase 2 — Feature Engineering

| Date | Action | Result |
|------|--------|--------|
| 2026-11-27 | Built 50 causal features + leakage tests | 0 future-corr flags, causal perturbation PASSED; 31/31 tests pass |

---

## Phase 3 — Frequentist HMM

| Model | OOS Period | Log-Loss | Brier | ECE | Weak-Label Acc | Mean Seed ARI |
|-------|------------|----------|-------|-----|----------------|---------------|
| `hmm_freq` (5-state, full cov) | 2019–2026 (n=2,060) | 5.9772 | 1.1041 | 0.5431 | 0.4223 | 0.9145 |

- BIC comparison (N=4,885): K=3 (`40,747.03`), K=5 (`35,680.31`), K=7 (`32,685.41`).
- Note: OOS log-loss/ECE is uncalibrated (raw Gaussian HMM likelihoods are sharp); Phase 10 conformal/calibration will address this. MLflow local run hit `_cffi_backend` import issue on Python 3.14 Windows.

---

## Phase 4 — Bayesian HMM (PyMC)

| Model | OOS Period | Log-Loss | Brier | ECE | Weak-Label Acc | Max R-hat | Min ESS (Bulk/Tail) | Divergences |
|-------|------------|----------|-------|-----|----------------|-----------|---------------------|-------------|
| `hmm_bayes` (5-state, NUTS 4×2000) | 2019–2026 (n=2,060) | 6.6222 | 1.2322 | 0.6061 | 0.3505 | 1.0025 | 2,040.3 / 1,668.0 | 0 |
