# Bayesian Regime Detection Engine — Indian Equities

> **Internal use only. PRIVATE repository.**
> Multi-Asset Solutions Desk | Research & Quantitative Strategies

---

## Mission

Classify the Indian equity market into five probability-weighted regimes — **Risk-On, Risk-Off, Transitional, Late-Cycle, Post-Shock** — using an auditable, reproducible Bayesian ensemble. Output is calibrated regime probabilities with explicit uncertainty bounds, not index-level forecasts.

## Principles

| # | Principle |
|---|-----------|
| 1 | No fabricated results — every number comes from code run on real data |
| 2 | No time leakage — expanding-window splits, automated leakage test |
| 3 | Reproducible — fixed seeds, pinned env, config-driven, `make phase-N` |
| 4 | Diagnostic-gated Bayesian models (R-hat ≤ 1.01, zero divergences) |
| 5 | Every regime probability is conformalised with empirical coverage documented |
| 6 | Every ensemble member logged to MLflow before admission |
| 7 | Python × R cross-checks for all financial calculations |
| 8 | Full lineage on every output row |

## Quick Start

```bash
# 1. Create conda environment
conda env create -f environment.yml
conda activate regime

# 2. Install package in editable mode
pip install -e .

# 3. Run Phase 1 (data ingestion + quality report)
make phase-1

# 4. Run all phases end-to-end
make all
```

## Repo Structure

```
regime-engine/
  configs/          YAML config for each phase (data, features, models, conformal, backtest)
  data/
    raw/            Downloaded / manually placed raw files
    processed/      Versioned parquet outputs
    lineage/        JSON lineage records (data hash, feature hash, model hash, timestamp)
  src/regime/
    data/           Loaders, quality checks, versioning
    features/       Returns, trend, vol, breadth, flows, macro, TDA, GNN
    models/         HMM (freq + Bayes), RS-VAR, BDL, foundation hybrids
    ensemble/       BMA, stacking, WAIC/LOO, output contract (Pydantic)
    online/         Particle filter, BOCPD, streaming updates
    calibration/    Split-conformal, APS, ACI/EnbPI, ECE, reliability diagrams
    montecarlo/     Regime-conditioned paths, VaR, CVaR
    backtest/       Overlay, IR, tracking error, case-study replay
    audit/          Lineage tracker, model card generator, IC artefact
  r/                Parallel R implementations (depmixS4, MSwM, Stan, bcp)
  notebooks/        One Jupyter notebook per phase
  reports/          Figures, tables, HTML/PDF reports, 18-slide deck
  tests/            pytest suite for every phase
  docs/             decisions.md, model_card.md, experiment_log.md
```

## Regime Definitions

| Regime | Key Signature |
|--------|--------------|
| **Risk-On** | Nifty > 200-DMA · 50DMA > 200DMA · VIX < 14 · FII inflows ↑ · breadth > 70% |
| **Risk-Off** | Nifty < 200-DMA · VIX > 22 · FII outflows persistent · INR deprecating · gilt yields ↑ |
| **Transitional** | Mixed breadth · VIX 14–22 · conflicting signals · low conviction |
| **Late-Cycle** | Trend ↑ but narrow breadth · valuation z-score > 1.5 · sector rotation |
| **Post-Shock** | Vol decay after drawdown · oversold breadth · credit spreads compressing |

> **Note**: The rule-based label series (see `src/regime/data/labels.py`) is a **weak label** used for
> interpretability and accuracy scoring alongside log-likelihood. It is not ground truth.

## Phases

| Phase | Description | Acceptance |
|-------|-------------|------------|
| 1 | Data ingestion, quality | Clean parquet + quality report |
| 2 | 30+ features, leakage test | Unit tests pass, TDA/GNN optional |
| 3 | Frequentist HMM | Stable labels, BIC comparison |
| 4 | Bayesian HMM (PyMC) | Diagnostics pass |
| 5 | Regime-switching VAR | Bayesian RS-VAR diagnostics pass |
| 6 | Bayesian deep learning | Comparison table: log-loss, Brier, ECE |
| 7 | Foundation-model hybrids | Both members logged to MLflow |
| 8 | Ensemble | Beats all members on OOS calibrated log-likelihood |
| 9 | Online inference | Streamed-day demo runs end-to-end |
| 10 | Conformal calibration | Coverage within ±3pp of nominal |
| 11 | Case studies | 8+ page report with charts |
| 12 | Backtest & Monte Carlo | No lookahead; results reported honestly |
| 13 | R cross-checks | Reconciliation table committed |
| 14 | Report, model card, deck | `make all` rebuilds everything |
| 15 | Handover | Tag release; transfer to ZethetaIntern |

## Manual Data Required (Phase 1)

Place these files in `data/raw/manual/` before running `make phase-1`:

| File | Source | Notes |
|------|--------|-------|
| `fii_dii_flows.csv` | [NSE India](https://www.nseindia.com/reports/fii-dii) | FII/DII daily flows |
| `sip_monthly.csv` | [AMFI](https://www.amfiindia.com/research-information/industry-data) | Monthly SIP data |
| `india_10y_rbi.csv` | [RBI DBIE](https://dbie.rbi.org.in/) | 10Y benchmark yield |
| `credit_spread.csv` | CRISIL / CMIE | AAA-Gilt spread |
| `nifty_pe.csv` | NSE Bhavcopy / manual | Trailing P/E for valuation z-score |

> If files are missing, Phase 1 will log the gap and use proxies where available.

## Experiment Tracking

MLflow UI: `mlflow ui --port 5000`  
All experiments logged under the `regime-detection` experiment.

## Docs

- [`docs/decisions.md`](docs/decisions.md) — All design decisions logged
- [`docs/experiment_log.md`](docs/experiment_log.md) — Model experiment history
- [`docs/model_card.md`](docs/model_card.md) — Full model card (populated after Phase 14)

## Licence

Proprietary. For internal use only. Not for distribution.
