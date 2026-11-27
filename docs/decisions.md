# Design Decisions Log

> Every significant design choice is recorded here with the decision, alternatives considered, and rationale.
> Format: `## D-NNN: Title` → Decision | Alternatives | Reason | Date

---

## D-001: 5 Regime Taxonomy

**Decision:** Use exactly 5 regimes: Risk-On, Risk-Off, Transitional, Late-Cycle, Post-Shock.  
**Alternatives:** 3-state (Bull/Bear/Sideways); Hamilton 2-state; dynamic K via BIC.  
**Reason:** 5 states map to IC-actionable positions; 3-state loses Late-Cycle nuance critical for factor tilts; BIC on HMM across K∈{3,5,7} to validate (Phase 3).  
**Date:** 2026-11-26

---

## D-002: Weak Label Reference Series

**Decision:** Build a rule-based label series from regime definitions but treat it as a *weak label*, not ground truth.  
**Alternatives:** Expert-labelled series; market phases from sell-side research.  
**Reason:** No agreed ground truth for Indian equity regimes exists; rule-based labels are reproducible and auditable; acknowledged as imperfect in model card.  
**Date:** 2026-11-26

---

## D-003: yfinance as Primary Data Source

**Decision:** Use yfinance for index and macro data; supplement with NSE/RBI/AMFI manual downloads for flows, yield, credit spread.  
**Alternatives:** Bloomberg terminal; Refinitiv Eikon; CMIE Prowess.  
**Reason:** yfinance is free, no licence restrictions for research; NSE/RBI/AMFI data is public. Paid data sources unavailable in this environment. All proxies documented.  
**Date:** 2026-11-26

---

## D-004: TDA and GNN as Optional Pipeline Steps

**Decision:** TDA (giotto-tda) and GNN (torch_geometric) features are disabled by default (`use_tda: false`, `use_gnn: false` in `configs/features.yaml`).  
**Alternatives:** Always-on; remove entirely.  
**Reason:** Both are computationally expensive and add dependency risk. The core pipeline must stay green without GPU. Enable via config and document separately.  
**Date:** 2026-11-26

---

## D-005: Foundation Model Fallback Strategy

**Decision:** If Chronos or TimesFM checkpoints cannot be loaded (no GPU, model unavailable), fall back to a from-scratch small Transformer trained on the same window, labelled `fallback_transformer` in MLflow.  
**Alternatives:** Skip Phase 7 entirely; use pre-downloaded embeddings.  
**Reason:** Pipeline must stay green; honest fallback is better than silent skip.  
**Date:** 2026-11-26

---

## D-006: Expanding-Window CV Only

**Decision:** All cross-validation uses strictly expanding windows (train grows, gap = 63 days, test 1 year).  
**Alternatives:** Walk-forward (fixed train window); k-fold (rejected — time leakage).  
**Reason:** Rule 2. Random k-fold violates temporal ordering and inflates OOS metrics.  
**Date:** 2026-11-26

---

## D-007: MLflow for Experiment Tracking

**Decision:** Use local MLflow (file-based backend) for experiment tracking.  
**Alternatives:** Weights & Biases; Neptune; ClearML.  
**Reason:** No external credentials needed; fully local and reproducible; easy to export.  
**Date:** 2026-11-26

---

## D-008: Nifty Midcap Proxy

**Decision:** Use Nifty Midcap 50 (`^NSEMDCP50`) as proxy for Midcap 100 on yfinance; note clearly in lineage.  
**Alternatives:** Download Midcap 100 CSV from NSE website.  
**Reason:** yfinance does not carry `^CNXMDCP100` reliably for the full 15-year history. NSE CSV is in `data/raw/manual/` instructions.  
**Date:** 2026-11-26

---

## D-009: Bayesian HMM Label-Switching Fix

**Decision:** Use ordered constraints on emission means (μ₁ < μ₂ < ... < μ₅) in PyMC to fix label switching. Post-fit, map state indices to regime names by (mean, variance) signature.  
**Alternatives:** Post-hoc permutation search; identifiability constraint on transition matrix diagonal.  
**Reason:** Ordered means is simplest constraint that eliminates permutation degeneracy for 1D emission; works cleanly in PyMC with `pm.math.sort`. Documented limitation: ordering by mean alone may conflate Late-Cycle (high mean, low vol) and Risk-On; use (mean, vol) jointly for final labelling.  
**Date:** 2026-11-26

---

## D-010: Conformal Significance Level

**Decision:** Use α = 0.10 (90% coverage) as primary conformal target; also report α = 0.05 (95%) in appendix.  
**Alternatives:** α = 0.20 (too wide); per-regime α.  
**Reason:** 90% coverage is IC-interpretable as "9 in 10 days the true regime is in the prediction set"; not too conservative for a 5-class problem.  
**Date:** 2026-11-26

---

## D-011: Bayesian HMM Weekly Posterior + Daily Causal Filter

**Decision:** Fit the PyMC Bayesian HMM NUTS posterior (`2,000 draws × 4 chains`) on weekly (`W-FRI`) training observations (`<= 2018-12-31`, ~620 weeks), then run the daily causal forward filter using `P_daily = 0.8 * I + 0.2 * P_weekly`.  
**Alternatives:** Full daily `T = 2,825` NUTS sampling (too slow on CPU: >10 hours for 4×3,000 draws); ADVI/mean-field VI (underestimates posterior correlation).  
**Reason:** Weekly resampling reduces daily autocorrelation and makes 4-chain exact NUTS feasible in minutes while preserving daily regime filtering and full 90% credible intervals.  
**Date:** 2026-11-27
