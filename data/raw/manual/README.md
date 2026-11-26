# Manual Data Files — Required for Phase 1

Place these files in `data/raw/manual/` **before** running `make phase-1`.
All files are publicly available from the sources listed below.

---

## 1. `fii_dii_flows.csv` — FII/DII Daily Flows

**Source:** NSE India → Reports → FII/DII Activity  
**URL:** https://www.nseindia.com/reports/fii-dii  
**Date range needed:** 2007-01-01 to present  
**Expected columns:**

```
date,fii_buy,fii_sell,fii_net,dii_buy,dii_sell,dii_net
2007-01-02,1234.56,987.65,246.91,500.00,450.00,50.00
...
```

---

## 2. `sip_monthly.csv` — Monthly SIP Data

**Source:** AMFI India → Industry Data  
**URL:** https://www.amfiindia.com/research-information/industry-data  
**Date range needed:** 2010-01-01 to present  
**Expected columns:**

```
date,sip_accounts,sip_amount_cr
2010-01-01,2500000,1234.56
...
```

---

## 3. `india_10y_rbi.csv` — India 10Y Benchmark Gilt Yield

**Source:** RBI DBIE (Database on Indian Economy)  
**URL:** https://dbie.rbi.org.in/ → Interest Rates → G-Sec yields  
**Date range needed:** 2007-01-01 to present  
**Expected columns:**

```
date,yield_10y_pct
2007-01-02,7.65
...
```

---

## 4. `credit_spread.csv` — AAA-Gilt Credit Spread (Proxy)

**Source:** CRISIL / CMIE Prowess / RBI publications  
**Note:** AAA corporate bond yield minus 10Y gilt yield  
**Expected columns:**

```
date,credit_spread_aaa_gilt_bps
2007-01-02,65.0
...
```

---

## 5. `nifty_pe.csv` — Nifty 50 Trailing P/E

**Source:** NSE India → Indices → Historical Data → P/E  
**URL:** https://www.nseindia.com/reports/indices-historical-data  
**Expected columns:**

```
date,pe_ratio
2007-01-02,22.5
...
```

---

## Notes on Missing Files

If a file is missing, Phase 1 will:
1. Log a **WARNING** naming the missing file
2. Record it in `data/lineage/phase1_lineage.json` under `missing_series`
3. Continue with available data
4. Mark any dependent features as `proxy` in Phase 2

The pipeline will NOT fail silently — missing data is always documented.
