# HeatShield AI — Forecast v4 Improvement Design

**Date:** 2026-05-05  
**Status:** Draft  
**Goal:** Improve v3 LightGBM forecast model from negative skill scores to meeting all acceptance criteria by combining 5-year data ingestion with parallel model improvements.

---

## Problem Statement

Current v3 model fails on all key metrics:

| Metric | Current | Target |
|---|---|---|
| MAE h24 | ~3.5°C | ≤ 2.40°C |
| skill_score | < 0 on 20/25 slots | ≥ 0 all slots, ≥ 0.55 primary |
| danger_recall@42°C | 0.28 | ≥ 0.40 |
| PI coverage 90% | 0.95 | 0.85–0.93 |
| PI width ratio | 1.18 | ≥ 1.50 |

Root causes:
1. **Insufficient data** — 1 year (8,688 rows), danger class only 131 rows for BKK_01, near 0 for other stations
2. **DangerGate not wired** — `danger_proba` computed but never overrides `hi_mean`; regression head biased low drives every alert
3. **Marginal conformal** — EnbPI ignores station/time-of-day variation → over-coverage (0.95) with narrow width ratio (1.18)
4. **Feature gaps** — `wind_ms` and `precip_mm` excluded from features despite being available; no station geometry

---

## Approach: Parallel Execution (Chosen)

Run data ingestion and model code improvements simultaneously. Code changes do not depend on data. When both complete, run a single training pass with the full 5-year dataset.

---

## Section 1: Data Ingestion

**Sources to ingest:**
- ERA5 (CDSAPI): hourly reanalysis, temp/rh/wind/precip, 2020-01-01 → 2025-04-30
- TMD: observational ground truth for Thai stations, same period
- NASA POWER: remains as gap-fill / fallback

**Data priority order** (updated in `app/data/loaders.py`):
TMD > ERA5 > NASA POWER — for duplicate `(station_id, ts_utc)` rows, higher-priority source wins.

**Expected outcome:**
- ~43,800 rows (5× current)
- BKK_01 danger events: ~131 → ~650 rows
- Other stations: enough danger events for meaningful DangerGate training

**Commands:**
```powershell
python scripts/ingest_era5.py --start 2020-01-01 --end 2025-04-30
python scripts/ingest_tmd.py --start 2020-01-01 --end 2025-04-30
```

---

## Section 2: Model Code Improvements

Four independent groups, executed in parallel via multi-agent dispatch.

### Group A — `app/ml/forecast/features.py`

1. Split `build_features` → `build_X_once(df)` + `build_y_for_horizon(df, h, target_kind)` to cache features once per station and slice per horizon (eliminates redundant recompute across 5 horizons)
2. Remove `wind_ms`, `precip_mm` from `_NON_FEATURE`; add lags 1h/3h for both
3. Inject `lat`, `lon`, `elevation_m` from `STATIONS` registry as constant columns per station
4. Add evening-peak interaction: `temp_c × indicator(local_hour ∈ [16, 20])`
5. Update `get_feature_names()` to stay in sync with predict-time validation

### Group B — `app/ml/forecast/backends/lgbm_backend.py`

1. **Fix NameError** — `n_val` referenced but undefined at line 125; define from `n_train`
2. **Wire DangerGate** — in `predict_with_pi()`: when `danger_proba ≥ station_threshold`, override `hi_mean ← max(hi_mean, hi_q97)` and widen lower PI bound. This is the primary safety fix.
3. Add `q=0.97` to `_QUANTILES` (model can now reach ≥42°C band)
4. Tail sample weight ≥ 2.0 for training rows where `heat_index_c ≥ 40`
5. Persistent Optuna study + HyperbandPruner: single SQLite study per `(target_kind, horizon)` at `app/models/forecast_v3/optuna_studies.db`
6. Vectorize `_compute_hi_array` with NumPy (minor speed improvement)

### Group C — `app/ml/forecast/conformal.py`

Replace marginal EnbPI with **Mondrian CQR**: stratify calibration residuals on `(station_id, local_hour // 6)`. Use existing q05/q95 booster outputs as base intervals; apply per-stratum conformal correction with locally-weighted residuals. This directly fixes PI over-coverage and improves width ratio for danger conditions.

### Group D — `scripts/train_forecast.py` + `app/ml/forecast/danger_gate.py`

1. Refactor outer loop to use cached features from Group A: `for sid → build_X_once; for h → build_y_for_horizon`
2. Per-station threshold tuning in `danger_gate.py`: argmax recall subject to precision ≥ 0.55; persist threshold in slot bundle
3. Update data loading priority: TMD > ERA5 > NASA POWER (coordinate with loaders.py change)
4. Eval charts remain inline after every training run (8 charts per slot, versioned folder)

---

## Section 3: Integration

After all four groups complete and data ingestion finishes:

1. **Smoke test** — `python scripts/train_forecast.py --station BKK_01 --horizons 24 --trials 5` (catches code errors quickly)
2. **Full training** — `python scripts/train_forecast.py --trials 100` across all 5 stations × 5 horizons
3. **Acceptance tests** — `pytest tests/test_forecast_v2_regression.py -v` (all 5 must pass) + manual check skill_score ≥ 0 every slot
4. **Eval report** — 8 charts inline + summary heatmap + versioned folder (v4/)

---

## File Ownership (parallel-safe, no overlap)

| Agent | Files |
|---|---|
| Agent-Features | `app/ml/forecast/features.py` |
| Agent-Backend | `app/ml/forecast/backends/lgbm_backend.py` |
| Agent-Conformal | `app/ml/forecast/conformal.py` |
| Agent-TrainScript | `scripts/train_forecast.py`, `app/ml/forecast/danger_gate.py` |
| Main session | Integration (smoke → full train → tests → eval) |

One additional change in `app/data/loaders.py` (TMD priority) — handled by Agent-TrainScript or main session.

---

## Acceptance Criteria

| Metric | Target |
|---|---|
| MAE h24 | ≤ 2.40°C |
| skill_score | ≥ 0 all 25 slots; ≥ 0.55 for BKK_01/h24 |
| danger_recall@42°C | ≥ 0.40 |
| PI coverage 90% | 0.85–0.93 |
| PI width ratio danger/normal | ≥ 1.50 |
