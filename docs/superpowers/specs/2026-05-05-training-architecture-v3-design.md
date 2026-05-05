# Training Architecture v3 — Design Spec

**Date:** 2026-05-05  
**Status:** Approved  
**Execution:** Google Colab Free (T4 GPU, CUDA)

---

## Problem Statement

Current v3 LightGBM model fails all 5 acceptance tests:

| Metric | Current | Target |
|---|---|---|
| MAE (h24) | 3.252 | ≤ 2.40 |
| skill_score | -1.21 (h6) / 0.515 (h24) | ≥ 0.55 |
| danger_recall@42°C | 0.276 | ≥ 0.40 |
| PI coverage 90% | 0.953 | [0.85, 0.93] |
| PI width ratio | 1.18 | ≥ 1.50 |

Root causes (verified):
1. **Single 70/15/15 split** → Optuna overfits val set → negative skill score
2. **Manual 9x sample weight** for HI ≥ 42°C → warm bias +2.99°C
3. **LGBM binary DangerGate** → mechanistically inferior to BRF at <1% imbalance
4. **No EVT-conformal** → extreme tail intervals unreliable with few calibration samples

---

## Architecture Overview

```
Raw observations
     │
     ▼
Feature Builder (unchanged)
     │
     ▼
Expanding-Window CV (2 folds, 72h purge)
     │
     ├──────────────────────────┐
     ▼                          ▼
LGBM Quantile Regressor    BRF DangerGate (3-class)
(temp_c + rh → HI)         (safe/warning/danger)
DenseLoss sample_weight     OOB proba → feature
     │                          │
     └──────────┬───────────────┘
                ▼
         HI Composition (Rothfusz)
                │
     ┌──────────┴──────────┐
     ▼                      ▼
Mondrian CQR            EVT-Conformal
(60 strata)             (extreme tail only)
     │
     ▼
PredictionBundle (hi_mean, hi_lower, hi_upper, danger_proba, danger_tier)
```

---

## Section 1: Training Protocol

### Expanding-Window CV (2 folds)

```
Fold 1: Train[2020–2022] → Val[2023]  (72h purge gap)
Fold 2: Train[2020–2023] → Val[2024]  (72h purge gap)
Final:  Train[2020–2024] → Test[2025-01 to 2025-04]
```

- Optuna: 50 trials × 2 folds = 100 fits/slot (same compute as current 100 × 1)
- Objective: minimize average CV MAE across 2 folds
- Purge gap: 72h > max horizon (h=72), prevents lag-feature leakage
- References: Roberts 2017 (1,594 citations), Schnaubelt 2019

### Files changed
- `scripts/train_forecast.py` — outer loop refactor, CV split logic
- `app/ml/forecast/backends/lgbm_backend.py` — `_tune()` uses CV MAE objective

---

## Section 2: DenseLoss Sample Weighting

Replace manual weights (5x >38°C, 9x >42°C) with KDE-based inverse-density weights:

```python
def compute_dense_weights(y_hi: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    kde = gaussian_kde(y_hi)
    density = kde(y_hi)
    weights = 1.0 / (density ** alpha)
    weights /= weights.mean()
    return weights
```

- `alpha` tuned by Optuna in range [0.3, 1.5]
- Eliminates cliff-edge at 38°C/42°C that causes warm bias
- Reference: Steininger 2021 (DenseLoss, 159 citations, Machine Learning)

### Files changed
- `app/ml/forecast/backends/lgbm_backend.py` — `fit()` computes DenseLoss weights; `_tune()` adds `alpha` to search space

---

## Section 3: BRF DangerGate (3-Class)

### Class Definition

```
0 = safe    (HI < 38°C)
1 = warning (38°C ≤ HI < 42°C)
2 = danger  (HI ≥ 42°C)
```

### Classifier

```python
from imblearn.ensemble import BalancedRandomForestClassifier

clf = BalancedRandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=2,
    sampling_strategy="all",
    replacement=True,
    n_jobs=-1,
    random_state=42,
)
```

### OOB Reentrancy (no leakage)

Training: BRF OOB estimates on X_train → concatenate [X_train | OOB_proba_3d] → LGBM  
Inference: BRF predict_proba(X) → concatenate [X | proba_3d] → LGBM

### Threshold Optimization

Per-station sweep on val set:
- warning threshold: maximize recall s.t. precision ≥ 0.50
- danger threshold: maximize recall s.t. precision ≥ 0.55, recall ≥ 0.40

- References: Imani 2025, Ayodele 2023 (BRF wins at <2% minority)
- User empirical confirmation: BRF previously performed best

### Files changed
- `app/ml/forecast/danger_gate.py` — full rewrite: 3-class BRF, OOB, 2-threshold sweep
- `app/ml/forecast/backends/lgbm_backend.py` — `fit()` inserts DangerGate proba into X; `predict_with_pi()` uses 3 tiers

---

## Section 4: Calibration

### Mondrian CQR (enhanced)

Strata: `(station_id, danger_tier, local_hour // 6)` → 5 × 3 × 4 = 60 strata  
Fallback: global q_hat when stratum < 10 samples  
Unchanged: finite-sample correction, adjust() interface

### EVT-Conformal (new, extreme tail only)

Applied only when `danger_tier == 2` (predicted HI ≥ 42°C):

```python
from scipy.stats import genpareto

# Fit GPD on extreme conformity scores (calibration set, tier=2 only)
scores_extreme = conformity_scores[tier == 2]
params = genpareto.fit(scores_extreme)
q_hat_evt = genpareto.ppf(1 - alpha, *params)  # alpha=0.10

# Apply at inference
if danger_tier == 2:
    hi_lower -= q_hat_evt
    hi_upper += q_hat_evt
```

- Works with 20–50 extreme calibration samples (GPD asymptotic theory)
- Reference: Pasche et al. 2025 (EVT-conformal for high-impact events)

### Files changed
- `app/ml/forecast/conformal.py` — `MondianCQRCalibrator`: add danger_tier to strata; add `EVTConformalCalibrator` class

---

## Section 5: Google Colab Integration

### Notebook: `notebooks/train_colab.ipynb`

**Cell structure:**
1. Setup: pip install + Drive mount
2. Credentials: Colab Secrets (CDSAPI_KEY, TMD_API_KEY, TMD_API_TOKEN)
3. Data pipeline: API download → Drive cache (7-day TTL)
4. Train loop: slot-by-slot with checkpoint
5. Artifacts: zip models → Drive download

### Data Pipeline

- Download from TMD + ERA5 + NASA POWER APIs on first run
- Cache parquet at `Drive/HeatShield/data/raw/station_id={id}/`
- Re-download if cache > 7 days old
- ERA5 CDS API: expected 30–120 min first download

### Checkpointing

```python
# Drive/HeatShield/checkpoints/progress.json
{"BKK_01_h6": "done", "BKK_01_h12": "in_progress", ...}
```

- Write "in_progress" before training each slot
- Write "done" after slot saves successfully
- On resume: skip "done" slots, retry "in_progress"
- Optuna SQLite on Drive → warm-start after disconnect

### GPU Configuration

```python
os.environ["LGBM_DEVICE"] = "cuda"  # T4 CUDA on Colab (Linux pip wheel supports)
```

### Compute Budget (Free T4, 12h limit)

| Phase | Per slot | 25 slots |
|---|---|---|
| ERA5 download | 90 min total | once |
| Optuna 50×2-fold | ~15 min | ~6.2h |
| Final model (24 boosters) | ~5 min | ~2.1h |
| **Total** | | **~9.5h** ✓ |

### Model Sync

After training: `zip Drive/HeatShield/models/forecast_v3/ → download → extract to app/models/forecast_v3/`

### Files created
- `notebooks/train_colab.ipynb`

---

## Expected Outcomes

| Metric | Current | Target | Expected |
|---|---|---|---|
| MAE (h24) | 3.252 | ≤ 2.40 | 2.0–2.3 |
| skill_score | 0.515 | ≥ 0.55 | 0.60–0.70 |
| danger_recall@42°C | 0.276 | ≥ 0.40 | 0.40–0.55 |
| PI coverage 90% | 0.953 | [0.85, 0.93] | 0.87–0.92 |
| PI width ratio | 1.18 | ≥ 1.50 | 1.6–2.0 |

---

## Implementation Order

1. `danger_gate.py` — BRF 3-class (standalone, no deps)
2. `conformal.py` — EVT-conformal + Mondrian strata update
3. `lgbm_backend.py` — DenseLoss + CV objective + DangerGate reentrancy
4. `scripts/train_forecast.py` — expanding CV loop
5. `notebooks/train_colab.ipynb` — Colab notebook

---

## Invariants (must not break)

- Feature engineering no-leakage rule (all lags use `.shift(n)`)
- v3 bundle.json schema (additive only: `danger_tier`, `alpha`, `evt_params`)
- API response format unchanged
- `predict.py` v3→v2→v1 fallback chain intact
