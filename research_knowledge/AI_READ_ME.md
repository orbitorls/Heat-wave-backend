# AI RESEARCH GUIDE: Heatwave Prediction in Thailand

**Objective:** Predict sub-seasonal heatwaves in Thailand using ERA5 data and Machine Learning.
**Primary Challenge:** Data Imbalance (Heatwaves are rare, approx. 1-2% of days).
**Target Approach:** Balanced Random Forest (BRF) and Spatial-Aware Ensemble.

## 📊 Technical Directives for AI (Read Before Coding)

### 1. Data Definitions (Thailand Context)
- **Heatwave Threshold:** Use the **95th percentile** of historical $T_{max}$ (T2M) for the specific location and month.
- **Duration:** 3+ consecutive days where $T_{max} > \text{Threshold}$.
- **Humidity Factor:** For Thailand (tropical), $T_{max}$ alone is insufficient. Humidity (Specific/Relative) must be included to account for "moist heatwaves" (Heat Index).

### 2. Balanced Random Forest (BRF) Implementation
- **Algorithm:** Use `imblearn.ensemble.BalancedRandomForestClassifier`.
- **Mechanism:** It performs random under-sampling of the majority class (non-heatwave days) *within each bootstrap sample* of each decision tree.
- **Advantage:** Avoids data loss from global under-sampling while ensuring each tree sees a balanced subset.
- **Hyperparameters to Tune:**
  - `sampling_strategy='all'` (ensures balance in each tree).
  - `replacement=True`.
  - `n_estimators=300+`.

### 3. Feature Selection (ERA5 Variables)
| Variable | Code | Role |
| :--- | :--- | :--- |
| 2m Temperature | `t2m` | Target Variable (Max Daily). |
| 500 hPa Geopotential | `z` | Atmospheric Blocking / Heat Dome detection. |
| Soil Moisture | `swvl1` | Land-Atmosphere Feedback (Dry soil = hotter surface). |
| Total Precipitation | `tp` | Preceding rainfall impact on soil cooling. |
| Relative Humidity | `rh` | Critical for "Feels Like" temperature in tropics. |

### 4. Evaluation Metrics (DO NOT USE ACCURACY)
- **Primary:** **F1-Score** (Balance of Precision and Recall).
- **Secondary:** **Recall (Hit Rate)** - Missing a heatwave is worse than a false alarm.
- **Secondary:** **Precision-Recall AUC** (Better than ROC-AUC for imbalanced data).

## 📚 Key Research References
- **Chongtaku et al. (2024):** *"Integrating Remote Sensing and Ground-Based Data for Enhanced Spatiotemporal Analysis of Heatwaves."* (Thailand focus).
- **Chen et al. (2004):** *"Using Random Forest for Imbalanced Data."* (Foundation of BRF).
- **ECMWF ERA5 Documentation:** Guidelines for $Z_{500}$ and $T_{2m}$ extraction.

## 🛠️ Implementation Workflow (Future Steps)
1. Pre-process ERA5 into (Lat, Lon, Time) cubes.
2. Label indices as 1 (Heatwave) or 0 (Normal).
3. Extract features including **7-day lags** (history).
4. Train using `BalancedRandomForestClassifier`.
5. Evaluate using a temporal split (e.g., train on 2000-2015, test on 2016+).
