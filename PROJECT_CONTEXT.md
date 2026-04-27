# HEATWAVE-AI-Prediction — PROJECT CONTEXT

**Version**: 2.0.0
**Date**: 2026-03-07
**Status**: ✅ v2.0 Refactor Complete — Ready for Retrain with New Definition + NDVI
**Author**: HEATWAVE-AI Team

---

## Changelog v1.0 → v2.0

| รายการ                  | v1.0                 | v2.0                                    |
| ----------------------- | -------------------- | --------------------------------------- |
| **Heatwave Definition** | `t2m_c >= 35°C`      | **Heat Index >= 41°C** (Rothfusz)       |
| **RH Calculation**      | ไม่มี                | **August-Roche-Magnus formula**         |
| **Heat Index Formula**  | Steadman (approx.)   | **Rothfusz Regression (NWS standard)**  |
| **NDVI**                | ไม่มี                | **MODIS MOD13A3 (GEE)** — optional      |
| **Training Period**     | 2000–2015            | 2000–2015 (+ 2016–2025 extension พร้อม) |
| **Feature Count**       | 5                    | 6 (+ ndvi/lag เมื่อ enabled)            |
| **Encoding**            | Windows cp1252 (bug) | **UTF-8 ทุกไฟล์**                       |
| **Labeling Mode**       | Fixed                | Dual-mode: `heat_index` / `temperature` |

---

## 1. Project Overview

HEATWAVE-AI-Prediction เป็น **modular AI experimentation platform** สำหรับ Binary Heatwave Classification ในพื้นที่ประเทศไทย ใช้ ERA5 Reanalysis Climate Data และ MODIS NDVI ในการเทรน 5 โมเดล Machine Learning

### Goals

- เทรน 5 ML models บน ERA5 + NDVI features ของประเทศไทย
- Benchmark และ rank models อัตโนมัติด้วย Leaderboard
- บันทึก trained models เพื่อ real-time inference
- แสดงผลผ่าน premium dark-mode web dashboard
- เปิดระบบทั้งหมดผ่าน [`Start.bat`](Start.bat) คลิกเดียว

---

## 2. System Architecture

```
HEATWAVE-AI-TRAIN/
│
├── Era5-data-2000-2026/          ← ERA5 NetCDF files (era5_surface_YYYY.nc)
│
├── data/
│   └── ndvi/                     ← [NEW v2.0] MODIS NDVI GeoTIFF + ndvi_aligned_era5.nc
│
├── config/
│   └── config.yaml               ← Central config (paths, thresholds, NDVI settings)
│
├── utils/
│   ├── data_loader.py            ← ERA5 NetCDF → pandas DataFrame
│   ├── preprocessing.py          ← [UPDATED v2.0] Rothfusz HI, dual-mode labels, NDVI merge
│   ├── gpu_utils.py              ← CUDA detection helper
│   ├── ndvi_downloader.py        ← [NEW v2.0] MODIS NDVI download via GEE
│   └── ndvi_processor.py         ← [NEW v2.0] Reproject + Resample + Lag features
│
├── models/                       ← 5 ML model implementations (unchanged)
│   ├── balanced_random_forest.py
│   ├── xgboost_model.py
│   ├── lightgbm_model.py
│   ├── mlp_model.py
│   └── kan_model.py
│
├── training/                     ← Trainer, CV, hyperparameter tuning
├── pipelines/training_pipeline.py
├── evaluation/                   ← metrics.py, benchmark.py
├── prediction/                   ← predictor.py, predict.py
├── dashboard/                    ← Flask + Chart.js dark-mode UI
│
├── experiments/
│   ├── results/                  ← JSON results + leaderboard.json
│   └── models/                   ← .pkl models + scaler.pkl
│
├── download_extension_data.py    ← [NEW v2.0] Download ERA5+NDVI for 2016-2025
├── main.py                       ← Unified CLI (train | dashboard | predict)
├── Start.bat                     ← One-click launcher
└── requirements.txt              ← Python dependencies (+ GEE, rioxarray, rasterio)
```

---

## 3. Data Pipeline (v2.0)

### 3.1 ERA5 Variables

| ตัวย่อ | ชื่อเต็ม                | หน่วย  |
| ------ | ----------------------- | ------ |
| `t2m`  | 2m Temperature          | K → °C |
| `d2m`  | 2m Dewpoint Temperature | K → °C |
| `sp`   | Surface Pressure        | Pa     |
| `u10`  | 10m U-wind Component    | m/s    |
| `v10`  | 10m V-wind Component    | m/s    |

### 3.2 Feature Engineering Pipeline (v2.0)

```
ERA5 NetCDF
    ↓
[1] K → °C conversion    (t2m, d2m)
    ↓
[2] _compute_rh_from_era5()       ← NEW Magnus formula: RH(%) จาก t2m_c + d2m_c
    ↓
[3] _compute_derived_features()   ← Rothfusz Heat Index, wind_speed
    ↓
[4] _merge_ndvi_features()        ← NEW MODIS NDVI + lag (skip ถ้า enabled: false)
    ↓
[5] _generate_labels()            ← Dual-mode: heat_index >= 41°C (default) | t2m_c >= 35°C
    ↓
[6] StandardScaler → train/val/test split (70/15/15 stratified)
```

### 3.3 Final Feature List

| #   | Feature      | ที่มา              | หน่วย                   |
| --- | ------------ | ------------------ | ----------------------- |
| 1   | `t2m_c`      | ERA5               | °C                      |
| 2   | `d2m_c`      | ERA5               | °C                      |
| 3   | `rh`         | Derived (Magnus)   | %                       |
| 4   | `heat_index` | Derived (Rothfusz) | °C                      |
| 5   | `wind_speed` | Derived (u10+v10)  | m/s                     |
| 6   | `sp`         | ERA5               | Pa                      |
| 7   | `ndvi`       | MODIS MOD13A3      | [-1, 1] — เมื่อ enabled |
| 8   | `ndvi_lag1`  | MODIS (lag 1m)     | [-1, 1] — เมื่อ enabled |
| 9   | `ndvi_lag2`  | MODIS (lag 2m)     | [-1, 1] — เมื่อ enabled |

### 3.4 Heatwave Label Definition (v2.0)

```python
# Heat Index Mode (default, config.yaml → labeling_method: "heat_index")
RH = August-Roche-Magnus(t2m_c, d2m_c)
HI = Rothfusz_Regression(t2m_c, RH)        # °C
heatwave = 1 if HI >= 41.0 else 0

# Legacy Temperature Mode (backward compat)
heatwave = 1 if t2m_c >= 35.0 else 0
```

**ผลการทดสอบ:**
| กรณี | t2m | RH | HI | Label |
|---|---|---|---|---|
| ชื้น (Bangkok ก่อนฝน) | 35°C | 84.5% | 59.7°C | **1** ✅ |
| แห้ง (ภาคเหนือฤดูแล้ง) | 35°C | 21.8% | 33.3°C | **0** ✅ |

---

## 4. NDVI Integration (v2.0)

### 4.1 Data Source

| รายการ      | ค่า                                  |
| ----------- | ------------------------------------ |
| Product     | MODIS MOD13A3 v061                   |
| Resolution  | 1 km → resample to 0.25° (ERA5 grid) |
| Temporal    | Monthly composite                    |
| Download    | Google Earth Engine (GEE)            |
| GEE Project | `gen-lang-client-0381821743`         |

### 4.2 NDVI Status

| ขั้นตอน                  | สถานะ                                                     |
| ------------------------ | --------------------------------------------------------- |
| GEE Authentication       | ✅ เสร็จแล้ว                                              |
| GEE Tasks 2000–2015      | ✅ Submitted (16 tasks) — รอ export                       |
| GEE Tasks 2016–2025      | ✅ Submitted (10 tasks) ผ่าน `download_extension_data.py` |
| Download .tif จาก Drive  | ⏳ รอ GEE export เสร็จ                                    |
| `ndvi_processor.py`      | ⏳ รัน หลัง download                                      |
| `ndvi.enabled` ใน config | ❌ `false` (เปลี่ยนเป็น `true` หลัง process)              |

### 4.3 NDVI Workflow

```bash
# 1. รอ GEE export → monitor: https://code.earthengine.google.com/tasks
# 2. Download .tif จาก Google Drive folder "HeatAI_NDVI" → data/ndvi/
python utils/ndvi_processor.py          # Reproject + Resample + saves ndvi_aligned_era5.nc
# 3. เปิด NDVI (config.yaml → ndvi.enabled: true)
python main.py --mode train             # Retrain with NDVI features
```

---

## 5. Data Coverage

| Dataset          | ปีที่มี   | สถานะ                                   |
| ---------------- | --------- | --------------------------------------- |
| ERA5 (original)  | 2000–2015 | ✅ พร้อม                                |
| ERA5 (extension) | 2016–2025 | ⏳ รันผ่าน `download_extension_data.py` |
| NDVI (original)  | 2000–2015 | ⏳ รอ GEE export                        |
| NDVI (extension) | 2016–2025 | ⏳ รอ GEE export                        |

---

## 6. Models

| #   | Model                      | Key Characteristics                                                          |
| --- | -------------------------- | ---------------------------------------------------------------------------- |
| 1   | **Balanced Random Forest** | `n_estimators=200`, `max_depth=15`, handles imbalance via balanced bootstrap |
| 2   | **XGBoost**                | `n_estimators=300`, early stopping, GPU support                              |
| 3   | **LightGBM**               | Fast leaf-wise boosting, early stopping                                      |
| 4   | **MLP Neural Network**     | 3-layer (256→128→64), ReLU, Dropout 0.3, early stopping                      |
| 5   | **KAN**                    | Kolmogorov–Arnold Network, B-spline activations, grid_size=5                 |

### 6.1 Best Results (Label = old `t2m >= 35°C` — ต้อง Retrain ด้วย definition ใหม่)

| Rank | Model           | F1     | ROC-AUC | หมายเหตุ                            |
| ---- | --------------- | ------ | ------- | ----------------------------------- |
| 🥇   | Balanced RF     | 0.9963 | 1.000   | Best — 190 วินาที                   |
| 🥈   | LightGBM        | 0.6500 | 0.999   | Precision ต่ำ (0.48)                |
| 3–5  | KAN/MLP/XGBoost | 0.0000 | —       | Predict 0 ทั้งหมด (class imbalance) |

> ⚠️ **ต้อง Retrain ทุกโมเดล** หลังเปลี่ยน Label Definition เป็น Heat Index

---

## 7. Training & Split

| รายการ           | ค่า                                 |
| ---------------- | ----------------------------------- |
| **Split Method** | Random Stratified (ไม่ใช่ Temporal) |
| **Train**        | 70%                                 |
| **Validation**   | 15%                                 |
| **Test**         | 15%                                 |
| **random_state** | 42                                  |

---

## 8. Configuration Reference (v2.0)

`config/config.yaml` — fields ใหม่ที่เพิ่มใน v2.0:

```yaml
data:
  labeling_method: "heat_index" # "heat_index" | "temperature"
  heatwave_heat_index_threshold: 41.0 # °C
  heat_index_thresholds:
    caution: 27.0
    extreme_caution: 32.0
    danger: 41.0
    extreme_danger: 54.0

ndvi:
  enabled: false # true หลัง download + process เสร็จ
  gee_project: "gen-lang-client-0381821743"
  start_year: 2000
  end_year: 2015
  output_dir: "data/ndvi/"
  processed_file: "data/ndvi/ndvi_aligned_era5.nc"
  lag_months: [0, 1, 2]
```

---

## 9. Quick Start (v2.0)

```bash
# --- Train (current definition, no NDVI) ---
Start.bat → [1] Train ALL

# --- เมื่อ NDVI พร้อม ---
# 1. วางไฟล์ .tif ใน data/ndvi/
python utils/ndvi_processor.py          # สร้าง ndvi_aligned_era5.nc
# 2. config.yaml → ndvi.enabled: true
python main.py --mode train

# --- ดาวน์โหลดข้อมูลเพิ่ม 2016-2025 ---
python download_extension_data.py --update-config

# --- Dashboard ---
python main.py --mode dashboard         # http://localhost:5000

# --- Predict ---
python main.py --mode predict --model balanced_rf --input data.csv --proba
```

---

## 10. Files Reference

| ไฟล์                         | คำอธิบาย                                              |
| ---------------------------- | ----------------------------------------------------- |
| `DATA_FAQ.md`                | ERA5 variables, NDVI source, Resolution, Label method |
| `MODEL_SPECS.md`             | Train/test split, Feature list, BRF params, ผลจริง    |
| `NDVI_SETUP_GUIDE.md`        | Step-by-step GEE authentication + download guide      |
| `download_extension_data.py` | Download ERA5+NDVI 2016-2025                          |
| `utils/ndvi_downloader.py`   | GEE export tasks                                      |
| `utils/ndvi_processor.py`    | GeoTIFF → ERA5-aligned NetCDF                         |
| `Heatwave-definition.md`     | Spec ของ Heat Index definition ใหม่                   |
| `NDVI_INTEGRATION_PLAN.md`   | Spec ของ NDVI integration                             |

---

## 11. Implementation Status (v2.0)

| Component                         | Status                               |
| --------------------------------- | ------------------------------------ |
| Heatwave Definition (Rothfusz HI) | ✅ Implemented + Tested              |
| RH Calculation (Magnus formula)   | ✅ Implemented + Tested              |
| Dual-mode label generation        | ✅ Implemented                       |
| NDVI Downloader (GEE)             | ✅ Implemented — Tasks submitted     |
| NDVI Processor                    | ✅ Implemented — รอ .tif files       |
| NDVI Feature Merge                | ✅ Implemented (skip เมื่อ disabled) |
| UTF-8 encoding fix (ทุกไฟล์)      | ✅ Fixed                             |
| ERA5 + NDVI extension 2016–2025   | ⏳ Download in progress              |
| **Retrain ด้วย definition ใหม่**  | ❌ **ยังไม่ได้ทำ — ต้องทำ**          |

---

## 12. Roadmap (v2.0)

| Priority  | Task                                                          |
| --------- | ------------------------------------------------------------- |
| 🔴 สูงมาก | **Retrain ทุกโมเดลด้วย Heat Index label**                     |
| 🔴 สูง    | รอ GEE export เสร็จ → download .tif → run `ndvi_processor.py` |
| 🟡 กลาง   | Retrain พร้อม NDVI features (9 features)                      |
| 🟡 กลาง   | Download ERA5 2016–2025 ผ่าน CDS API (ต้องมี API key ก่อน)    |
| 🟢 ต่ำ    | SHAP explainability plots ใน dashboard                        |
| 🟢 ต่ำ    | `/api/predict` REST endpoint                                  |
| 🟢 ต่ำ    | Low/Medium/High Risk classification (prob threshold)          |

---

## 13. Dependencies (v2.0)

| Library                 | Purpose                               |
| ----------------------- | ------------------------------------- |
| `xarray`, `netCDF4`     | ERA5 NetCDF ingestion                 |
| `pandas`, `numpy`       | Data manipulation                     |
| `scikit-learn`          | Preprocessing, metrics, CV            |
| `imbalanced-learn`      | Balanced Random Forest                |
| `xgboost`, `lightgbm`   | Gradient boosting models              |
| `torch`                 | PyTorch — MLP & KAN                   |
| `flask`                 | Web dashboard                         |
| `pyyaml`                | Config parsing                        |
| `earthengine-api`       | **[NEW]** MODIS NDVI download via GEE |
| `rioxarray`, `rasterio` | **[NEW]** GeoTIFF reproject/resample  |
| `cdsapi`                | **[NEW]** ERA5 extension download     |
