# System Architecture

## Overview

Thailand heatwave forecasting system. Pipeline: ERA5 NetCDF ingestion → ConvLSTM + ML training → Flask inference API → React frontend.

---

## Components

### Data Layer

- **ERA5 data** (`era5_data/`) — NetCDF files with T2M, precipitation, soil water, dewpoint
- **data_loader.py** / **src/data/loader.py** — crop to Thailand bbox, merge variables, normalize, build `(T, C, H, W)` sequences
- **Freshness tracking** (`src/data/freshness.py`) — timestamps last download, alerts on stale data

### Model Layer

- **heatwave_model.py** / **src/models/convlstm.py** — `HeatwaveConvLSTM` with `SpatialAttention` gates + autoregressive rollout; `PhysicsInformedLoss` (MSE + adiabatic penalty)
- **Train_Ai.py** — trains RandomForest/LightGBM/XGBoost classifiers for binary heatwave detection; versioned checkpoint output
- **Checkpoints** — `models/heatwave_model_checkpoint_v{N}.pth` (ConvLSTM), XGBoost `.pkl` files

### API Layer (`api_server.py`)

Flask REST API, port 5000. Swagger UI at `/api/docs`.

| Endpoint | Method | Auth | Purpose |
|----------|--------|------|---------|
| `/api/health` | GET | — | System status + data freshness |
| `/api/predict` | GET | — | Current risk summary (XGBoost → ConvLSTM fallback) |
| `/api/forecast` | GET | — | 7-day forecast with regional breakdown |
| `/api/map` | GET | — | GeoJSON risk polygons |
| `/api/daily/predict` | POST | — | XGBoost prediction from daily weather inputs |
| `/api/daily/health` | GET | — | XGBoost model health |
| `/api/daily/model_info` | GET | — | XGBoost model metadata |
| `/api/predict_xgb` | GET/POST | — | Dashboard-compatible XGBoost prediction |
| `/api/training/preflight` | GET | — | Training readiness check |
| `/api/training/status` | GET | — | Live training job state |
| `/api/training/history` | GET | — | Recent training run records |
| `/api/training/start` | POST | 🔒 API key | Trigger background retraining |
| `/api/docs` | GET | — | Swagger UI |
| `/trainer` | GET | — | Training console HTML |
| `/dashboard` | GET | — | Dashboard HTML |

### Frontend (`agni-web/`)

React/TypeScript dashboard with Leaflet map, risk overview, and 7-day forecast chart.

---

## Configuration

All settings via environment variables or `config/config.yaml`, loaded by `config.py`:

| Variable | Default | Purpose |
|----------|---------|---------|
| `DATA_DIR` | `era5_data` | ERA5 NetCDF directory |
| `MODELS_DIR` | `models` | Checkpoint directory |
| `PORT` | `5000` | API server port |
| `API_KEY` | *(none)* | Auth key for protected endpoints |
| `CORS_ORIGINS` | `*` | Allowed CORS origins (comma-separated) |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `HW_BATCH_SIZE` | `16` | Training batch size |
| `HW_SEQ_LEN` | `7` | Input sequence length |
| `HW_EPOCHS` | `50` | Training epochs |

---

## Deployment

### Local

```bash
pip install -r requirements.txt
cp .env.example .env   # configure as needed
python api_server.py
```

### Docker

```bash
docker-compose up
```

---

## Data Flow

```
ERA5 NetCDF files (era5_data/)
        │
        ▼
  data_loader.py
  ├─ Crop to Thailand bbox
  ├─ Merge variables (T2M, Z500, SWVL1, ...)
  ├─ Normalize + clean NaN / outliers
  └─ Build (Batch, Time, Channels, H, W) windows
        │
        ▼
  HeatwaveConvLSTM  ←→  XGBoost / LightGBM / RF
        │
        ▼
  api_server.py  →  /api/predict, /api/forecast, /api/map
        │
        ▼
  agni-web React dashboard
```

---

## Risk Thresholds

| Level | Index | Temperature |
|-------|-------|-------------|
| LOW | 0 | < 35 °C |
| MEDIUM | 1 | 35–38 °C |
| HIGH | 2 | 38–41 °C |
| CRITICAL | 3 | > 41 °C |

---

## Model Checkpoint Naming

```
models/heatwave_model_checkpoint_v{N}.pth   # ConvLSTM (PyTorch)
models/xgboost_daily_v{N}.pkl               # XGBoost (scikit-learn)
```

The API auto-discovers the latest version via glob patterns defined in `CHECKPOINT_PATTERNS`.
