# Heatwave Forecast API Reference

Base URL: `http://localhost:5000`  
Swagger UI: `http://localhost:5000/api/docs`

## Authentication

Set the `API_KEY` environment variable to protect sensitive endpoints.  
Protected endpoints require the header: `X-API-Key: <your-key>`

---

## Endpoints

### GET /api/health

Returns system health status including model load state and data freshness.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "data_freshness": {},
  "data_stale": false
}
```

---

### GET /api/predict

Returns the current heatwave risk prediction summary. Tries XGBoost daily model first; falls back to ConvLSTM/RF.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| days | int | 7 | Number of forecast days (used internally) |

**Response:**
```json
{
  "status": "ok",
  "date": "2024-04-15",
  "data_date": "2024-04-15",
  "generated_at": "2024-04-15T12:00:00",
  "risk_level": "HIGH",
  "risk_code": "HIGH",
  "risk_label": "High Risk",
  "risk_index": 2,
  "risk": {
    "risk_code": "HIGH",
    "risk_label": "High Risk",
    "risk_index": 2
  },
  "probability": 0.82,
  "advice": "Observed max temperature 38.5C with 82.0% heatwave probability. High Risk.",
  "model_type": "xgboost_daily",
  "temperature_source": "observed_latest_input",
  "forecast_available": false,
  "weather": {
    "T2M_MAX": 38.5,
    "T2M": 33.0,
    "T2M_MIN": 27.0,
    "RH2M": null,
    "WS10M": null
  },
  "anomaly": {
    "is_anomaly": true,
    "severity": "high",
    "n_triggers": 0,
    "triggers": []
  },
  "bbox": [97.5, 5.5, 105.7, 20.5],
  "regions": []
}
```

**Error Responses:**
- `500` — Model not loaded or prediction failed

---

### GET /api/forecast

Returns a 7-day heatwave forecast with per-day and per-region breakdowns.

**Response:**
```json
{
  "status": "ok",
  "model_type": "convlstm",
  "days": 7,
  "base_date": "2024-04-15",
  "generated_at": "2024-04-15T12:00:00",
  "forecast_available": true,
  "forecasts": [
    {
      "day": 1,
      "date": "2024-04-16",
      "day_name": "Tue",
      "probability": 0.75,
      "risk_level": "HIGH",
      "risk_code": "HIGH",
      "risk_label": "High Risk",
      "risk_index": 2,
      "risk": { "risk_code": "HIGH", "risk_label": "High Risk", "risk_index": 2 },
      "advice": "Generated from convlstm temperature forecast",
      "temperature_source": "forecast",
      "forecast_available": true,
      "weather": {
        "T2M": 33.5,
        "T2M_MAX": 39.2,
        "T2M_MIN": 27.1,
        "PRECTOTCORR": null,
        "WS10M": null,
        "RH2M": null,
        "NDVI": null
      },
      "observed_weather": { "T2M": 32.0, "T2M_MAX": 37.5, "T2M_MIN": 26.5 },
      "forecast_weather": { "T2M": 33.5, "T2M_MAX": 39.2, "T2M_MIN": 27.1 }
    }
  ],
  "region_forecasts": [
    {
      "name": "Northern Thailand",
      "zone": "north",
      "lat": 18.5,
      "lng": 99.0,
      "forecast": [
        {
          "day": 1,
          "date": "2024-04-16",
          "day_name": "Tue",
          "probability": 0.72,
          "risk_level": "HIGH",
          "risk_code": "HIGH",
          "risk_label": "High Risk",
          "risk_index": 2,
          "risk": { "risk_code": "HIGH", "risk_label": "High Risk", "risk_index": 2 },
          "weather": { "T2M": 36.0, "T2M_MAX": 36.0, "T2M_MIN": 36.0 },
          "temperature_source": "forecast",
          "temperature": 36.0,
          "temperature_c": 36.0
        }
      ]
    }
  ]
}
```

**Error Responses:**
- `500` — Model not loaded or forecast failed

---

### GET /api/map

Returns heatwave risk zones as a GeoJSON `FeatureCollection`. Grid cells below 28 °C are excluded to reduce payload size. Response is cached for 5 minutes (`Cache-Control: public, max-age=300`).

**Response:**
```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[100.0, 13.5], [100.25, 13.5], [100.25, 13.75], [100.0, 13.75], [100.0, 13.5]]]
      },
      "properties": {
        "temperature": 38.5,
        "temperature_c": 38.5,
        "risk_level": 2,
        "risk_code": "HIGH",
        "risk_label": "High Risk",
        "risk_index": 2,
        "risk": { "risk_code": "HIGH", "risk_label": "High Risk", "risk_index": 2 }
      }
    }
  ],
  "risk_schema": {
    "legacy_fields": { "risk_level": "numeric index (0..3)" },
    "canonical_fields": {
      "feature.properties.temperature_c": "temperature in Celsius",
      "feature.properties.risk_code": "LOW|MEDIUM|HIGH|CRITICAL",
      "feature.properties.risk_label": "LOW|MEDIUM|HIGH|CRITICAL",
      "feature.properties.risk_index": "0..3"
    },
    "levels": []
  }
}
```

**Error Responses:**
- `500` — Model not loaded or map generation failed

---

### POST /api/daily/predict

Predict heatwave from daily weather inputs using the XGBoost model.

**Request Body:**
```json
{
  "temp_mean": 28.5,
  "temp_max": 35.2,
  "temp_min": 22.1,
  "temp_std": 3.5,
  "humidity": 75.0,
  "pressure": 1013.25,
  "wind_speed": 5.2,
  "precipitation": 0.0,
  "date": "2024-01-15"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| temp_mean | float | ✅ | Mean temperature (°C) |
| temp_max | float | | Max temperature (°C) — defaults to `temp_mean + 5` |
| temp_min | float | | Min temperature (°C) — defaults to `temp_mean - 5` |
| temp_std | float | | Temperature std dev |
| humidity | float | | Relative humidity (%) |
| pressure | float | | Surface pressure (hPa) |
| wind_speed | float | | Wind speed (m/s) |
| precipitation | float | | Precipitation (mm) |
| date | string | | ISO date string |

**Response:**
```json
{
  "heatwave_probability": 0.75,
  "heatwave_predicted": true,
  "risk_level": "HIGH",
  "threshold_used": 28.43,
  "input_features": {},
  "timestamp": "2024-04-15T12:00:00"
}
```

**Error Responses:**
- `400` — Missing required field (`temp_mean`)
- `500` — Prediction error

---

### GET /api/daily/health

Health check for the XGBoost daily prediction endpoint.

**Response:**
```json
{
  "status": "ready",
  "model_available": true,
  "xgboost_available": true
}
```

---

### GET /api/daily/model_info

Returns metadata about the loaded XGBoost daily prediction model.

**Response:**
```json
{
  "model_type": "xgboost_daily",
  "model_path": "models/xgboost_daily_v1.pkl",
  "threshold_celsius": 35.0,
  "feature_names": ["temp_mean", "temp_max", "temp_min", "..."],
  "temp_mean_celsius": 30.5,
  "temp_std": 4.2
}
```

**Error Responses:**
- `404` — No daily model loaded

---

### GET /api/predict_xgb  ·  POST /api/predict_xgb

Dashboard-compatible prediction endpoint using the XGBoost model. Returns a response in the same shape as `GET /api/predict`.

**Request Body (POST, optional):**
```json
{
  "temp_mean": 30.0,
  "temp_max": 35.0,
  "temp_min": 25.0,
  "temp_std": 4.0,
  "humidity": 70.0,
  "precipitation": 0.0
}
```

**Response:**
```json
{
  "status": "ok",
  "date": "2024-04-15",
  "generated_at": "2024-04-15T12:00:00",
  "risk_level": "HIGH",
  "risk_code": "HIGH",
  "risk_label": "High Risk",
  "risk_index": 2,
  "probability": 0.7512,
  "heatwave_probability": 0.7512,
  "advice": "Temperature 35.0C with 75.1% heatwave probability. High Risk.",
  "model_type": "xgboost_daily",
  "input_features": {},
  "threshold_used": 28.43
}
```

---

### GET /api/training/preflight

Returns readiness information before starting a training run.

**Response:**
```json
{
  "data_dir_exists": true,
  "models_dir_exists": true,
  "data_file_count": 5,
  "checkpoint_count": 3,
  "resources_loaded": true,
  "gpu": { "available": false, "name": null },
  "issues": [],
  "ready_for_training": true
}
```

---

### GET /api/training/status

Returns the current training job state including live epoch metrics.

**Response:**
```json
{
  "status": "running",
  "started_at": "2024-04-15T12:00:00",
  "finished_at": null,
  "message": "Epoch 3/10 train=0.0412 val=0.0518",
  "config": {},
  "metrics": {
    "epoch": 3,
    "total_epochs": 10,
    "train_loss": 0.0412,
    "val_loss": 0.0518,
    "val_rmse": 1.23,
    "val_event_f1": 0.81,
    "elapsed_seconds": 142.5
  },
  "result": null,
  "error": null,
  "gpu": { "available": false, "name": null }
}
```

---

### GET /api/training/history

Returns the most recent training run records (latest first).

**Response:**
```json
{
  "items": [
    {
      "status": "success",
      "finished_at": "2024-04-15T12:30:00",
      "config": {},
      "metrics": {},
      "result": "heatwave_model_checkpoint_v5.pth",
      "error": null
    }
  ]
}
```

---

### POST /api/training/start 🔒

Triggers a background model training job. **Requires `X-API-Key` header** if `API_KEY` env var is set.

**Request Body:**
```json
{
  "epochs": 10,
  "batch_size": 16,
  "learning_rate": 0.001,
  "seq_len": 7
}
```

**Response (`200`):**
```json
{
  "status": "started",
  "message": "Training worker started",
  "config": {},
  "preflight": {}
}
```

**Error Responses:**
- `400` — Invalid training configuration
- `401` — Missing or invalid API key
- `409` — Training is already running

---

### GET /

Redirects to `/trainer`.

---

### GET /trainer

Serves the training console HTML UI (`trainer.html`).

---

### GET /dashboard

Serves the main dashboard HTML UI (`dashboard.html`).

---

## Risk Levels

| Code | Index | Temperature |
|------|-------|-------------|
| LOW | 0 | < 35 °C |
| MEDIUM | 1 | 35–38 °C |
| HIGH | 2 | 38–41 °C |
| CRITICAL | 3 | > 41 °C |
