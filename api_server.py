from flask import Flask, jsonify, render_template
from flask_cors import CORS
from flask import redirect, request
from flasgger import Swagger, swag_from
import threading
import time
import shutil
import subprocess
import torch
import numpy as np
import os
import glob
import datetime
import json
import logging
import pickle
from typing import Optional, Any
import xarray as xr
from Train_Ai import train as run_training
from src.data.loader import (
    load_era5_data,
    create_sequences,
    normalize_data,
    clean_data,
    compute_normalization_stats,
)
from src.models.convlstm import HeatwaveConvLSTM
from api_daily_predict import init_daily_routes  # Daily XGBoost prediction

try:
    from src.data.freshness import get_freshness_summary, is_data_stale
    _freshness_available = True
except ImportError:
    _freshness_available = False

try:
    from src.monitoring.drift_detector import record_prediction as _record_drift, get_drift_status as _get_drift_status
    _drift_available = True
except ImportError:
    _drift_available = False

app = Flask(__name__)
_cors_origins = os.getenv("CORS_ORIGINS", "*")
CORS(app, origins=_cors_origins.split(",") if _cors_origins != "*" else "*")


@app.before_request
def _track_request() -> None:
    _request_stats["total"] += 1
    _request_stats["last_request_time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# --- Authentication ---
_API_KEY = os.getenv("API_KEY", "")


def require_api_key(f):
    """Decorator: require X-API-Key header if API_KEY env var is set."""
    import functools

    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if _API_KEY:
            key = request.headers.get("X-API-Key", "")
            if key != _API_KEY:
                return jsonify({"error": "Unauthorized", "message": "Invalid or missing API key"}), 401
        return f(*args, **kwargs)

    return decorated

swagger_config = {
    "headers": [],
    "specs": [{"endpoint": "apispec", "route": "/api/apispec.json", "rule_filter": lambda rule: True, "model_filter": lambda tag: True}],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/api/docs",
}
swagger_template = {
    "info": {"title": "Heatwave Forecast API", "description": "Thailand Heatwave Forecasting System API", "version": "1.0.0"},
    "basePath": "/",
}
Swagger(app, config=swagger_config, template=swagger_template)

DATA_DIR = "era5_data"
MODELS_DIR = "models"
SEQ_LEN = 7
FUTURE_SEQ = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGGER = logging.getLogger(__name__)
EPSILON = 1e-6
REGION_SAMPLING_RADIUS_DEGREES = 0.75
REGION_SAMPLING_SIGMA_KM = 45.0
CHECKPOINT_PATTERNS = (
    "heatwave_model_checkpoint_v*.pth",
    "heatwave_convlstm_v*.pth",
)
SUPPORTED_LABELING_METHODS = ("temperature", "heat_index", "anomaly")


class _TrainingPollLogFilter(logging.Filter):
    """Suppress very noisy GET poll logs from the web trainer."""

    _noisy_paths = (
        "GET /api/training/status",
        "GET /api/training/preflight",
        "GET /api/training/history",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        return not any(path in message for path in self._noisy_paths)


def _configure_access_log_filter() -> None:
    if os.environ.get("HW_SUPPRESS_TRAINING_POLL_LOGS", "1").strip() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        werk_logger = logging.getLogger("werkzeug")
        if not any(isinstance(f, _TrainingPollLogFilter) for f in werk_logger.filters):
            werk_logger.addFilter(_TrainingPollLogFilter())


_configure_access_log_filter()

# Global variables
model: Optional[Any] = None
X_test = None
Y_test = None
lats = None
lons = None
mean = None
std = None
temp_mean_scalar = None
temp_std_scalar = None
bbox = None
_prediction_cache = {}
runtime_seq_len = SEQ_LEN
runtime_future_seq = FUTURE_SEQ
runtime_model_type = "balanced_random_forest"
_request_stats: dict = {"total": 0, "errors": 0, "predict_calls": 0, "last_request_time": None}
_startup_time: float = time.time()
temperature_forecast_context: Optional[dict[str, Any]] = None
training_lock = threading.Lock()
_resource_lock = threading.RLock()
# Track original model type for fallback
_original_model_loaded = False
# XGBoost model reference
_xgboost_model = None

# Track original data time for accurate date reporting
_test_time_index: Optional[list] = None
_test_time_base: Optional[datetime.datetime] = None
WEB_REGIONS = [
    {"name": "Bangkok", "zone": "Central", "lat": 13.7563, "lng": 100.5018},
    {"name": "Chiang Mai", "zone": "North", "lat": 18.7883, "lng": 98.9853},
    {"name": "Chiang Rai", "zone": "North", "lat": 19.9072, "lng": 99.8329},
    {"name": "Khon Kaen", "zone": "Northeast", "lat": 16.4423, "lng": 102.1426},
    {"name": "Nakhon Si Thammarat", "zone": "South", "lat": 8.4333, "lng": 99.9333},
    {"name": "Surat Thani", "zone": "South", "lat": 9.1401, "lng": 99.3331},
    {"name": "Pattaya", "zone": "Central", "lat": 12.9333, "lng": 100.8833},
    {"name": "Phitsanulok", "zone": "North", "lat": 16.8281, "lng": 100.2624},
    {"name": "Korat", "zone": "Northeast", "lat": 14.9799, "lng": 102.0782},
]
training_state = {
    "status": "idle",
    "started_at": None,
    "finished_at": None,
    "message": "No training started yet.",
    "config": None,
    "metrics": None,
    "result": None,
    "error": None,
}
training_history = []
MAX_TRAINING_HISTORY = 30


def get_latest_model(model_dir):
    if not os.path.exists(model_dir):
        return None
    checkpoint_files = []
    for pattern in CHECKPOINT_PATTERNS:
        checkpoint_files.extend(glob.glob(os.path.join(model_dir, pattern)))

    if not checkpoint_files:
        return None

    def get_version(path):
        try:
            stem = os.path.splitext(os.path.basename(path))[0]
            return int(stem.rsplit("_v", 1)[-1])
        except (ValueError, IndexError):
            return 0

    def checkpoint_sort_key(path):
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0.0
        return (get_version(path), mtime, os.path.basename(path))

    return max(checkpoint_files, key=checkpoint_sort_key)


def _safe_torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location)
    except pickle.UnpicklingError:
        # PyTorch >=2.6 defaults weights_only=True; our checkpoint carries metadata.
        return torch.load(path, map_location=map_location, weights_only=False)


def _to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _build_result_from_metadata(checkpoint_path, metadata, model_type):
    test_metrics = metadata.get("test_metrics") or metadata.get("test_event_metrics") or {}
    train_metrics = (
        metadata.get("train_metrics") or metadata.get("train_event_metrics") or {}
    )
    val_metrics = metadata.get("val_metrics") or metadata.get("val_event_metrics") or {}

    if isinstance(train_metrics, dict) and "accuracy" not in train_metrics:
        train_acc = _calculate_event_accuracy(train_metrics)
        if train_acc is not None:
            train_metrics = dict(train_metrics)
            train_metrics["accuracy"] = train_acc
    if isinstance(test_metrics, dict) and "accuracy" not in test_metrics:
        test_acc = _calculate_event_accuracy(test_metrics)
        if test_acc is not None:
            test_metrics = dict(test_metrics)
            test_metrics["accuracy"] = test_acc
    if isinstance(val_metrics, dict) and "accuracy" not in val_metrics:
        val_acc = _calculate_event_accuracy(val_metrics)
        if val_acc is not None:
            val_metrics = dict(val_metrics)
            val_metrics["accuracy"] = val_acc

    baseline_metrics = metadata.get("baseline_metrics") or {}
    result = {
        "save_path": checkpoint_path,
        "model_type": model_type,
        "epochs": metadata.get("epochs", 1),
        "batch_size": metadata.get("batch_size"),
        "learning_rate": metadata.get("learning_rate"),
        "train_event_metrics": train_metrics,
        "val_event_metrics": val_metrics,
        "test_event_metrics": test_metrics,
        "baseline_metrics": baseline_metrics,
        "seasonal_metrics": metadata.get("seasonal_metrics") or {},
        "monthly_metrics": metadata.get("monthly_metrics") or {},
        "regional_metrics": metadata.get("regional_metrics") or {},
        "data_quality_report": metadata.get("data_quality_report") or {},
        "labeling_method": metadata.get("labeling_method"),
        "heatwave_threshold_c": metadata.get("heatwave_threshold_c"),
        "heatwave_heat_index_threshold": metadata.get("heatwave_heat_index_threshold"),
        "heatwave_temperature_threshold": metadata.get("heatwave_temperature_threshold"),
        "heatwave_anomaly_threshold": metadata.get("heatwave_anomaly_threshold"),
        "threshold_selection_mode": metadata.get("threshold_selection_mode"),
        "event_probability_threshold": metadata.get("event_probability_threshold"),
        "train_positive_count": metadata.get("train_positive_count"),
        "val_positive_count": metadata.get("val_positive_count"),
        "test_positive_count": metadata.get("test_positive_count"),
        "train_positive_rate": metadata.get("train_positive_rate"),
        "val_positive_rate": metadata.get("val_positive_rate"),
        "test_positive_rate": metadata.get("test_positive_rate"),
    }
    return _to_jsonable(result)


def _build_config_from_metadata(metadata):
    keys = [
        "batch_size",
        "seq_len",
        "future_seq",
        "epochs",
        "learning_rate",
        "model_backend",
        "use_gpu",
        "force_gpu",
        "rf_n_estimators",
        "rf_max_depth",
        "rf_min_samples_leaf",
        "rf_sampling_strategy",
        "rf_replacement",
        "train_ratio",
        "val_ratio",
        "heatwave_percentile",
        "clip_low_percentile",
        "clip_high_percentile",
        "event_min_duration_days",
        "event_min_hot_fraction",
        "allow_sample_mean_fallback",
        "require_dynamic_features",
        "min_train_positive_rate",
        "max_train_positive_rate",
        "min_eval_positive_count",
        "labeling_method",
        "heatwave_heat_index_threshold",
        "heatwave_temperature_threshold",
        "heatwave_anomaly_threshold",
    ]
    config = {}
    for key in keys:
        if key in metadata and metadata.get(key) is not None:
            config[key] = _to_jsonable(metadata.get(key))
    return config


def _calculate_event_accuracy(metrics):
    if not isinstance(metrics, dict):
        return None
    tp = metrics.get("tp")
    tn = metrics.get("tn")
    fp = metrics.get("fp")
    fn = metrics.get("fn")
    if any(value is None for value in (tp, tn, fp, fn)):
        return None
    try:
        tp_f = float(tp)
        tn_f = float(tn)
        fp_f = float(fp)
        fn_f = float(fn)
        total = tp_f + tn_f + fp_f + fn_f
        if total <= 0:
            return None
        return float((tp_f + tn_f) / total)
    except Exception:
        return None


def _as_coord_array(values, axis_name):
    if values is None:
        return None
    try:
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
    except Exception:
        LOGGER.warning("Unable to parse checkpoint %s coordinates", axis_name)
        return None
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        LOGGER.warning("Checkpoint %s coordinates are empty or invalid", axis_name)
        return None
    return arr


def _coords_match(source, target, atol=1e-6):
    return (
        source is not None
        and target is not None
        and source.shape == target.shape
        and np.allclose(source, target, atol=atol)
    )


def _align_data_to_checkpoint_grid(data, source_lats, source_lons, metadata):
    target_lats = _as_coord_array(metadata.get("lats"), "latitude")
    target_lons = _as_coord_array(metadata.get("lons"), "longitude")
    if target_lats is None or target_lons is None:
        return data, source_lats, source_lons, False

    source_lats = np.asarray(source_lats, dtype=np.float32).reshape(-1)
    source_lons = np.asarray(source_lons, dtype=np.float32).reshape(-1)

    if _coords_match(source_lats, target_lats) and _coords_match(source_lons, target_lons):
        return data, source_lats, source_lons, False

    LOGGER.warning(
        "Spatial grid mismatch between checkpoint and runtime data. "
        "Regridding API data from (%s lat, %s lon) to (%s lat, %s lon).",
        len(source_lats),
        len(source_lons),
        len(target_lats),
        len(target_lons),
    )

    aligned = xr.DataArray(
        data,
        dims=("time", "channel", "latitude", "longitude"),
        coords={"latitude": source_lats, "longitude": source_lons},
    ).interp(latitude=target_lats, longitude=target_lons, method="linear")

    aligned_data = np.asarray(
        aligned.transpose("time", "channel", "latitude", "longitude").values,
        dtype=np.float32,
    )
    np.nan_to_num(aligned_data, copy=False)
    return aligned_data, target_lats, target_lons, True


def _refresh_static_coordinate_channels(data, lats, lons):
    channel_layouts = {
        6: (4, 5),
        8: (6, 7),
    }
    coord_indices = channel_layouts.get(int(data.shape[1]))
    if coord_indices is None:
        return data

    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    lat_idx, lon_idx = coord_indices
    data[:, lat_idx, :, :] = lat_grid[None, :, :]
    data[:, lon_idx, :, :] = lon_grid[None, :, :]
    return data


RISK_LEVELS = (
    {
        "index": 0,
        "code": "LOW",
        "label": "LOW",
        "default_probability": 0.1,
    },
    {
        "index": 1,
        "code": "MEDIUM",
        "label": "MEDIUM",
        "default_probability": 0.5,
    },
    {
        "index": 2,
        "code": "HIGH",
        "label": "HIGH",
        "default_probability": 0.8,
    },
    {
        "index": 3,
        "code": "CRITICAL",
        "label": "CRITICAL",
        "default_probability": 0.95,
    },
)
RISK_CODE_TO_LEVEL = {entry["code"]: entry for entry in RISK_LEVELS}
RISK_INDEX_TO_LEVEL = {entry["index"]: entry for entry in RISK_LEVELS}
# Temperature thresholds for Thailand heatwave risk classification
# 28.6°C is normal/cool for Thailand (LOW risk)
RISK_TEMP_THRESHOLDS = (
    (40.0, "CRITICAL"),  # Extreme heatwave - above 40°C
    (38.0, "HIGH"),      # Dangerous heat - 38-40°C
    (35.0, "MEDIUM"),    # Elevated heat - 35-38°C
)


def _coerce_risk_index(value):
    try:
        risk_index = int(value)
    except (TypeError, ValueError):
        risk_index = 0
    max_index = max(RISK_INDEX_TO_LEVEL)
    return int(min(max(risk_index, 0), max_index))


def _coerce_risk_code(value):
    code = str(value).strip().upper()
    if code in RISK_CODE_TO_LEVEL:
        return code
    return "LOW"


def _risk_code_for_temperature(max_temp):
    for threshold, code in RISK_TEMP_THRESHOLDS:
        if max_temp >= threshold:
            return code
    return "LOW"


def _risk_probability_for_code(risk_code, model_probability=None):
    if model_probability is not None:
        try:
            return float(np.clip(float(model_probability), 0.0, 1.0))
        except (TypeError, ValueError):
            pass
    return float(RISK_CODE_TO_LEVEL[risk_code]["default_probability"])


def _build_risk_payload_from_code(risk_code, model_probability=None):
    code = _coerce_risk_code(risk_code)
    level = RISK_CODE_TO_LEVEL[code]
    payload = {
        "risk_code": code,
        "risk_label": level["label"],
        "risk_index": int(level["index"]),
    }
    if model_probability is not None:
        payload["probability"] = _risk_probability_for_code(
            code, model_probability=model_probability
        )
    return payload


def _build_risk_payload_from_temperature(max_temp, model_probability=None):
    risk_code = _risk_code_for_temperature(float(max_temp))
    payload = _build_risk_payload_from_code(risk_code)
    payload["probability"] = _risk_probability_for_code(
        risk_code, model_probability=model_probability
    )
    return payload


def _build_risk_payload_from_index(risk_index):
    level = RISK_INDEX_TO_LEVEL[_coerce_risk_index(risk_index)]
    return {
        "risk_code": level["code"],
        "risk_label": level["label"],
        "risk_index": int(level["index"]),
    }


def _risk_legend():
    return [
        {
            "risk_index": int(level["index"]),
            "risk_code": level["code"],
            "risk_label": level["label"],
        }
        for level in RISK_LEVELS
    ]


def get_risk_and_probability(max_temp, model_probability=None):
    risk_payload = _build_risk_payload_from_temperature(
        max_temp, model_probability=model_probability
    )
    return risk_payload["risk_code"], risk_payload["probability"]


def detect_gpu_capability():
    torch_cuda = bool(torch.cuda.is_available())
    nvidia_smi = False
    candidates = []

    found = shutil.which("nvidia-smi")
    if found:
        candidates.append(found)
    candidates.extend(
        [
            "nvidia-smi",
            r"C:\Windows\System32\nvidia-smi.exe",
            r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
        ]
    )

    for candidate in candidates:
        try:
            proc = subprocess.run(
                [candidate, "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                nvidia_smi = True
                break
        except Exception:
            continue

    return {
        "gpu_detected": bool(torch_cuda or nvidia_smi),
        "torch_cuda": torch_cuda,
        "nvidia_smi": nvidia_smi,
    }


def get_default_training_config():
    gpu = detect_gpu_capability()
    return {
        "batch_size": 4,
        "seq_len": 7,
        "future_seq": 2,
        "epochs": 20,
        "learning_rate": 1e-3,
        "model_backend": "balanced_rf",
        "use_gpu": False,
        "force_gpu": False,
        "rf_n_estimators": 300,
        "rf_max_depth": 10,
        "rf_min_samples_leaf": 2,
        "rf_sampling_strategy": "all",
        "rf_replacement": True,
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "heatwave_percentile": 95.0,
        "clip_low_percentile": 0.5,
        "clip_high_percentile": 99.5,
        "event_min_duration_days": 3,
        "event_min_hot_fraction": 0.10,
        "allow_sample_mean_fallback": True,
        "require_dynamic_features": False,
        "min_train_positive_rate": 0.01,
        "max_train_positive_rate": 0.35,
        "min_eval_positive_count": 10,
        "labeling_method": "temperature",
        "heatwave_heat_index_threshold": 41.0,
        "heatwave_temperature_threshold": 35.0,
        "heatwave_anomaly_threshold": 6.0,
    }


def _coerce_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return None


def _coerce_int(config, payload, key, min_value, max_value, errors):
    value = payload.get(key, config[key])
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        errors.append(f"{key} must be an integer")
        return
    if parsed < min_value or parsed > max_value:
        errors.append(f"{key} must be between {min_value} and {max_value}")
        return
    config[key] = parsed


def _coerce_float(config, payload, key, min_value, max_value, errors):
    value = payload.get(key, config[key])
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        errors.append(f"{key} must be a number")
        return
    if parsed < min_value or parsed > max_value:
        errors.append(f"{key} must be between {min_value} and {max_value}")
        return
    config[key] = parsed


def sanitize_training_config(payload):
    config = get_default_training_config()
    errors = []

    if not isinstance(payload, dict):
        payload = {}

    backend_raw = str(payload.get("model_backend", config["model_backend"]))
    backend = backend_raw.strip().lower().replace("-", "_")
    if backend in {
        "balanced_random_forest",
        "random_forest",
        "xgb",
        "xgboost",
        "xgboost_classifier",
        "xgboost_gpu_classifier",
    }:
        backend = "balanced_rf"
    if backend != "balanced_rf":
        errors.append("model_backend must be balanced_rf")
    config["model_backend"] = "balanced_rf"

    _coerce_int(config, payload, "batch_size", 1, 1024, errors)
    _coerce_int(config, payload, "seq_len", 1, 64, errors)
    _coerce_int(config, payload, "future_seq", 1, 32, errors)
    _coerce_int(config, payload, "epochs", 1, 1000, errors)
    _coerce_int(config, payload, "rf_n_estimators", 10, 5000, errors)
    _coerce_int(config, payload, "rf_max_depth", 1, 512, errors)
    _coerce_int(config, payload, "rf_min_samples_leaf", 1, 128, errors)
    _coerce_int(config, payload, "event_min_duration_days", 1, 14, errors)
    _coerce_int(config, payload, "min_eval_positive_count", 1, 5000, errors)

    _coerce_float(config, payload, "learning_rate", 1e-6, 1.0, errors)
    _coerce_float(config, payload, "train_ratio", 0.05, 0.95, errors)
    _coerce_float(config, payload, "val_ratio", 0.01, 0.9, errors)
    _coerce_float(config, payload, "heatwave_percentile", 80.0, 99.99, errors)
    _coerce_float(config, payload, "clip_low_percentile", 0.0, 20.0, errors)
    _coerce_float(config, payload, "clip_high_percentile", 80.0, 100.0, errors)
    _coerce_float(config, payload, "event_min_hot_fraction", 0.01, 1.0, errors)
    _coerce_float(config, payload, "min_train_positive_rate", 0.001, 0.95, errors)
    _coerce_float(config, payload, "max_train_positive_rate", 0.01, 0.99, errors)
    _coerce_float(config, payload, "heatwave_heat_index_threshold", 20.0, 70.0, errors)
    _coerce_float(config, payload, "heatwave_temperature_threshold", 25.0, 50.0, errors)
    _coerce_float(config, payload, "heatwave_anomaly_threshold", 0.1, 20.0, errors)

    for key in (
        "use_gpu",
        "force_gpu",
        "rf_replacement",
        "allow_sample_mean_fallback",
        "require_dynamic_features",
    ):
        parsed = _coerce_bool(payload.get(key, config[key]))
        if parsed is None:
            errors.append(f"{key} must be a boolean")
        else:
            config[key] = parsed

    labeling_method = str(payload.get("labeling_method", config["labeling_method"])).strip().lower()
    if labeling_method not in SUPPORTED_LABELING_METHODS:
        allowed = "', '".join(SUPPORTED_LABELING_METHODS)
        errors.append(f"labeling_method must be one of '{allowed}'")
    else:
        config["labeling_method"] = labeling_method

    # BalancedRandomForest backend runs on CPU in this project.
    config["use_gpu"] = False
    config["force_gpu"] = False

    sampling = str(payload.get("rf_sampling_strategy", config["rf_sampling_strategy"]))
    sampling = sampling.strip().lower()
    if sampling not in {"all", "auto", "majority", "not minority", "not majority"}:
        errors.append("rf_sampling_strategy is not supported")
    else:
        config["rf_sampling_strategy"] = sampling

    if (float(config["train_ratio"]) + float(config["val_ratio"])) >= 1.0:
        errors.append("train_ratio + val_ratio must be less than 1.0")

    if float(config["clip_low_percentile"]) >= float(config["clip_high_percentile"]):
        errors.append("clip_low_percentile must be less than clip_high_percentile")
    if float(config["min_train_positive_rate"]) >= float(
        config["max_train_positive_rate"]
    ):
        errors.append("min_train_positive_rate must be less than max_train_positive_rate")

    return config, errors


def training_preflight_summary():
    data_files = glob.glob(os.path.join(DATA_DIR, "*.nc"))
    model_files = []
    for pattern in CHECKPOINT_PATTERNS:
        model_files.extend(glob.glob(os.path.join(MODELS_DIR, pattern)))

    issues = []
    if not os.path.isdir(DATA_DIR):
        issues.append(f"{DATA_DIR}/ directory is missing")
    elif len(data_files) == 0:
        issues.append("No .nc files found in era5_data")

    if not os.path.isdir(MODELS_DIR):
        issues.append(f"{MODELS_DIR}/ directory is missing")

    return {
        "data_dir_exists": os.path.isdir(DATA_DIR),
        "models_dir_exists": os.path.isdir(MODELS_DIR),
        "data_file_count": len(data_files),
        "checkpoint_count": len(model_files),
        "resources_loaded": resources_ready(),
        "gpu": detect_gpu_capability(),
        "issues": issues,
        "ready_for_training": len(data_files) > 0,
    }


def snapshot_training_state():
    with training_lock:
        return dict(training_state)


def set_training_state(**updates):
    with training_lock:
        training_state.update(updates)


def append_training_history(entry):
    with training_lock:
        training_history.insert(0, _to_jsonable(entry))
        if len(training_history) > MAX_TRAINING_HISTORY:
            del training_history[MAX_TRAINING_HISTORY:]


def snapshot_training_history():
    with training_lock:
        return list(training_history)


def _build_epoch_callback():
    def on_epoch_end(metrics):
        set_training_state(
            metrics={
                "epoch": metrics.epoch,
                "total_epochs": metrics.total_epochs,
                "train_loss": metrics.train_loss,
                "val_loss": metrics.val_loss,
                "val_rmse": metrics.val_rmse,
                "val_event_f1": metrics.val_event_f1,
                "elapsed_seconds": metrics.elapsed_seconds,
            },
            message=(
                f"Epoch {metrics.epoch}/{metrics.total_epochs} "
                f"train={metrics.train_loss:.4f} val={metrics.val_loss:.4f}"
            ),
        )

    return on_epoch_end


def run_training_job(config):
    set_training_state(
        status="running",
        started_at=datetime.datetime.now().isoformat(),
        finished_at=None,
        message="Training started.",
        config=config,
        metrics=None,
        result=None,
        error=None,
    )

    try:
        result = run_training(config=config, on_epoch_end=_build_epoch_callback())
        if not result:
            set_training_state(
                status="failed",
                finished_at=datetime.datetime.now().isoformat(),
                message="Training failed. Check server logs.",
                error="train() returned no result",
            )
            return

        progress_metrics = snapshot_training_state().get("metrics") or {}
        checkpoint_path = result.get("save_path")
        reloaded = load_resources(checkpoint_path=checkpoint_path, skip_data_reload=True)
        if not reloaded:
            # Fallback to full reload to keep compatibility when data is missing.
            load_resources(checkpoint_path=checkpoint_path, skip_data_reload=False)
        train_metrics = result.get("train_event_metrics") or {}
        test_metrics = result.get("test_event_metrics") or result.get("test_metrics") or {}
        train_accuracy = _calculate_event_accuracy(train_metrics)
        test_accuracy = _calculate_event_accuracy(test_metrics)
        merged_metrics = dict(progress_metrics)
        merged_metrics["train_accuracy"] = (
            train_metrics.get("accuracy")
            if isinstance(train_metrics, dict) and train_metrics.get("accuracy") is not None
            else train_accuracy
        )
        merged_metrics["test_accuracy"] = (
            test_metrics.get("accuracy")
            if isinstance(test_metrics, dict) and test_metrics.get("accuracy") is not None
            else test_accuracy
        )
        merged_metrics["test_f1"] = (
            test_metrics.get("f1") if isinstance(test_metrics, dict) else None
        )
        merged_metrics["test_brier_score"] = (
            test_metrics.get("brier_score") if isinstance(test_metrics, dict) else None
        )

        set_training_state(
            status="completed",
            finished_at=datetime.datetime.now().isoformat(),
            message="Training completed and model reloaded.",
            result=result,
            metrics=merged_metrics,
            error=None,
        )
        append_training_history(
            {
                "status": "completed",
                "finished_at": datetime.datetime.now().isoformat(),
                "config": config,
                "metrics": merged_metrics,
                "result": result,
                "error": None,
            }
        )
    except Exception as exc:
        LOGGER.exception("Training job failed: %s", exc)
        set_training_state(
            status="failed",
            finished_at=datetime.datetime.now().isoformat(),
            message="Training failed.",
            error=str(exc),
        )
        append_training_history(
            {
                "status": "failed",
                "finished_at": datetime.datetime.now().isoformat(),
                "config": config,
                "metrics": snapshot_training_state().get("metrics"),
                "result": None,
                "error": str(exc),
            }
        )


def resources_ready():
    return (
        model is not None
        and X_test is not None
        and len(X_test) > 0
        and lats is not None
        and lons is not None
        and mean is not None
        and std is not None
    )


def temperature_forecast_ready() -> bool:
    if runtime_model_type == "convlstm":
        return resources_ready()
    return bool(
        temperature_forecast_context
        and temperature_forecast_context.get("model") is not None
        and temperature_forecast_context.get("X_test") is not None
        and len(temperature_forecast_context.get("X_test")) > 0
    )


def get_sample_date(sample_idx=-1, day_offset=0):
    """Get the date for a sample index with optional day offset.
    
    Returns the actual date from the test set time index if available,
    otherwise falls back to datetime.now() with offset.
    """
    global _test_time_base, runtime_seq_len, runtime_future_seq
    
    if _test_time_base is not None:
        # Calculate the date based on sample position in test set
        # sample_idx=-1 means last sample, which is the most recent
        base_date = _test_time_base
        if sample_idx == -1:
            # Last sample = most recent date in test set
            return base_date + datetime.timedelta(days=day_offset)
        else:
            # Other samples are earlier
            return base_date + datetime.timedelta(days=sample_idx + day_offset)
    
    # Fallback: use datetime.now() with offset from sample_idx
    base_date = datetime.datetime.now()
    if sample_idx == -1:
        # For last sample, estimate based on sequence length
        return base_date + datetime.timedelta(days=day_offset)
    else:
        return base_date + datetime.timedelta(days=sample_idx + day_offset)


def _parse_checkpoint_times(metadata):
    """Parse time information from checkpoint metadata."""
    global _test_time_base
    
    # Try to get time info from checkpoint metadata
    time_index = metadata.get("time_index", [])
    if time_index and len(time_index) > 0:
        try:
            # Parse the last timestamp
            last_time_str = time_index[-1]
            _test_time_base = datetime.datetime.fromisoformat(last_time_str.replace("Z", "+00:00"))
            return
        except (ValueError, TypeError):
            pass
    
    # Try to get from data_time_base
    data_time_base = metadata.get("data_time_base")
    if data_time_base:
        try:
            _test_time_base = datetime.datetime.fromisoformat(str(data_time_base).replace("Z", "+00:00"))
            return
        except (ValueError, TypeError):
            pass
    
    # No time info available, will use datetime.now() fallback
    LOGGER.warning("No time information found in checkpoint, using datetime.now() for dates")
    _test_time_base = None


def load_resources(checkpoint_path=None, skip_data_reload=False):
    global model, X_test, Y_test, lats, lons, mean, std
    global temp_mean_scalar, temp_std_scalar, bbox, _prediction_cache
    global runtime_seq_len, runtime_future_seq, runtime_model_type
    global temperature_forecast_context

    checkpoint_path = checkpoint_path or get_latest_model(MODELS_DIR)
    if not checkpoint_path:
        LOGGER.error("No model found in %s", MODELS_DIR)
        return False

    print(f"Loading model from {checkpoint_path}...")
    checkpoint = _safe_torch_load(checkpoint_path, map_location=DEVICE)
    metadata = checkpoint.get("metadata", {}) if isinstance(checkpoint, dict) else {}

    # Parse time information from checkpoint for accurate date reporting
    _parse_checkpoint_times(metadata)

    runtime_seq_len = int(metadata.get("seq_len", SEQ_LEN))
    runtime_future_seq = int(metadata.get("future_seq", FUTURE_SEQ))
    # Branch on model type
    if "sklearn_model" in checkpoint:
        runtime_model_type = "balanced_random_forest"
        model = checkpoint.get("sklearn_model")
        if model is None:
            LOGGER.error(
                "Checkpoint is missing sklearn_model for BalancedRandomForest inference"
            )
            return False
    elif "model_state_dict" in checkpoint:
        runtime_model_type = "convlstm"
        raw_sd = checkpoint["model_state_dict"]
        # Remap legacy key names produced by older model definitions:
        #   cell_list.N.conv.*  ->  encoder_cells.N.conv.*
        #   final_conv.*        ->  output_conv.*
        remapped = {}
        for k, v in raw_sd.items():
            nk = k.replace("cell_list.", "encoder_cells.").replace("final_conv.", "output_conv.")
            remapped[nk] = v

        hidden_dim = metadata.get("hidden_dim", checkpoint.get("hidden_dim", [32, 32]))
        kernel_size = metadata.get(
            "kernel_size", checkpoint.get("kernel_size", [(3, 3), (3, 3)])
        )
        num_layers = int(metadata.get("num_layers", checkpoint.get("num_layers", 2)))

        # Infer input_dim from the first encoder cell weight when metadata is absent.
        # Weight shape: [4*hidden_dim, input_dim + hidden_dim, kH, kW]
        if metadata.get("input_dim") or checkpoint.get("input_dim"):
            input_dim = int(metadata.get("input_dim", checkpoint.get("input_dim", 8)))
        else:
            first_w = remapped.get("encoder_cells.0.conv.weight")
            if first_w is not None:
                in_channels = first_w.shape[1]  # input_dim + hidden_dim[0]
                input_dim = in_channels - hidden_dim[0]
                LOGGER.info(
                    "Inferred input_dim=%d from checkpoint weight shape %s",
                    input_dim, list(first_w.shape),
                )
            else:
                input_dim = 8

        model = HeatwaveConvLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            num_layers=num_layers,
        ).to(DEVICE)
        model.load_state_dict(remapped)
        model.eval()
        LOGGER.info("Loaded ConvLSTM model: input_dim=%d", input_dim)
        # Write back so channel-slicing logic below (expected_input_dim) picks it up.
        metadata["input_dim"] = input_dim
    else:
        LOGGER.error("Checkpoint format not recognized: keys=%s", list(checkpoint.keys()))
        return False

    if skip_data_reload and resources_ready():
        with _resource_lock:
            _prediction_cache = {}
        print("Model reloaded without full data refresh.")
        return True

    print("Loading data...")
    try:
        data_raw, lats, lons, data_mean, data_std = load_era5_data(
            DATA_DIR, normalize=False
        )
    except Exception as e:
        LOGGER.exception("Error loading data: %s", e)
        return False

    data_raw, lats, lons, grid_aligned = _align_data_to_checkpoint_grid(
        data_raw, lats, lons, metadata
    )
    if grid_aligned:
        data_raw = _refresh_static_coordinate_channels(data_raw, lats, lons)
        data_mean, data_std = compute_normalization_stats(data_raw)

    full_data_raw = np.array(data_raw, copy=True)
    full_data_mean = np.array(data_mean, copy=True)
    full_data_std = np.array(data_std, copy=True)

    expected_input_dim = int(metadata.get("input_dim", data_raw.shape[1]))
    current_input_dim = int(data_raw.shape[1])
    if current_input_dim != expected_input_dim:
        # Backward compatibility: old checkpoints used 6 channels
        # [z, t2m, swvl1, elev, lat, lon].
        # New loader emits 8 channels:
        # [z, t2m, swvl1, tp, humidity, elev, lat, lon].
        if expected_input_dim == 6 and current_input_dim >= 8:
            legacy_indices = [0, 1, 2, 5, 6, 7]
            data_raw = data_raw[:, legacy_indices, :, :]
            data_mean = data_mean[:, legacy_indices, :, :]
            data_std = data_std[:, legacy_indices, :, :]
            LOGGER.info(
                "Channel mapping applied for legacy checkpoint: 8 -> 6 channels"
            )
        elif expected_input_dim < current_input_dim:
            data_raw = data_raw[:, :expected_input_dim, :, :]
            data_mean = data_mean[:, :expected_input_dim, :, :]
            data_std = data_std[:, :expected_input_dim, :, :]
            LOGGER.info(
                "Truncated input channels for checkpoint compatibility: %s -> %s",
                current_input_dim,
                expected_input_dim,
            )
        else:
            LOGGER.error(
                "Checkpoint expects %s channels but data loader provides %s",
                expected_input_dim,
                current_input_dim,
            )
            return False

    expected_flat_dim = None
    if runtime_model_type != "convlstm":
        expected_flat_dim = int(getattr(model, "n_features_in_", 0) or 0)
        current_flat_dim = int(
            runtime_seq_len * data_raw.shape[1] * data_raw.shape[2] * data_raw.shape[3]
        )
        if expected_flat_dim and current_flat_dim != expected_flat_dim:
            LOGGER.error(
                "Feature layout mismatch after resource load: expected_flat=%s actual_flat=%s "
                "(seq_len=%s channels=%s lat=%s lon=%s)",
                expected_flat_dim,
                current_flat_dim,
                runtime_seq_len,
                data_raw.shape[1],
                data_raw.shape[2],
                data_raw.shape[3],
            )
            return False

    ckpt_mean = metadata.get("normalization_mean")
    ckpt_std = metadata.get("normalization_std")
    if ckpt_mean is not None and ckpt_std is not None:
        ckpt_mean_arr = np.asarray(ckpt_mean)
        ckpt_std_arr = np.asarray(ckpt_std)
        if (
            ckpt_mean_arr.ndim == 4
            and ckpt_std_arr.ndim == 4
            and ckpt_mean_arr.shape[1] == data_raw.shape[1]
            and ckpt_std_arr.shape[1] == data_raw.shape[1]
        ):
            _new_mean = ckpt_mean_arr
            _new_std = ckpt_std_arr
        else:
            LOGGER.warning(
                "Checkpoint normalization stats channel mismatch (ckpt=%s, data=%s). "
                "Using data-derived stats.",
                ckpt_mean_arr.shape,
                data_raw.shape,
            )
            _new_mean = data_mean
            _new_std = data_std
    else:
        _new_mean = data_mean
        _new_std = data_std
    data_raw, _clip_bounds = clean_data(data_raw)
    data = normalize_data(data_raw, _new_mean, _new_std)

    # Create sequences
    # We take the last part for testing/demo
    X, Y = create_sequences(data, seq_len=runtime_seq_len, pred_len=runtime_future_seq)
    if len(X) == 0:
        LOGGER.error("No sequences available from loaded data")
        return False

    split_idx = int(len(X) * 0.85)
    _new_X_test = X[split_idx:]
    _new_Y_test = Y[split_idx:]

    # Load temperature stats - prefer Celsius from checkpoint if available
    # Train_Ai.py saves temp_mean in Celsius (converted from Kelvin at line 1588)
    checkpoint_temp_mean = metadata.get("temp_mean")
    checkpoint_temp_std = metadata.get("temp_std")

    if checkpoint_temp_mean is not None and checkpoint_temp_std is not None:
        # Use the Celsius values saved by Train_Ai.py
        _new_temp_mean = float(checkpoint_temp_mean)
        _new_temp_std = float(checkpoint_temp_std)
        LOGGER.info(f"Using checkpoint temperature stats: mean={_new_temp_mean:.2f}°C, std={_new_temp_std:.2f}")
    else:
        # Fallback: extract from normalization stats (likely in Kelvin)
        _new_temp_mean = float(_new_mean[0, 1, 0, 0])
        _new_temp_std = float(_new_std[0, 1, 0, 0])
        LOGGER.warning(f"No checkpoint temp_mean found, using normalization stats: mean={_new_temp_mean:.2f}")

    _new_bbox = {
        "north": float(np.max(lats)),
        "south": float(np.min(lats)),
        "east": float(np.max(lons)),
        "west": float(np.min(lons)),
    }
    _new_tfc = None
    if runtime_model_type != "convlstm":
        _new_tfc = _load_secondary_temperature_context(
            primary_checkpoint_path=checkpoint_path,
            data_raw=full_data_raw,
            data_mean=full_data_mean,
            data_std=full_data_std,
        )
    with _resource_lock:
        mean = _new_mean
        std = _new_std
        X_test = _new_X_test
        Y_test = _new_Y_test
        temp_mean_scalar = _new_temp_mean
        temp_std_scalar = _new_temp_std
        bbox = _new_bbox
        temperature_forecast_context = _new_tfc
        _prediction_cache = {}

    print(f"Loaded {len(X_test)} test samples.")

    # Bootstrap dashboard metrics from latest checkpoint on startup.
    state_snapshot = snapshot_training_state()
    if state_snapshot.get("status") == "idle" and not state_snapshot.get("result"):
        checkpoint_result = _build_result_from_metadata(
            checkpoint_path=checkpoint_path,
            metadata=metadata,
            model_type=runtime_model_type,
        )
        checkpoint_metrics = checkpoint_result.get("test_event_metrics") or {}
        checkpoint_train_metrics = checkpoint_result.get("train_event_metrics") or {}
        set_training_state(
            status="ready",
            started_at=None,
            finished_at=datetime.datetime.now().isoformat(),
            message=f"Loaded checkpoint: {os.path.basename(checkpoint_path)}",
            config=_build_config_from_metadata(metadata),
            metrics={
                "train_accuracy": _calculate_event_accuracy(checkpoint_train_metrics),
                "test_accuracy": _calculate_event_accuracy(checkpoint_metrics),
                "test_f1": checkpoint_metrics.get("f1")
                if isinstance(checkpoint_metrics, dict)
                else None,
                "test_brier_score": checkpoint_metrics.get("brier_score")
                if isinstance(checkpoint_metrics, dict)
                else None,
            },
            result=checkpoint_result,
            error=None,
        )
        append_training_history(
            {
                "status": "ready",
                "finished_at": datetime.datetime.now().isoformat(),
                "config": _build_config_from_metadata(metadata),
                "metrics": {
                    "train_accuracy": _calculate_event_accuracy(checkpoint_train_metrics),
                    "test_accuracy": _calculate_event_accuracy(checkpoint_metrics),
                    "test_f1": checkpoint_metrics.get("f1")
                    if isinstance(checkpoint_metrics, dict)
                    else None,
                    "test_brier_score": checkpoint_metrics.get("brier_score")
                    if isinstance(checkpoint_metrics, dict)
                    else None,
                },
                "result": checkpoint_result,
                "error": None,
            }
        )

    print("Model loaded successfully.")
    return True


def _build_convlstm_from_checkpoint(checkpoint, metadata):
    raw_sd = checkpoint["model_state_dict"]
    remapped = {}
    for key, value in raw_sd.items():
        nk = key.replace("cell_list.", "encoder_cells.").replace("final_conv.", "output_conv.")
        remapped[nk] = value

    hidden_dim = metadata.get("hidden_dim", checkpoint.get("hidden_dim", [32, 32]))
    kernel_size = metadata.get("kernel_size", checkpoint.get("kernel_size", [(3, 3), (3, 3)]))
    num_layers = int(metadata.get("num_layers", checkpoint.get("num_layers", 2)))

    if metadata.get("input_dim") or checkpoint.get("input_dim"):
        input_dim = int(metadata.get("input_dim", checkpoint.get("input_dim", 8)))
    else:
        first_w = remapped.get("encoder_cells.0.conv.weight")
        if first_w is not None:
            input_dim = int(first_w.shape[1] - hidden_dim[0])
        else:
            input_dim = 8

    conv_model = HeatwaveConvLSTM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        kernel_size=kernel_size,
        num_layers=num_layers,
    ).to(DEVICE)
    conv_model.load_state_dict(remapped)
    conv_model.eval()
    metadata["input_dim"] = input_dim
    return conv_model, metadata


def _prepare_sequences_for_checkpoint(data_raw, data_mean, data_std, metadata):
    prepared_raw = np.array(data_raw, copy=True)
    prepared_mean = np.array(data_mean, copy=True)
    prepared_std = np.array(data_std, copy=True)

    expected_input_dim = int(metadata.get("input_dim", prepared_raw.shape[1]))
    current_input_dim = int(prepared_raw.shape[1])
    if current_input_dim != expected_input_dim:
        if expected_input_dim == 6 and current_input_dim >= 8:
            legacy_indices = [0, 1, 2, 5, 6, 7]
            prepared_raw = prepared_raw[:, legacy_indices, :, :]
            prepared_mean = prepared_mean[:, legacy_indices, :, :]
            prepared_std = prepared_std[:, legacy_indices, :, :]
        elif expected_input_dim < current_input_dim:
            prepared_raw = prepared_raw[:, :expected_input_dim, :, :]
            prepared_mean = prepared_mean[:, :expected_input_dim, :, :]
            prepared_std = prepared_std[:, :expected_input_dim, :, :]
        else:
            raise RuntimeError(
                f"Checkpoint expects {expected_input_dim} channels but runtime data provides {current_input_dim}"
            )

    ckpt_mean = metadata.get("normalization_mean")
    ckpt_std = metadata.get("normalization_std")
    if ckpt_mean is not None and ckpt_std is not None:
        ckpt_mean_arr = np.asarray(ckpt_mean)
        ckpt_std_arr = np.asarray(ckpt_std)
        if (
            ckpt_mean_arr.ndim == 4
            and ckpt_std_arr.ndim == 4
            and ckpt_mean_arr.shape[1] == prepared_raw.shape[1]
            and ckpt_std_arr.shape[1] == prepared_raw.shape[1]
        ):
            prepared_mean = ckpt_mean_arr
            prepared_std = ckpt_std_arr

    prepared_raw, _ = clean_data(prepared_raw)
    prepared = normalize_data(prepared_raw, prepared_mean, prepared_std)
    seq_len = int(metadata.get("seq_len", SEQ_LEN))
    future_seq = int(metadata.get("future_seq", FUTURE_SEQ))
    x_all, y_all = create_sequences(prepared, seq_len=seq_len, pred_len=future_seq)
    if len(x_all) == 0:
        raise RuntimeError("No sequences available for checkpoint-aligned inference")

    split_idx = int(len(x_all) * 0.85)
    x_test_local = x_all[split_idx:]
    y_test_local = y_all[split_idx:]
    if len(x_test_local) == 0:
        raise RuntimeError("No test sequences available after split")

    checkpoint_temp_mean = metadata.get("temp_mean")
    checkpoint_temp_std = metadata.get("temp_std")
    if checkpoint_temp_mean is not None and checkpoint_temp_std is not None:
        local_temp_mean = float(checkpoint_temp_mean)
        local_temp_std = float(checkpoint_temp_std)
    else:
        local_temp_mean = float(prepared_mean[0, 1, 0, 0])
        local_temp_std = float(prepared_std[0, 1, 0, 0])

    return {
        "X_test": x_test_local,
        "Y_test": y_test_local,
        "mean": prepared_mean,
        "std": prepared_std,
        "temp_mean_scalar": local_temp_mean,
        "temp_std_scalar": local_temp_std,
        "seq_len": seq_len,
        "future_seq": future_seq,
    }


def _load_secondary_temperature_context(primary_checkpoint_path, data_raw, data_mean, data_std):
    convlstm_files = glob.glob(os.path.join(MODELS_DIR, "heatwave_convlstm_v*.pth"))
    if not convlstm_files:
        return None

    primary_abs = os.path.abspath(primary_checkpoint_path)
    candidates = [path for path in convlstm_files if os.path.abspath(path) != primary_abs]
    if not candidates:
        return None

    def sort_key(path):
        try:
            stem = os.path.splitext(os.path.basename(path))[0]
            version = int(stem.rsplit("_v", 1)[-1])
        except (ValueError, IndexError):
            version = 0
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0.0
        return (version, mtime, os.path.basename(path))

    checkpoint_path = max(candidates, key=sort_key)
    try:
        checkpoint = _safe_torch_load(checkpoint_path, map_location=DEVICE)
        metadata = checkpoint.get("metadata", {}) if isinstance(checkpoint, dict) else {}
        if "model_state_dict" not in checkpoint:
            return None
        conv_model, metadata = _build_convlstm_from_checkpoint(checkpoint, metadata)
        sequence_context = _prepare_sequences_for_checkpoint(
            data_raw=data_raw,
            data_mean=data_mean,
            data_std=data_std,
            metadata=metadata,
        )
        sequence_context.update(
            {
                "model": conv_model,
                "model_type": "convlstm",
                "checkpoint_path": checkpoint_path,
            }
        )
        LOGGER.info("Loaded secondary ConvLSTM temperature context from %s", checkpoint_path)
        return sequence_context
    except Exception as exc:
        LOGGER.warning("Unable to load secondary ConvLSTM temperature context: %s", exc)
        return None


def _denormalize_temperature_grid(temp_norm, local_temp_mean, local_temp_std):
    temp_denorm = temp_norm * (local_temp_std + EPSILON) + local_temp_mean
    if np.nanmean(temp_denorm) > 200:
        temp_celsius = temp_denorm - 273.15
    else:
        temp_celsius = temp_denorm
    return np.clip(temp_celsius, -10.0, 60.0)


def _extract_observed_temperature_grid(sample_idx=-1, context=None):
    local_context = context or {
        "X_test": X_test,
        "temp_mean_scalar": temp_mean_scalar,
        "temp_std_scalar": temp_std_scalar,
    }
    x_test_local = local_context.get("X_test")
    if x_test_local is None or len(x_test_local) == 0:
        raise RuntimeError("Observed temperature data is not available")
    x_seq = np.array(x_test_local[sample_idx], copy=True)
    temp_norm = x_seq[-1, 1, :, :]
    return _denormalize_temperature_grid(
        temp_norm,
        float(local_context["temp_mean_scalar"]),
        float(local_context["temp_std_scalar"]),
    )


def _predict_temperature_from_context(context, sample_idx=-1, days=7):
    x_seq = np.array(context["X_test"][sample_idx], copy=True)
    x_tensor = torch.from_numpy(x_seq).unsqueeze(0).float().to(DEVICE)
    outputs = []
    with torch.no_grad():
        output = context["model"](x_tensor, future_seq=days)
    for day_idx in range(days):
        temp_norm = output[0, day_idx, 1, :, :].cpu().numpy()
        outputs.append(
            _denormalize_temperature_grid(
                temp_norm,
                float(context["temp_mean_scalar"]),
                float(context["temp_std_scalar"]),
            )
        )
    return outputs


def _temperature_forecast_context():
    if runtime_model_type == "convlstm":
        return {
            "model": model,
            "X_test": X_test,
            "temp_mean_scalar": temp_mean_scalar,
            "temp_std_scalar": temp_std_scalar,
            "model_type": runtime_model_type,
            "checkpoint_path": None,
        }
    return temperature_forecast_context


def get_prediction_data(sample_idx=-1):
    """Run model inference and return (display temperature grid, probability)."""
    if not resources_ready():
        raise RuntimeError("Model or data resources are not ready")
    cache_key = (sample_idx, 1)
    if cache_key in _prediction_cache:
        entry = _prediction_cache[cache_key][0]
        return entry["temperature_grid"], entry["probability"]
    entry = get_prediction_sequence(sample_idx=sample_idx, days=1)[0]
    return entry["temperature_grid"], entry["probability"]


def get_prediction_sequence(sample_idx=-1, days=7):
    """Return inference payloads with observed and forecasted temperature separated."""
    if not resources_ready():
        raise RuntimeError("Model or data resources are not ready")
    cache_key = (sample_idx, days)
    with _resource_lock:
        if model is None:
            raise RuntimeError("Model not loaded. Call /api/health to check status.")
        if X_test is None:
            raise RuntimeError("Test data not loaded. Resources not initialized.")
        if temp_mean_scalar is None:
            raise RuntimeError("Temperature normalization stats not loaded.")
        if temp_std_scalar is None:
            raise RuntimeError("Temperature normalization stats not loaded.")
        if cache_key in _prediction_cache:
            return _prediction_cache[cache_key]

    outputs = []

    observed_grid = _extract_observed_temperature_grid(sample_idx=sample_idx)
    forecast_context = _temperature_forecast_context()
    forecast_grids = None
    if forecast_context is not None:
        forecast_grids = _predict_temperature_from_context(
            forecast_context,
            sample_idx=sample_idx,
            days=days,
        )

    if runtime_model_type == "convlstm":
        assert forecast_grids is not None
        for temp_celsius in forecast_grids:
            max_temp = float(np.nanmax(temp_celsius))
            prob = min(1.0, max(0.0, (max_temp - 30.0) / 15.0))
            outputs.append(
                {
                    "temperature_grid": temp_celsius,
                    "forecast_temperature_grid": temp_celsius,
                    "observed_temperature_grid": observed_grid,
                    "probability": prob,
                    "temperature_source": "forecast_model",
                    "forecast_available": True,
                }
            )
    else:
        x_seq = np.array(X_test[sample_idx], copy=True)
        expected_flat_dim = int(getattr(model, "n_features_in_", 0) or 0)
        for day_idx in range(days):
            x_flat = x_seq.reshape(1, -1)
            if expected_flat_dim and x_flat.shape[1] != expected_flat_dim:
                raise RuntimeError(
                    "Feature shape mismatch after API resource alignment, "
                    f"expected: {expected_flat_dim}, got: {x_flat.shape[1]}"
                )
            prob = float(model.predict_proba(x_flat)[0, 1])
            pred_step = np.array(x_seq[-1], copy=True)
            fallback_observed = _denormalize_temperature_grid(
                pred_step[1],
                float(temp_mean_scalar),
                float(temp_std_scalar),
            )
            forecast_grid = (
                forecast_grids[day_idx] if forecast_grids is not None else None
            )
            display_grid = forecast_grid if forecast_grid is not None else fallback_observed
            outputs.append(
                {
                    "temperature_grid": display_grid,
                    "forecast_temperature_grid": forecast_grid,
                    "observed_temperature_grid": fallback_observed,
                    "probability": prob,
                    "temperature_source": (
                        "forecast_model" if forecast_grid is not None else "observed_persistence"
                    ),
                    "forecast_available": forecast_grid is not None,
                }
            )

            # Autoregressive update for classifier backend
            x_seq = np.concatenate([x_seq[1:], pred_step[None, :, :, :]], axis=0)

    with _resource_lock:
        _prediction_cache[cache_key] = outputs
    return outputs


def _nearest_grid_indices(target_lat, target_lon):
    assert lats is not None
    assert lons is not None
    lat_idx = int(np.abs(np.asarray(lats, dtype=np.float32) - float(target_lat)).argmin())
    lon_idx = int(np.abs(np.asarray(lons, dtype=np.float32) - float(target_lon)).argmin())
    return lat_idx, lon_idx


def _sample_region_temperature(temp_grid, target_lat, target_lon):
    assert lats is not None
    assert lons is not None

    lat_values = np.asarray(lats, dtype=np.float32)
    lon_values = np.asarray(lons, dtype=np.float32)
    target_lat = float(target_lat)
    target_lon = float(target_lon)

    lat_mask = np.abs(lat_values - target_lat) <= REGION_SAMPLING_RADIUS_DEGREES
    lon_mask = np.abs(lon_values - target_lon) <= REGION_SAMPLING_RADIUS_DEGREES

    lat_indices = np.where(lat_mask)[0]
    lon_indices = np.where(lon_mask)[0]

    fallback_lat_idx, fallback_lon_idx = _nearest_grid_indices(target_lat, target_lon)

    if lat_indices.size == 0 or lon_indices.size == 0:
        sampled_temperature = float(temp_grid[fallback_lat_idx, fallback_lon_idx])
        return {
            "temperature": sampled_temperature,
            "grid_lat": float(lats[fallback_lat_idx]),
            "grid_lng": float(lons[fallback_lon_idx]),
            "sample_count": 1,
            "sampling_method": "nearest_grid_fallback",
            "sampling_radius_degrees": REGION_SAMPLING_RADIUS_DEGREES,
            "sampling_sigma_km": REGION_SAMPLING_SIGMA_KM,
        }

    local_grid = np.asarray(temp_grid[np.ix_(lat_indices, lon_indices)], dtype=np.float32)
    local_lats = lat_values[lat_indices]
    local_lons = lon_values[lon_indices]
    mesh_lats, mesh_lons = np.meshgrid(local_lats, local_lons, indexing="ij")

    lat_distance_km = (mesh_lats - target_lat) * 111.0
    lon_distance_km = (mesh_lons - target_lon) * 111.0 * max(
        np.cos(np.radians(target_lat)), 0.2
    )
    distance_km = np.sqrt(lat_distance_km**2 + lon_distance_km**2)

    finite_mask = np.isfinite(local_grid)
    if not np.any(finite_mask):
        sampled_temperature = float(temp_grid[fallback_lat_idx, fallback_lon_idx])
        return {
            "temperature": sampled_temperature,
            "grid_lat": float(lats[fallback_lat_idx]),
            "grid_lng": float(lons[fallback_lon_idx]),
            "sample_count": 1,
            "sampling_method": "nearest_grid_fallback",
            "sampling_radius_degrees": REGION_SAMPLING_RADIUS_DEGREES,
            "sampling_sigma_km": REGION_SAMPLING_SIGMA_KM,
        }

    weights = np.exp(-0.5 * (distance_km / REGION_SAMPLING_SIGMA_KM) ** 2)
    weights = np.where(finite_mask, weights, 0.0)
    weight_sum = float(np.sum(weights))

    if weight_sum <= 0:
        sampled_temperature = float(temp_grid[fallback_lat_idx, fallback_lon_idx])
        return {
            "temperature": sampled_temperature,
            "grid_lat": float(lats[fallback_lat_idx]),
            "grid_lng": float(lons[fallback_lon_idx]),
            "sample_count": 1,
            "sampling_method": "nearest_grid_fallback",
            "sampling_radius_degrees": REGION_SAMPLING_RADIUS_DEGREES,
            "sampling_sigma_km": REGION_SAMPLING_SIGMA_KM,
        }

    sampled_temperature = float(np.sum(local_grid * weights) / weight_sum)
    nearest_local_index = int(np.argmin(np.where(finite_mask, distance_km, np.inf)))
    nearest_row, nearest_col = np.unravel_index(nearest_local_index, distance_km.shape)

    return {
        "temperature": sampled_temperature,
        "grid_lat": float(mesh_lats[nearest_row, nearest_col]),
        "grid_lng": float(mesh_lons[nearest_row, nearest_col]),
        "sample_count": int(np.count_nonzero(finite_mask)),
        "sampling_method": "distance_weighted_area_average",
        "sampling_radius_degrees": REGION_SAMPLING_RADIUS_DEGREES,
        "sampling_sigma_km": REGION_SAMPLING_SIGMA_KM,
    }


def _region_payload_from_grid(temp_grid, model_probability=None):
    regions = []

    for region in WEB_REGIONS:
        sampled_region = _sample_region_temperature(
            temp_grid, region["lat"], region["lng"]
        )
        temperature = float(sampled_region["temperature"])
        risk_payload = _build_risk_payload_from_temperature(temperature)
        probability = risk_payload["probability"]

        if model_probability is not None:
            probability = max(
                probability,
                float(np.clip(float(model_probability), 0.0, 1.0))
                if risk_payload["risk_code"] != "LOW"
                else probability,
            )

        regions.append(
            {
                "name": region["name"],
                "zone": region["zone"],
                "lat": float(region["lat"]),
                "lng": float(region["lng"]),
                "grid_lat": sampled_region["grid_lat"],
                "grid_lng": sampled_region["grid_lng"],
                "temperature": round(temperature, 2),
                "temperature_c": round(temperature, 2),
                "risk_level": risk_payload["risk_code"],
                "risk_code": risk_payload["risk_code"],
                "risk_label": risk_payload["risk_label"],
                "risk_index": risk_payload["risk_index"],
                "probability": round(float(probability), 4),
                "sample_count": sampled_region["sample_count"],
                "sampling_method": sampled_region["sampling_method"],
                "sampling_radius_degrees": sampled_region["sampling_radius_degrees"],
                "sampling_sigma_km": sampled_region["sampling_sigma_km"],
                "risk": {
                    "risk_code": risk_payload["risk_code"],
                    "risk_label": risk_payload["risk_label"],
                    "risk_index": risk_payload["risk_index"],
                },
            }
        )

    return regions


@app.route("/api/predict", methods=["GET"])
def predict_summary():
    """
    Get heatwave prediction summary.
    ---
    tags:
      - Prediction
    parameters:
      - name: days
        in: query
        type: integer
        default: 7
        description: Number of forecast days
    responses:
      200:
        description: Prediction result
        schema:
          type: object
          properties:
            current_temp:
              type: number
            risk_level:
              type: string
              enum: [LOW, MODERATE, HIGH, CRITICAL]
            probability:
              type: number
      503:
        description: Resources not loaded
    """
    # Model selection: use ?model=convlstm|xgboost|rf|auto
    # - auto: tries XGBoost first, falls back to ConvLSTM/RF
    # - convlstm: forces ConvLSTM spatial model
    # - xgboost: forces daily XGBoost tabular model
    # - rf: forces RandomForest model
    # Response always includes: model_used, model_version, confidence_interval
    requested_model = request.args.get("model", "auto").lower()
    _model_version = get_latest_model(MODELS_DIR) or "unknown"

    # Try XGBoost first (unless caller forced convlstm or rf)
    try:
        from api_daily_predict import daily_model_ready, load_daily_model, predict_from_daily_weather, _daily_threshold

        if requested_model not in ("convlstm", "rf") and (requested_model == "xgboost" or daily_model_ready() or load_daily_model()):
            if X_test is not None and len(X_test) > 0:
                observed_grid = _extract_observed_temperature_grid()
                temp_max = float(np.nanmax(observed_grid))
                temp_mean = float(np.nanmean(observed_grid))
                temp_min = float(np.nanmin(observed_grid))
            else:
                temp_max = 35.0
                temp_mean = 30.0
                temp_min = 25.0
            
            # Get XGBoost prediction
            weather_data = {
                'temp_mean': temp_mean,
                'temp_max': temp_max,
                'temp_min': temp_min,
                'temp_std': 4.0,
                'humidity': 70.0,
            }
            result = predict_from_daily_weather(weather_data)
            
            # Build response
            actual_date = get_sample_date(sample_idx=-1, day_offset=0) if _test_time_base else datetime.datetime.now()
            
            risk_map = {
                'LOW': {'code': 'LOW', 'label': 'Low Risk', 'index': 0, 'probability': result['heatwave_probability']},
                'MEDIUM': {'code': 'MEDIUM', 'label': 'Medium Risk', 'index': 1, 'probability': result['heatwave_probability']},
                'HIGH': {'code': 'HIGH', 'label': 'High Risk', 'index': 2, 'probability': result['heatwave_probability']},
                'CRITICAL': {'code': 'CRITICAL', 'label': 'Critical Risk', 'index': 3, 'probability': result['heatwave_probability']},
            }
            
            risk_info = risk_map.get(result['risk_level'], risk_map['LOW'])

            if _drift_available:
                _alert = _record_drift(float(temp_max))
                if _alert:
                    LOGGER.warning("Drift alert: %s", _alert.get("message"))

            _request_stats["predict_calls"] += 1
            return jsonify({
                "status": "ok",
                "date": actual_date.strftime("%Y-%m-%d") if hasattr(actual_date, 'strftime') else str(actual_date),
                "data_date": actual_date.strftime("%Y-%m-%d") if hasattr(actual_date, 'strftime') else str(actual_date),
                "generated_at": datetime.datetime.now().isoformat(),
                "risk_level": risk_info['code'],
                "risk_code": risk_info['code'],
                "risk_label": risk_info['label'],
                "risk_index": risk_info['index'],
                "probability": result['heatwave_probability'],
                "advice": f"Observed max temperature {temp_max:.1f}C with {result['heatwave_probability']*100:.1f}% heatwave probability. {risk_info['label']}.",
                "model_type": "xgboost_daily",
                "model_used": "xgboost",
                "model_version": _model_version,
                "temperature_source": "observed_latest_input",
                "forecast_available": False,
                "confidence_interval": {
                    "lower": round(result['heatwave_probability'] - round(min(result['heatwave_probability'], 1 - result['heatwave_probability']) * 0.3, 3), 3),
                    "upper": round(result['heatwave_probability'] + round(min(result['heatwave_probability'], 1 - result['heatwave_probability']) * 0.3, 3), 3),
                },
                "weather": {
                    "T2M_MAX": temp_max,
                    "T2M": temp_mean,
                    "T2M_MIN": temp_min,
                    "RH2M": None,
                    "WS10M": None,
                },
                "anomaly": {
                    "is_anomaly": result['heatwave_predicted'],
                    "severity": result['risk_level'].lower(),
                    "n_triggers": 0,
                    "triggers": [],
                },
                "bbox": bbox,
                "regions": [],
            })
    except Exception as e:
        LOGGER.warning(f"XGBoost prediction failed, falling back to ConvLSTM: {e}")
    
    # Fallback to ConvLSTM/RF model
    if not resources_ready():
        return jsonify({"error": "No model loaded"}), 500

    try:
        prediction = get_prediction_sequence(days=1)[0]
        display_grid = prediction["temperature_grid"]
        observed_grid = prediction["observed_temperature_grid"]
        forecast_grid = prediction["forecast_temperature_grid"]
        prob = prediction["probability"]
        max_temp = float(np.max(display_grid))
        mean_temp = float(np.mean(display_grid))
        min_temp = float(np.min(display_grid))
        observed_max = float(np.max(observed_grid))
        observed_mean = float(np.mean(observed_grid))
        observed_min = float(np.min(observed_grid))

        # Get actual date from data
        actual_date = get_sample_date(sample_idx=-1, day_offset=0)
        
        risk_payload = _build_risk_payload_from_temperature(max_temp, model_probability=prob)
        region_payload = _region_payload_from_grid(display_grid, model_probability=prob)
        risk_level = risk_payload["risk_code"]
        probability = risk_payload["probability"]
        forecast_available = bool(prediction["forecast_available"])
        temperature_source = prediction["temperature_source"]

        if _drift_available:
            _alert = _record_drift(float(max_temp))
            if _alert:
                LOGGER.warning("Drift alert: %s", _alert.get("message"))

        _request_stats["predict_calls"] += 1
        return jsonify(
            {
                "status": "ok",
                "date": actual_date.strftime("%Y-%m-%d"),
                "data_date": actual_date.strftime("%Y-%m-%d"),
                "generated_at": datetime.datetime.now().isoformat(),
                "risk_level": risk_level,
                "risk_label": risk_payload["risk_label"],
                "risk_index": risk_payload["risk_index"],
                "risk": {
                    "risk_code": risk_payload["risk_code"],
                    "risk_label": risk_payload["risk_label"],
                    "risk_index": risk_payload["risk_index"],
                },
                "probability": probability,
                "advice": (
                    f"Forecast max temp reaching {max_temp:.1f} C. {risk_level} precautions advised."
                    if forecast_available
                    else f"Heatwave probability is {probability*100:.1f}% based on the latest observed max temperature {observed_max:.1f} C."
                ),
                "model_type": runtime_model_type,
                "model_used": runtime_model_type,
                "model_version": _model_version,
                "confidence_interval": {
                    "lower": round(probability - round(min(probability, 1 - probability) * 0.3, 3), 3),
                    "upper": round(probability + round(min(probability, 1 - probability) * 0.3, 3), 3),
                },
                "temperature_source": temperature_source,
                "forecast_available": forecast_available,
                "weather": {
                    "T2M_MAX": max_temp,
                    "T2M": mean_temp,
                    "T2M_MIN": min_temp,
                    "RH2M": None,
                    "WS10M": None,
                },
                "observed_weather": {
                    "T2M_MAX": observed_max,
                    "T2M": observed_mean,
                    "T2M_MIN": observed_min,
                },
                "forecast_weather": (
                    {
                        "T2M_MAX": max_temp,
                        "T2M": mean_temp,
                        "T2M_MIN": min_temp,
                    }
                    if forecast_grid is not None
                    else None
                ),
                "anomaly": {
                    "is_anomaly": risk_level in ["HIGH", "CRITICAL"],
                    "severity": risk_level.lower(),
                    "n_triggers": 0,
                    "triggers": [],
                },
                "bbox": bbox,
                "regions": region_payload,
            }
        )
    except Exception as e:
        LOGGER.exception("Predict summary failed: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/forecast", methods=["GET"])
def forecast_summary():
    """
    Get 7-day forecast.
    ---
    tags:
      - Prediction
    responses:
      200:
        description: 7-day forecast
        schema:
          type: object
          properties:
            forecast:
              type: array
              items:
                type: object
                properties:
                  date:
                    type: string
                  temp:
                    type: number
                  risk:
                    type: string
    """
    if not resources_ready():
        return jsonify({"error": "Model not loaded"}), 500

    try:
        predictions = get_prediction_sequence(days=7)

        days = []
        region_forecasts = {region["name"]: [] for region in WEB_REGIONS}
        
        # Use actual date from data for base
        base_date = get_sample_date(sample_idx=-1, day_offset=0)
        generated_at = datetime.datetime.now()

        for i in range(7):
            forecast_date = base_date + datetime.timedelta(days=i + 1)
            date_str = forecast_date.strftime("%Y-%m-%d")
            prediction = predictions[i]
            temp_grid = prediction["temperature_grid"]
            observed_grid = prediction["observed_temperature_grid"]
            forecast_grid = prediction["forecast_temperature_grid"]
            prob = prediction["probability"]
            mean_temp = float(np.mean(temp_grid))
            min_temp = float(np.min(temp_grid))
            t_max = float(np.max(temp_grid))
            observed_mean = float(np.mean(observed_grid))
            observed_min = float(np.min(observed_grid))
            observed_max = float(np.max(observed_grid))
            risk_payload = _build_risk_payload_from_temperature(t_max, model_probability=prob)
            risk_level = risk_payload["risk_code"]
            probability = risk_payload["probability"]

            days.append(
                {
                    "day": i + 1,
                    "date": date_str,
                    "day_name": forecast_date.strftime("%a"),
                    "probability": probability,
                    "risk_level": risk_level,
                    "risk_code": risk_payload["risk_code"],
                    "risk_label": risk_payload["risk_label"],
                    "risk_index": risk_payload["risk_index"],
                    "risk": {
                        "risk_code": risk_payload["risk_code"],
                        "risk_label": risk_payload["risk_label"],
                        "risk_index": risk_payload["risk_index"],
                    },
                    "advice": (
                        f"Generated from {runtime_model_type} temperature forecast"
                        if prediction["forecast_available"]
                        else "No dedicated temperature forecast model is available; temperatures reflect the latest observed persistence baseline."
                    ),
                    "temperature_source": prediction["temperature_source"],
                    "forecast_available": bool(prediction["forecast_available"]),
                    "weather": {
                        "T2M": mean_temp,
                        "T2M_MAX": t_max,
                        "T2M_MIN": min_temp,
                        "PRECTOTCORR": None,
                        "WS10M": None,
                        "RH2M": None,
                        "NDVI": None,
                    },
                    "observed_weather": {
                        "T2M": observed_mean,
                        "T2M_MAX": observed_max,
                        "T2M_MIN": observed_min,
                    },
                    "forecast_weather": (
                        {
                            "T2M": mean_temp,
                            "T2M_MAX": t_max,
                            "T2M_MIN": min_temp,
                        }
                        if forecast_grid is not None
                        else None
                    ),
                }
            )

            day_region_payload = _region_payload_from_grid(temp_grid, model_probability=prob)
            for region_entry in day_region_payload:
                region_forecasts[region_entry["name"]].append(
                    {
                        "day": i + 1,
                        "date": date_str,
                        "day_name": forecast_date.strftime("%a"),
                        "probability": region_entry["probability"],
                        "risk_level": region_entry["risk_code"],
                        "risk_code": region_entry["risk_code"],
                        "risk_label": region_entry["risk_label"],
                        "risk_index": region_entry["risk_index"],
                        "risk": {
                            "risk_code": region_entry["risk_code"],
                            "risk_label": region_entry["risk_label"],
                            "risk_index": region_entry["risk_index"],
                        },
                        "weather": {
                            "T2M": region_entry["temperature"],
                            "T2M_MAX": region_entry["temperature"],
                            "T2M_MIN": region_entry["temperature"],
                        },
                        "temperature_source": prediction["temperature_source"],
                        "temperature": region_entry["temperature"],
                        "temperature_c": region_entry["temperature"],
                    }
                )

        region_forecast_items = []
        for region in WEB_REGIONS:
            region_days = region_forecasts.get(region["name"], [])
            region_forecast_items.append(
                {
                    "name": region["name"],
                    "zone": region["zone"],
                    "lat": float(region["lat"]),
                    "lng": float(region["lng"]),
                    "forecast": region_days,
                }
            )

        return jsonify(
            {
                "status": "ok",
                "model_type": runtime_model_type,
                "days": 7,
                "base_date": base_date.strftime("%Y-%m-%d"),
                "generated_at": generated_at.isoformat(),
                "forecast_available": any(
                    bool(prediction["forecast_available"]) for prediction in predictions
                ),
                "forecasts": days,
                "region_forecasts": region_forecast_items,
            }
        )

    except Exception as e:
        LOGGER.exception("Forecast summary failed: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/map", methods=["GET"])
def predict_map():
    """
    Get heatwave risk map as GeoJSON.
    ---
    tags:
      - Map
    responses:
      200:
        description: GeoJSON FeatureCollection of risk zones
        schema:
          type: object
          properties:
            type:
              type: string
              example: FeatureCollection
            features:
              type: array
    """
    if not resources_ready():
        return jsonify({"error": "Model not loaded"}), 500
    assert lats is not None
    assert lons is not None

    try:
        temp_grid, _prob = get_prediction_data()

        features = []
        rows, cols = temp_grid.shape

        if len(lats) > 1:
            dlat = abs(lats[1] - lats[0])
        else:
            dlat = 0.25

        if len(lons) > 1:
            dlon = abs(lons[1] - lons[0])
        else:
            dlon = 0.25

        risk_grid = np.zeros_like(temp_grid, dtype=np.int8)
        risk_grid[temp_grid >= 35] = 1
        risk_grid[temp_grid >= 38] = 2
        risk_grid[temp_grid >= 41] = 3

        for r in range(rows):
            for c in range(cols):
                val = float(temp_grid[r, c])

                # Filter: skip cells below minimum risk threshold to reduce payload size
                if val < 28.0:
                    continue

                lat = float(lats[r])
                lon = float(lons[c])

                risk = int(risk_grid[r, c])
                risk_payload = _build_risk_payload_from_index(risk)

                min_lon, max_lon = lon - dlon / 2, lon + dlon / 2
                min_lat, max_lat = lat - dlat / 2, lat + dlat / 2

                poly_coords = [
                    [min_lon, min_lat],
                    [max_lon, min_lat],
                    [max_lon, max_lat],
                    [min_lon, max_lat],
                    [min_lon, min_lat],
                ]

                feature = {
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [poly_coords]},
                    "properties": {
                        "temperature": round(val, 2),
                        "temperature_c": round(val, 2),
                        "risk_level": risk,
                        "risk_code": risk_payload["risk_code"],
                        "risk_label": risk_payload["risk_label"],
                        "risk_index": risk_payload["risk_index"],
                        "risk": {
                            "risk_code": risk_payload["risk_code"],
                            "risk_label": risk_payload["risk_label"],
                            "risk_index": risk_payload["risk_index"],
                        },
                    },
                }
                features.append(feature)

        response = jsonify(
            {
                "type": "FeatureCollection",
                "features": features,
                "risk_schema": {
                    "legacy_fields": {
                        "risk_level": "numeric index (0..3)",
                    },
                    "canonical_fields": {
                        "feature.properties.temperature_c": "temperature in Celsius",
                        "feature.properties.risk_code": "LOW|MEDIUM|HIGH|CRITICAL",
                        "feature.properties.risk_label": "LOW|MEDIUM|HIGH|CRITICAL",
                        "feature.properties.risk_index": "0..3",
                        "feature.properties.risk": {
                            "risk_code": "LOW|MEDIUM|HIGH|CRITICAL",
                            "risk_label": "LOW|MEDIUM|HIGH|CRITICAL",
                            "risk_index": "0..3",
                        },
                    },
                    "levels": _risk_legend(),
                },
            }
        )
        response.headers['Cache-Control'] = 'public, max-age=300'
        return response

    except Exception as e:
        LOGGER.exception("Predict map failed: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def index():
    return redirect("/trainer")


@app.route("/trainer", methods=["GET"])
def trainer_console():
    return render_template(
        "trainer.html",
        defaults=json.dumps(get_default_training_config()),
    )


@app.route("/dashboard", methods=["GET"])
def dashboard():
    return render_template("dashboard.html")

@app.route("/api/training/preflight", methods=["GET"])
def training_preflight():
    return jsonify(training_preflight_summary())


@app.route("/api/training/status", methods=["GET"])
def training_status():
    state = snapshot_training_state()
    state["gpu"] = detect_gpu_capability()
    return jsonify(state)


@app.route("/api/training/history", methods=["GET"])
def training_history_api():
    return jsonify({"items": snapshot_training_history()})


@app.route("/api/training/start", methods=["POST"])
@require_api_key
def training_start():
    payload = request.get_json(silent=True) or {}
    config, errors = sanitize_training_config(payload)
    if errors:
        return (
            jsonify(
                {
                    "error": "Invalid training configuration",
                    "errors": errors,
                    "config": config,
                }
            ),
            400,
        )

    now = datetime.datetime.now().isoformat()
    with training_lock:
        current_status = training_state.get("status")
        if current_status in {"running", "starting"}:
            return jsonify({"error": "Training is already running"}), 409

        training_state.update(
            {
                "status": "starting",
                "started_at": now,
                "finished_at": None,
                "message": "Training request accepted. Starting worker...",
                "config": config,
                "metrics": None,
                "result": None,
                "error": None,
            }
        )

    append_training_history(
        {
            "status": "starting",
            "finished_at": now,
            "config": config,
            "metrics": None,
            "result": None,
            "error": None,
        }
    )

    worker = threading.Thread(target=run_training_job, args=(config,), daemon=True)
    worker.start()
    return jsonify(
        {
            "status": "started",
            "message": "Training worker started",
            "config": config,
            "preflight": training_preflight_summary(),
        }
    )


@app.route("/api/health", methods=["GET"])
def health_check():
    """
    Get API health status.
    ---
    tags:
      - Health
    responses:
      200:
        description: Health status
        schema:
          type: object
          properties:
            status:
              type: string
              example: ok
            resources_loaded:
              type: boolean
            model_type:
              type: string
    """
    if _freshness_available:
        freshness = get_freshness_summary()
        data_stale = is_data_stale()
    else:
        freshness = {}
        data_stale = None
    return jsonify({
        "status": "ok",
        "model_loaded": resources_ready(),
        "data_freshness": freshness,
        "data_stale": data_stale,
    })


@app.route("/api/health/extended", methods=["GET"])
def health_extended():
    """
    Extended health check with observability metrics.
    ---
    tags:
      - Health
    responses:
      200:
        description: Extended health metrics
    """
    uptime_seconds = int(time.time() - _startup_time)
    uptime_str = f"{uptime_seconds // 3600}h {(uptime_seconds % 3600) // 60}m {uptime_seconds % 60}s"

    latest_ckpt = get_latest_model(MODELS_DIR)

    if _freshness_available:
        freshness = get_freshness_summary()
        stale = is_data_stale()
    else:
        freshness = {}
        stale = None

    with training_lock:
        t_state = dict(training_state)

    result = {
        "status": "ok" if resources_ready() else "degraded",
        "uptime": uptime_str,
        "uptime_seconds": uptime_seconds,
        "resources_loaded": resources_ready(),
        "model": {
            "latest_checkpoint": latest_ckpt,
            "type": runtime_model_type,
        },
        "data": {
            "freshness": freshness,
            "stale": stale,
        },
        "requests": {
            "total": _request_stats["total"],
            "errors": _request_stats["errors"],
            "predict_calls": _request_stats["predict_calls"],
            "last_request_time": _request_stats["last_request_time"],
        },
        "training": {
            "active": t_state.get("active", t_state.get("status") == "running"),
            "progress": t_state.get("progress", 0),
            "last_trained": t_state.get("finished_at") or t_state.get("end_time"),
        },
    }
    if _drift_available:
        result["drift"] = _get_drift_status()
    return jsonify(result)


if __name__ == "__main__":
    # Try to load XGBoost daily model first (preferred)
    xgboost_loaded = False
    try:
        from api_daily_predict import load_daily_model, daily_model_ready
        if load_daily_model():
            print("XGBoost daily model loaded successfully!")
            xgboost_loaded = True
    except Exception as e:
        LOGGER.warning(f"Could not load XGBoost model: {e}")
    
    # Then load ConvLSTM/RF model (fallback)
    if not load_resources():
        LOGGER.warning(
            "ConvLSTM/RF model not loaded. Using XGBoost only."
        )
    
    # Initialize daily prediction routes
    try:
        from api_daily_predict import init_daily_routes
        init_daily_routes(app)
        print("Daily prediction API initialized.")
    except Exception as e:
        LOGGER.warning(f"Could not initialize daily prediction API: {e}")
    
    print("\n" + "=" * 50)
    print("AGNI HEATWAVE FORECAST API")
    print("=" * 50)
    print(f"XGBoost Daily Model: {'LOADED' if xgboost_loaded else 'NOT LOADED'}")
    print(f"ConvLSTM/RF Model: {'LOADED' if resources_ready() else 'NOT LOADED'}")
    print("\nAPI Endpoints:")
    print("  POST /api/daily/predict - XGBoost prediction (recommended)")
    print("  GET  /api/predict       - Legacy prediction")
    print("  GET  /api/health        - Health check")
    print("=" * 50 + "\n")
    
    print("Starting Flask API on port 5000...")
    app.run(host="0.0.0.0", port=5000, debug=False)

