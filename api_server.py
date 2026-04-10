from flask import Flask, jsonify, render_template
from flask_cors import CORS
from flask import redirect, request
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
from Train_Ai import train as run_training
from src.data.loader import load_era5_data, create_sequences, normalize_data, clean_data
from src.models.convlstm import HeatwaveConvLSTM

app = Flask(__name__)
CORS(app)  # Enable CORS

DATA_DIR = "era5_data"
MODELS_DIR = "models"
SEQ_LEN = 7
FUTURE_SEQ = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGGER = logging.getLogger(__name__)
EPSILON = 1e-6


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
training_lock = threading.Lock()
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
    rf_files = glob.glob(os.path.join(model_dir, "heatwave_model_checkpoint_v*.pth"))
    convlstm_files = glob.glob(os.path.join(model_dir, "heatwave_convlstm_v*.pth"))

    def get_version(f):
        try:
            return int(f.split("_v")[-1].split(".")[0])
        except (ValueError, IndexError):
            return 0

    # Prefer ConvLSTM (real temperature regression) if available
    if convlstm_files:
        return max(convlstm_files, key=get_version)
    if rf_files:
        return max(rf_files, key=get_version)
    return None


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


def get_risk_and_probability(max_temp, model_probability=None):
    risk_level = "LOW"
    if max_temp >= 35:
        risk_level = "MEDIUM"
    if max_temp >= 38:
        risk_level = "HIGH"
    if max_temp >= 41:
        risk_level = "CRITICAL"

    if model_probability is not None:
        probability = float(model_probability)
    else:
        fallback_lookup = {
            "LOW": 0.1,
            "MEDIUM": 0.5,
            "HIGH": 0.8,
            "CRITICAL": 0.95,
        }
        probability = fallback_lookup[risk_level]

    return risk_level, probability


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
    if labeling_method not in {"heat_index", "temperature"}:
        errors.append("labeling_method must be 'heat_index' or 'temperature'")
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
    model_files = glob.glob(os.path.join(MODELS_DIR, "heatwave_model_checkpoint_v*.pth"))

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


def load_resources(checkpoint_path=None, skip_data_reload=False):
    global model, X_test, Y_test, lats, lons, mean, std
    global temp_mean_scalar, temp_std_scalar, bbox, _prediction_cache
    global runtime_seq_len, runtime_future_seq, runtime_model_type

    checkpoint_path = checkpoint_path or get_latest_model(MODELS_DIR)
    if not checkpoint_path:
        LOGGER.error("No model found in %s", MODELS_DIR)
        return False

    print(f"Loading model from {checkpoint_path}...")
    checkpoint = _safe_torch_load(checkpoint_path, map_location=DEVICE)
    metadata = checkpoint.get("metadata", {}) if isinstance(checkpoint, dict) else {}

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
            mean = ckpt_mean_arr
            std = ckpt_std_arr
        else:
            LOGGER.warning(
                "Checkpoint normalization stats channel mismatch (ckpt=%s, data=%s). "
                "Using data-derived stats.",
                ckpt_mean_arr.shape,
                data_raw.shape,
            )
            mean = data_mean
            std = data_std
    else:
        mean = data_mean
        std = data_std
    data_raw, _clip_bounds = clean_data(data_raw)
    data = normalize_data(data_raw, mean, std)

    # Create sequences
    # We take the last part for testing/demo
    X, Y = create_sequences(data, seq_len=runtime_seq_len, pred_len=runtime_future_seq)
    if len(X) == 0:
        LOGGER.error("No sequences available from loaded data")
        return False

    split_idx = int(len(X) * 0.85)
    X_test = X[split_idx:]
    Y_test = Y[split_idx:]

    temp_mean_scalar = float(mean[0, 1, 0, 0])
    temp_std_scalar = float(std[0, 1, 0, 0])
    bbox = {
        "north": float(np.max(lats)),
        "south": float(np.min(lats)),
        "east": float(np.max(lons)),
        "west": float(np.min(lons)),
    }
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


def get_prediction_data(sample_idx=-1):
    """Run model inference and return (denormalized temperature grid, probability)."""
    if not resources_ready():
        raise RuntimeError("Model or data resources are not ready")
    cache_key = (sample_idx, 1)
    if cache_key in _prediction_cache:
        return _prediction_cache[cache_key][0]
    return get_prediction_sequence(sample_idx=sample_idx, days=1)[0]


def get_prediction_sequence(sample_idx=-1, days=7):
    """Run autoregressive inference and return (temperature grid, probability) by day."""
    if not resources_ready():
        raise RuntimeError("Model or data resources are not ready")
    assert model is not None
    assert X_test is not None
    assert temp_mean_scalar is not None
    assert temp_std_scalar is not None

    cache_key = (sample_idx, days)
    if cache_key in _prediction_cache:
        return _prediction_cache[cache_key]

    x_seq = np.array(X_test[sample_idx], copy=True)
    outputs = []

    if runtime_model_type == "convlstm":
        # ConvLSTM path: single forward pass for all days
        x_tensor = torch.from_numpy(x_seq).unsqueeze(0).float().to(DEVICE)
        with torch.no_grad():
            output = model(x_tensor, future_seq=days)
        # output shape: (1, days, channels, H, W)
        for d in range(days):
            temp_norm = output[0, d, 1, :, :].cpu().numpy()
            temp_celsius = temp_norm * (temp_std_scalar + EPSILON) + temp_mean_scalar
            if np.nanmean(temp_celsius) > 200:
                temp_celsius = temp_celsius - 273.15
            # Probability derived from predicted max temperature
            max_temp = float(np.nanmax(temp_celsius))
            prob = min(1.0, max(0.0, (max_temp - 30.0) / 15.0))
            outputs.append((temp_celsius, prob))
    else:
        # RF path: persistence baseline + classifier probability per day
        for _ in range(days):
            x_flat = x_seq.reshape(1, -1)
            prob = float(model.predict_proba(x_flat)[0, 1])

            # Persistence baseline: last observed temperature frame
            pred_step = np.array(x_seq[-1], copy=True)

            temp_norm = pred_step[1]
            temp_celsius = (temp_norm * (temp_std_scalar + EPSILON)) + temp_mean_scalar
            if np.nanmean(temp_celsius) > 200:
                temp_celsius -= 273.15
            outputs.append((temp_celsius, prob))

            # Autoregressive update for classifier backend
            x_seq = np.concatenate([x_seq[1:], pred_step[None, :, :, :]], axis=0)

    _prediction_cache[cache_key] = outputs
    return outputs


@app.route("/api/predict", methods=["GET"])
def predict_summary():
    """Returns summary statistics for the dashboard."""
    if not resources_ready():
        return jsonify({"error": "Model not loaded"}), 500

    try:
        temp_grid, prob = get_prediction_data()
        max_temp = float(np.max(temp_grid))
        mean_temp = float(np.mean(temp_grid))
        min_temp = float(np.min(temp_grid))

        risk_level, probability = get_risk_and_probability(
            max_temp, model_probability=prob
        )

        return jsonify(
            {
                "status": "ok",
                "date": datetime.datetime.now().strftime("%Y-%m-%d"),
                "risk_level": risk_level,
                "probability": probability,
                "advice": f"Max temp reaching {max_temp:.1f} C. {risk_level} precautions advised.",
                "model_type": runtime_model_type,
                "weather": {
                    "T2M_MAX": max_temp,
                    "T2M": mean_temp,
                    "T2M_MIN": min_temp,
                    "RH2M": None,
                    "WS10M": None,
                },
                "anomaly": {
                    "is_anomaly": risk_level in ["HIGH", "CRITICAL"],
                    "severity": risk_level.lower(),
                    "n_triggers": 0,
                    "triggers": [],
                },
                "bbox": bbox,
            }
        )
    except Exception as e:
        LOGGER.exception("Predict summary failed: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/forecast", methods=["GET"])
def forecast_summary():
    """Returns a 7-day forecast from autoregressive model predictions."""
    if not resources_ready():
        return jsonify({"error": "Model not loaded"}), 500

    try:
        temp_grids = get_prediction_sequence(days=7)

        days = []
        base_date = datetime.datetime.now()

        for i in range(7):
            date = (base_date + datetime.timedelta(days=i + 1)).strftime("%Y-%m-%d")
            temp_grid, prob = temp_grids[i]
            mean_temp = float(np.mean(temp_grid))
            min_temp = float(np.min(temp_grid))
            t_max = float(np.max(temp_grid))
            risk_level, probability = get_risk_and_probability(
                t_max, model_probability=prob
            )

            days.append(
                {
                    "day": i + 1,
                    "date": date,
                    "day_name": (base_date + datetime.timedelta(days=i + 1)).strftime(
                        "%a"
                    ),
                    "probability": probability,
                    "risk_level": risk_level,
                    "risk_label": risk_level,
                    "advice": f"Generated from {runtime_model_type} autoregressive forecast",
                    "weather": {
                        "T2M": mean_temp,
                        "T2M_MAX": t_max,
                        "T2M_MIN": min_temp,
                        "PRECTOTCORR": None,
                        "WS10M": None,
                        "RH2M": None,
                        "NDVI": None,
                    },
                }
            )

        return jsonify(
            {
                "status": "ok",
                "model_type": runtime_model_type,
                "days": 7,
                "generated_at": base_date.isoformat(),
                "forecasts": days,
            }
        )

    except Exception as e:
        LOGGER.exception("Forecast summary failed: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/map", methods=["GET"])
def predict_map():
    """Returns GeoJSON features for the map."""
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

                # Optimization: Only send polygons if temp > 30 to save bandwidth?
                # or send all for heatmap? Let's send all for now but maybe skip very low ones if needed.

                lat = float(lats[r])
                lon = float(lons[c])

                risk = int(risk_grid[r, c])

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
                    "properties": {"temperature": round(val, 2), "risk_level": risk},
                }
                features.append(feature)

        return jsonify({"type": "FeatureCollection", "features": features})

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
    """Simple health check for frontend to detect live backend."""
    return jsonify({"status": "ok", "model_loaded": resources_ready()})


if __name__ == "__main__":
    if not load_resources():
        LOGGER.warning(
            "Startup continuing without loaded model. Web trainer remains available."
        )
    print("Starting Flask API on port 5000...")
    app.run(host="0.0.0.0", port=5000, debug=False)

