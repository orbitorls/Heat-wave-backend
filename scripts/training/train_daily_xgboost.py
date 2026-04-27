"""
XGBoost trainer for daily heatwave prediction.

This model predicts heatwave occurrence from SINGLE DAY weather features,
unlike the sequence-based ConvLSTM that requires 7 days of history.

Usage:
    python train_daily_xgboost.py

Features:
- Uses daily weather features (temp_mean, temp_max, humidity, pressure, etc.)
- Labels each day as heatwave (1) or not (0) based on temperature threshold
- Trains XGBoost classifier with class balancing
- Much simpler and more memory-efficient than sequence models
"""

import os
import sys
import glob
import time
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.loader import DataLoader, fill_nan_along_time
from src.core.config import settings

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

try:
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
    from sklearn.model_selection import TimeSeriesSplit
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Error: sklearn required. Install with: pip install scikit-learn")


# Configuration (defaults, can be overridden by settings or config dict)
DATA_DIR = str(settings.DATA_DIR)
MODELS_DIR = str(settings.MODELS_DIR)
HEATWAVE_TEMP_THRESHOLD = settings.HEATWAVE_TEMP_THRESHOLD
HEATWAVE_MIN_DURATION = settings.HEATWAVE_MIN_DURATION
TRAIN_RATIO = settings.TRAIN_RATIO
VAL_RATIO = settings.VAL_RATIO
TEST_RATIO = settings.TEST_RATIO
RANDOM_SEED = settings.RANDOM_SEED


@dataclass
class DailyFeatures:
    """Daily weather features for heatwave prediction."""
    # Temperature features
    temp_mean: float
    temp_max: float
    temp_min: float
    temp_std: float
    temp_range: float
    
    # Humidity features
    humidity_mean: float
    humidity_max: float
    
    # Pressure/geopotential features
    z_mean: float
    z_std: float
    
    # Soil moisture
    swvl1_mean: float
    
    # Precipitation
    tp_mean: float
    tp_max: float
    
    # Spatial features
    hot_fraction: float  # Fraction of grid above threshold
    
    # Location features
    lat_mean: float
    lon_mean: float
    
    # Temporal features
    month: int
    day_of_year: int


def extract_daily_features(
    data: np.ndarray,
    temp_channel: int = 1,
    threshold_c: float = 35.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract daily features from ERA5 data.
    
    Args:
        data: Raw data array of shape (Time, Channels, H, W) in Kelvin
        temp_channel: Index of temperature channel
        threshold_c: Temperature threshold in Celsius
        
    Returns:
        features: Array of shape (Time, N_features)
        labels: Array of shape (Time,) with heatwave labels
        temps_c: Array of shape (Time,) with mean temperature in Celsius
    """
    n_time = data.shape[0]
    n_channels = data.shape[1]
    h, w = data.shape[2], data.shape[3]
    
    # Get temperature in Celsius
    temp_data = data[:, temp_channel, :, :]  # (Time, H, W) in Kelvin
    temp_c = temp_data - 273.15  # Convert to Celsius
    
    features_list = []
    
    for t in range(n_time):
        # Temperature features
        temp_t = temp_c[t]  # (H, W)
        temp_mean = np.nanmean(temp_t)
        temp_max = np.nanmax(temp_t)
        temp_min = np.nanmin(temp_t)
        temp_std = np.nanstd(temp_t)
        temp_range = temp_max - temp_min
        
        # Hot fraction: what percentage of grid is above threshold
        hot_fraction = np.nanmean(temp_t >= threshold_c)
        
        # Humidity features (if available)
        humidity_mean = 0.0
        humidity_max = 0.0
        if n_channels > 4:
            humidity_t = data[t, 4, :, :]
            humidity_mean = np.nanmean(humidity_t)
            humidity_max = np.nanmax(humidity_t)
        
        # Geopotential features
        z_mean = 0.0
        z_std = 0.0
        if n_channels > 0:
            z_t = data[t, 0, :, :]
            z_mean = np.nanmean(z_t)
            z_std = np.nanstd(z_t)
        
        # Soil moisture features
        swvl1_mean = 0.0
        if n_channels > 2:
            swvl1_t = data[t, 2, :, :]
            swvl1_mean = np.nanmean(swvl1_t)
        
        # Precipitation features
        tp_mean = 0.0
        tp_max = 0.0
        if n_channels > 3:
            tp_t = data[t, 3, :, :]
            tp_mean = np.nanmean(tp_t)
            tp_max = np.nanmax(tp_t)
        
        # Location features (static)
        lat_mean = 15.0  # Thailand center
        lon_mean = 100.0
        
        # Create feature vector
        feat = [
            temp_mean, temp_max, temp_min, temp_std, temp_range,
            humidity_mean, humidity_max,
            z_mean, z_std,
            swvl1_mean,
            tp_mean, tp_max,
            hot_fraction,
            lat_mean, lon_mean,
        ]
        
        features_list.append(feat)
    
    features = np.array(features_list, dtype=np.float32)
    temps_c = np.nanmean(temp_c, axis=(1, 2))  # Mean temp per day
    
    # Create labels: heatwave if temp >= threshold
    # But we need consecutive days for true heatwave detection
    # For single-day prediction, we use: is this day part of a heatwave event?
    labels = (temps_c >= threshold_c).astype(np.int32)
    
    return features, labels, temps_c


def create_heatwave_labels_with_duration(
    temps_c: np.ndarray,
    threshold_c: float = 35.0,
    min_duration: int = 3
) -> np.ndarray:
    """
    Create heatwave labels considering consecutive days.
    
    A day is labeled as heatwave (1) if:
    - Temperature >= threshold for at least min_duration consecutive days
    
    This is more realistic than single-day threshold.
    """
    n_days = len(temps_c)
    labels = np.zeros(n_days, dtype=np.int32)
    
    is_hot = temps_c >= threshold_c
    
    # Find consecutive hot days
    consecutive_count = 0
    for i in range(n_days):
        if is_hot[i]:
            consecutive_count += 1
        else:
            consecutive_count = 0
        
        # Mark as heatwave if we've had enough consecutive hot days
        if consecutive_count >= min_duration:
            # Mark this day and previous min_duration-1 days as heatwave
            for j in range(max(0, i - min_duration + 1), i + 1):
                labels[j] = 1
    
    return labels


def train_xgboost_daily(
    config: Optional[Dict[str, Any]] = None,
    on_progress: Optional[Callable[[str, float, Optional[Dict[str, Any]]], None]] = None,
    on_log: Optional[Callable[[str, str], None]] = None
) -> Optional[Dict[str, Any]]:
    """
    Train XGBoost model for daily heatwave prediction.

    Uses SAME DATA as the ConvLSTM model but with DAILY FEATURES
    instead of sequences. This makes it suitable for real-time prediction
    from current weather conditions.

    Args:
        config: Optional config dict to override defaults
        on_progress: Callback for progress updates (stage, progress, metadata)
        on_log: Callback for log messages (level, message)

    Returns:
        Dict with training results or None if failed
    """
    def log(level: str, message: str):
        """Internal logging helper."""
        if on_log:
            on_log(level, message)
        else:
            print(f"[{level.upper()}] {message}")

    def progress(stage: str, value: float, metadata: Optional[Dict[str, Any]] = None):
        """Internal progress helper."""
        if on_progress:
            on_progress(stage, value, metadata or {})

    if not XGBOOST_AVAILABLE:
        log("error", "XGBoost not installed. Run: pip install xgboost")
        return None

    if not SKLEARN_AVAILABLE:
        log("error", "sklearn not installed. Run: pip install scikit-learn")
        return None
    
    cfg = {
        "data_dir": DATA_DIR,
        "models_dir": MODELS_DIR,
        "heatwave_temp_threshold": HEATWAVE_TEMP_THRESHOLD,
        "heatwave_min_duration": HEATWAVE_MIN_DURATION,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
        "random_seed": RANDOM_SEED,
        "n_estimators": settings.RF_N_ESTIMATORS if hasattr(settings, 'RF_N_ESTIMATORS') else 200,
        "max_depth": settings.RF_MAX_DEPTH if hasattr(settings, 'RF_MAX_DEPTH') else 10,
        "learning_rate": settings.LEARNING_RATE if hasattr(settings, 'LEARNING_RATE') else 0.1,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }
    if config:
        cfg.update(config)

    log("info", "=" * 60)
    log("info", "XGBoost Daily Heatwave Prediction Training")
    log("info", "=" * 60)
    log("info", f"Config:")
    log("info", f"  - Temperature threshold: {cfg['heatwave_temp_threshold']}C")
    log("info", f"  - Min consecutive days: {cfg['heatwave_min_duration']}")
    log("info", f"  - Train/Val/Test: {cfg['train_ratio']:.0%}/{cfg['val_ratio']:.0%}/{cfg['test_ratio']:.0%}")
    log("info", f"  - XGBoost n_estimators: {cfg['n_estimators']}")
    log("info", f"  - XGBoost max_depth: {cfg['max_depth']}")

    progress("init", 0.0, {"stage": "initialization"})

    # Set random seed
    np.random.seed(cfg["random_seed"])

    # [1/5] Load data
    log("info", "[1/5] Loading ERA5 + NASA POWER data...")
    progress("loading", 0.2, {"stage": "data_loading"})
    from pathlib import Path
    loader = DataLoader()
    loader.data_dir = Path(cfg["data_dir"])

    try:
        full_ds = loader.load_combined()
        data_raw, stats = loader.prepare_training_data(full_ds, fill_nan=False)
        lats, lons = stats["lats"], stats["lons"]
        log("info", f"  - Data shape: {data_raw.shape}")
        log("info", f"  - Time steps: {data_raw.shape[0]}")
        log("info", f"  - Channels: {data_raw.shape[1]}")
        log("info", f"  - Grid: {data_raw.shape[2]} x {data_raw.shape[3]}")
    except Exception as e:
        log("error", f"Error loading data: {e}")
        return None
    
    # [2/5] Temporal split and fill NaN
    log("info", "\n[2/5] Temporal split (train/val/test)...")
    progress("splitting", 0.4, {"stage": "data_splitting"})
    train_end = int(data_raw.shape[0] * cfg["train_ratio"])
    val_end = int(data_raw.shape[0] * (cfg["train_ratio"] + cfg["val_ratio"]))

    train_data = data_raw[:train_end].copy()
    val_data = data_raw[train_end:val_end].copy()
    test_data = data_raw[val_end:].copy()

    # Fill NaN separately per split (no leakage)
    for ch in range(train_data.shape[1]):
        train_data[:, ch, :, :] = fill_nan_along_time(train_data[:, ch, :, :])
        val_data[:, ch, :, :] = fill_nan_along_time(val_data[:, ch, :, :])
        test_data[:, ch, :, :] = fill_nan_along_time(test_data[:, ch, :, :])

    log("info", f"  - Train: {train_data.shape[0]} days")
    log("info", f"  - Val: {val_data.shape[0]} days")
    log("info", f"  - Test: {test_data.shape[0]} days")
    
    # Compute normalization from training split only
    train_mean = train_data.mean(axis=(0, 2, 3), keepdims=True)
    train_std = train_data.std(axis=(0, 2, 3), keepdims=True)
    train_std = np.where(train_std < 1e-8, 1e-8, train_std)
    
    train_norm = (train_data - train_mean) / train_std
    val_norm = (val_data - train_mean) / train_std
    test_norm = (test_data - train_mean) / train_std
    
    # [3/5] Extract daily features
    log("info", "\n[3/5] Extracting daily features...")
    progress("features", 0.6, {"stage": "feature_extraction"})
    
    def extract_features_from_normalized(data_raw, temp_mean_k, temp_std_k):
        """Extract features from raw (not normalized) data - vectorized."""
        # Use raw data for temperature in Kelvin
        temp_channel = 1
        temp_data = data_raw[:, temp_channel, :, :].copy()  # Already in Kelvin
        
        # Convert to Celsius
        temp_c = temp_data - 273.15
        
        n_time = data_raw.shape[0]
        
        # Debug: print temperature range
        valid_temps = temp_data[temp_data > 0]  # Exclude zeros/NaNs
        if len(valid_temps) > 0:
            log("info", f"    Temperature range: {valid_temps.min():.1f}K to {valid_temps.max():.1f}K ({valid_temps.min()-273.15:.1f}C to {valid_temps.max()-273.15:.1f}C)")
        
        # Vectorized temperature statistics across time
        valid_mask = temp_c > -273  # Valid Celsius temps
        
        # Handle all time steps at once
        temp_mean_t = np.where(
            np.any(valid_mask, axis=(1, 2)),
            np.nanmean(np.where(valid_mask, temp_c, np.nan), axis=(1, 2)),
            0.0
        )
        temp_max_t = np.where(
            np.any(valid_mask, axis=(1, 2)),
            np.nanmax(np.where(valid_mask, temp_c, np.nan), axis=(1, 2)),
            0.0
        )
        temp_min_t = np.where(
            np.any(valid_mask, axis=(1, 2)),
            np.nanmin(np.where(valid_mask, temp_c, np.nan), axis=(1, 2)),
            0.0
        )
        temp_std_t = np.where(
            np.any(valid_mask, axis=(1, 2)),
            np.nanstd(np.where(valid_mask, temp_c, np.nan), axis=(1, 2)),
            0.0
        )
        temp_range_t = temp_max_t - temp_min_t
        hot_frac = np.where(
            np.any(valid_mask, axis=(1, 2)),
            np.nanmean(np.where(valid_mask, temp_c >= cfg["heatwave_temp_threshold"], 0), axis=(1, 2)),
            0.0
        )
        
        # Vectorized other channel statistics
        z_mean = np.zeros(n_time, dtype=np.float32)
        z_std = np.zeros(n_time, dtype=np.float32)
        swvl1_mean = np.zeros(n_time, dtype=np.float32)
        tp_mean = np.zeros(n_time, dtype=np.float32)
        humidity_mean = np.zeros(n_time, dtype=np.float32)
        
        if data_raw.shape[1] > 0:
            z_mean = np.nanmean(data_raw[:, 0, :, :], axis=(1, 2))
            z_std = np.nanstd(data_raw[:, 0, :, :], axis=(1, 2))
        if data_raw.shape[1] > 2:
            swvl1_mean = np.nanmean(data_raw[:, 2, :, :], axis=(1, 2))
        if data_raw.shape[1] > 3:
            tp_mean = np.nanmean(data_raw[:, 3, :, :], axis=(1, 2))
        if data_raw.shape[1] > 4:
            humidity_mean = np.nanmean(data_raw[:, 4, :, :], axis=(1, 2))
        
        # Stack features
        features = np.column_stack([
            temp_mean_t, temp_max_t, temp_min_t, temp_std_t, temp_range_t,
            hot_frac, z_mean, z_std, swvl1_mean, tp_mean, humidity_mean
        ]).astype(np.float32)
        
        # Compute mean temperature per day (vectorized)
        daily_mean_temps = np.where(
            np.any(valid_mask, axis=(1, 2)),
            np.nanmean(np.where(valid_mask, temp_c, np.nan), axis=(1, 2)),
            np.nan
        )
        
        # Remove NaN from output
        valid_day_mask = ~np.isnan(daily_mean_temps)
        if not np.all(valid_day_mask):
            log("warning", f"    Warning: {np.sum(~valid_day_mask)} days have no valid temperature data")
        
        return features, daily_mean_temps
    
    # Get temperature mean/std in Kelvin
    temp_mean_k = float(train_mean[0, 1, 0, 0])
    temp_std_k = float(train_std[0, 1, 0, 0])
    
    # Use RAW data for temperature extraction (before normalization)
    # This ensures we get actual Celsius values
    X_train, train_temps_c = extract_features_from_normalized(train_data, temp_mean_k, temp_std_k)
    X_val, val_temps_c = extract_features_from_normalized(val_data, temp_mean_k, temp_std_k)
    X_test, test_temps_c = extract_features_from_normalized(test_data, temp_mean_k, temp_std_k)

    log("info", f"  - Feature shape: {X_train.shape}")

    # [4/5] Create labels
    log("info", "\n[4/5] Creating heatwave labels...")
    progress("labels", 0.7, {"stage": "label_creation"})
    
    # Use MAX temperature (not mean) for Thailand heatwave detection
    # Thailand can reach 40-44C during hot season
    train_max_temps = X_train[:, 1]  # temp_max is at index 1
    val_max_temps = X_val[:, 1]
    test_max_temps = X_test[:, 1]
    
    # Debug: print temperature stats
    valid_max_temps = train_max_temps[train_max_temps > -273]
    log("info", f"  - Max Temperature range: {valid_max_temps.min():.1f}C to {valid_max_temps.max():.1f}C")
    log("info", f"  - Mean Max Temperature: {valid_max_temps.mean():.1f}C")
    log("info", f"  - Heatwave threshold: {cfg['heatwave_temp_threshold']}C (Max Temp)")

    # Count days above threshold
    hot_days_train = np.sum(train_max_temps >= cfg['heatwave_temp_threshold'])
    hot_days_val = np.sum(val_max_temps >= cfg['heatwave_temp_threshold'])
    hot_days_test = np.sum(test_max_temps >= cfg['heatwave_temp_threshold'])
    log("info", f"  - Days >= {cfg['heatwave_temp_threshold']}C: Train={hot_days_train}, Val={hot_days_val}, Test={hot_days_test}")
    
    # Create labels based on MAX temperature threshold
    y_train = create_heatwave_labels_with_duration(
        train_max_temps, 
        threshold_c=cfg["heatwave_temp_threshold"],
        min_duration=cfg["heatwave_min_duration"]
    )
    y_val = create_heatwave_labels_with_duration(
        val_max_temps, 
        threshold_c=cfg["heatwave_temp_threshold"],
        min_duration=cfg["heatwave_min_duration"]
    )
    y_test = create_heatwave_labels_with_duration(
        test_max_temps,
        threshold_c=cfg["heatwave_temp_threshold"],
        min_duration=cfg["heatwave_min_duration"]
    )

    log("info", f"  - Train: {y_train.sum()}/{len(y_train)} positive ({y_train.mean()*100:.1f}%)")
    log("info", f"  - Val: {y_val.sum()}/{len(y_val)} positive ({y_val.mean()*100:.1f}%)")
    log("info", f"  - Test: {y_test.sum()}/{len(y_test)} positive ({y_test.mean()*100:.1f}%)")

    if y_train.sum() == 0 or y_val.sum() == 0 or y_test.sum() == 0:
        log("warning", "Warning: No positive samples in one or more splits!")
        log("warning", "  - Trying with lower threshold...")
        # Fallback to percentile-based threshold using valid MAX temps only
        valid_max_temps = train_max_temps[train_max_temps > -273]  # Exclude NaN/invalid
        if len(valid_max_temps) == 0:
            log("error", "  - ERROR: No valid temperature values!")
            return None
        # Use 90th percentile of MAX temperature as fallback
        threshold_c = np.percentile(valid_max_temps, 90)
        y_train = (train_max_temps >= threshold_c).astype(np.int32)
        y_val = (val_max_temps >= threshold_c).astype(np.int32)
        y_test = (test_max_temps >= threshold_c).astype(np.int32)
        log("info", f"  - Using threshold: {threshold_c:.2f}C (90th percentile of MAX temps)")
        log("info", f"  - Valid max temp range: {valid_max_temps.min():.1f}C to {valid_max_temps.max():.1f}C")
        log("info", f"  - Train: {y_train.sum()}/{len(y_train)} positive ({y_train.mean()*100:.1f}%)")

    # [5/5] Train XGBoost
    log("info", "\n[5/5] Training XGBoost classifier...")
    progress("training", 0.8, {"stage": "model_training"})
    
    # Compute class weight
    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    scale_pos_weight = n_neg / max(n_pos, 1)

    log("info", f"  - Class balance: {n_neg} negative, {n_pos} positive")
    log("info", f"  - scale_pos_weight: {scale_pos_weight:.2f}")

    start_time = time.time()
    
    model = xgb.XGBClassifier(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        learning_rate=cfg["learning_rate"],
        min_child_weight=cfg["min_child_weight"],
        subsample=cfg["subsample"],
        colsample_bytree=cfg["colsample_bytree"],
        scale_pos_weight=scale_pos_weight,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=cfg["random_seed"],
        n_jobs=-1,
        verbosity=0,
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    train_time = time.time() - start_time
    log("info", f"  - Training time: {train_time:.2f}s")

    # Evaluate
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    train_metrics = {
        "accuracy": accuracy_score(y_train, train_pred),
        "precision": precision_score(y_train, train_pred, zero_division=0),
        "recall": recall_score(y_train, train_pred, zero_division=0),
        "f1": f1_score(y_train, train_pred, zero_division=0),
    }

    val_metrics = {
        "accuracy": accuracy_score(y_val, val_pred),
        "precision": precision_score(y_val, val_pred, zero_division=0),
        "recall": recall_score(y_val, val_pred, zero_division=0),
        "f1": f1_score(y_val, val_pred, zero_division=0),
    }

    test_metrics = {
        "accuracy": accuracy_score(y_test, test_pred),
        "precision": precision_score(y_test, test_pred, zero_division=0),
        "recall": recall_score(y_test, test_pred, zero_division=0),
        "f1": f1_score(y_test, test_pred, zero_division=0),
    }

    log("info", f"\n  Train: Acc={train_metrics['accuracy']:.3f}, P={train_metrics['precision']:.3f}, R={train_metrics['recall']:.3f}, F1={train_metrics['f1']:.3f}")
    log("info", f"  Val:   Acc={val_metrics['accuracy']:.3f}, P={val_metrics['precision']:.3f}, R={val_metrics['recall']:.3f}, F1={val_metrics['f1']:.3f}")
    log("info", f"  Test:  Acc={test_metrics['accuracy']:.3f}, P={test_metrics['precision']:.3f}, R={test_metrics['recall']:.3f}, F1={test_metrics['f1']:.3f}")

    # Feature importance
    feature_names = [
        "temp_mean", "temp_max", "temp_min", "temp_std", "temp_range", "hot_fraction",
        "z_mean", "z_std", "swvl1_mean", "tp_mean", "humidity_mean"
    ]
    importance = model.feature_importances_
    log("info", f"\n  Feature importance:")
    for name, imp in sorted(zip(feature_names[:len(importance)], importance), key=lambda x: -x[1]):
        log("info", f"    - {name}: {imp:.4f}")
    
    # Save checkpoint
    version = _get_next_version(cfg["models_dir"])
    save_filename = f"heatwave_daily_xgboost_v{version}.pth"
    save_path = os.path.join(cfg["models_dir"], save_filename)
    
    # Temp mean in Celsius for inference
    temp_mean_celsius = temp_mean_k - 273.15 if temp_mean_k > 200 else temp_mean_k
    
    checkpoint = {
        "model_type": "xgboost_daily",
        "sklearn_model": model,
        "feature_names": feature_names,
        "metadata": {
            "task_type": "daily_heatwave_classification",
            "heatwave_temp_threshold": cfg["heatwave_temp_threshold"],
            "heatwave_min_duration": cfg["heatwave_min_duration"],
            "n_estimators": cfg["n_estimators"],
            "max_depth": cfg["max_depth"],
            "learning_rate": cfg["learning_rate"],
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "temp_mean_celsius": temp_mean_celsius,
            "temp_std": temp_std_k,
            "normalization_mean": train_mean.tolist(),
            "normalization_std": train_std.tolist(),
            "feature_importance": dict(zip(feature_names[:len(importance)], importance.tolist())),
            "lats": lats.tolist(),
            "lons": lons.tolist(),
            "train_ratio": cfg["train_ratio"],
            "val_ratio": cfg["val_ratio"],
            "test_ratio": cfg["test_ratio"],
            "train_positive_rate": float(y_train.mean()),
            "val_positive_rate": float(y_val.mean()),
            "test_positive_rate": float(y_test.mean()),
            "training_time_seconds": train_time,
            "created_at": datetime.now().isoformat(),
        },
    }
    
    os.makedirs(cfg["models_dir"], exist_ok=True)
    import torch
    torch.save(checkpoint, save_path)
    log("info", f"\nModel saved to: {save_path}")
    progress("saving", 0.9, {"stage": "saving_model"})

    # Generate training report
    report_path = _generate_training_report(
        train_metrics, val_metrics, test_metrics,
        checkpoint["metadata"]["feature_importance"],
        cfg, save_path, version, on_log
    )
    
    return {
        "save_path": save_path,
        "report_path": report_path,
        "model_type": "xgboost_daily",
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "feature_importance": checkpoint["metadata"]["feature_importance"],
    }


def _get_next_version(model_dir: str, base_name: str = "heatwave_daily_xgboost") -> int:
    """Get next version number for checkpoint."""
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        return 1
    
    files = glob.glob(os.path.join(model_dir, f"{base_name}_v*.pth"))
    if not files:
        return 1
    
    versions = []
    for f in files:
        try:
            version = int(f.split("_v")[-1].split(".")[0])
            versions.append(version)
        except (ValueError, IndexError):
            continue
    
    return max(versions) + 1 if versions else 1


def _generate_training_report(train_metrics, val_metrics, test_metrics,
                              feature_importance, cfg, save_path, version=1, on_log=None):
    """Generate a training report in output directory."""
    def log(level: str, message: str):
        if on_log:
            on_log(level, message)
        else:
            print(f"[{level.upper()}] {message}")

    import matplotlib.pyplot as plt
    import matplotlib
    
    matplotlib.use('Agg')  # Non-interactive backend
    
    os.makedirs("output", exist_ok=True)
    
    # Create report figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"XGBoost Daily Heatwave Training Report\n{datetime.now().strftime('%Y-%m-%d %H:%M')}", 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Metrics Comparison
    ax1 = axes[0, 0]
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1']
    train_vals = [train_metrics['accuracy'], train_metrics['precision'], 
                  train_metrics['recall'], train_metrics['f1']]
    val_vals = [val_metrics['accuracy'], val_metrics['precision'], 
                val_metrics['recall'], val_metrics['f1']]
    test_vals = [test_metrics['accuracy'], test_metrics['precision'], 
                 test_metrics['recall'], test_metrics['f1']]
    
    x = np.arange(len(metrics_names))
    width = 0.25
    
    ax1.bar(x - width, train_vals, width, label='Train', color='#2ecc71')
    ax1.bar(x, val_vals, width, label='Val', color='#3498db')
    ax1.bar(x + width, test_vals, width, label='Test', color='#e74c3c')
    
    ax1.set_ylabel('Score')
    ax1.set_title('Model Performance Metrics')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_names)
    ax1.legend()
    ax1.set_ylim(0, 1.1)
    ax1.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5, label='Good threshold')
    
    # Plot 2: Feature Importance
    ax2 = axes[0, 1]
    if feature_importance:
        sorted_features = sorted(feature_importance.items(), key=lambda x: -x[1])[:10]
        names = [f[0] for f in sorted_features]
        values = [f[1] for f in sorted_features]
        
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(names)))
        ax2.barh(names, values, color=colors)
        ax2.set_xlabel('Importance')
        ax2.set_title('Top 10 Feature Importance')
        ax2.invert_yaxis()
    
    # Plot 3: Summary Table
    ax3 = axes[1, 0]
    ax3.axis('off')
    
    summary_text = (
        f"Model: XGBoost Classifier\n"
        f"─────────────────────────────\n"
        f"Temperature Threshold: {cfg['heatwave_temp_threshold']}C\n"
        f"Min Duration: {cfg['heatwave_min_duration']} days\n"
        f"─────────────────────────────\n"
        f"Training Samples: N/A\n"
        f"Validation Samples: N/A\n"
        f"Test Samples: N/A\n"
        f"─────────────────────────────\n"
        f"XGBoost Parameters:\n"
        f"  n_estimators: {cfg['n_estimators']}\n"
        f"  max_depth: {cfg['max_depth']}\n"
        f"  learning_rate: {cfg['learning_rate']}\n"
        f"─────────────────────────────\n"
        f"Test F1 Score: {test_metrics['f1']:.4f}\n"
        f"Test Precision: {test_metrics['precision']:.4f}\n"
        f"Test Recall: {test_metrics['recall']:.4f}\n"
        f"─────────────────────────────\n"
        f"Saved to: {os.path.basename(save_path)}"
    )
    
    ax3.text(0.1, 0.9, summary_text, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8))
    
    # Plot 4: Performance Assessment
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    test_f1 = test_metrics['f1']
    if test_f1 >= 0.8:
        assessment = "Good"
        color = '#2ecc71'
        emoji = "Good"
    elif test_f1 >= 0.6:
        assessment = "Moderate"
        color = '#f39c12'
        emoji = "Moderate"
    else:
        assessment = "Needs Improvement"
        color = '#e74c3c'
        emoji = "Needs Improvement"
    
    assessment_text = (
        f"Performance Assessment\n"
        f"─────────────────────────────\n\n"
        f"F1 Score: {test_f1:.4f}\n\n"
        f"Assessment: {emoji}\n\n"
        f"Performance: {assessment}\n\n"
    )
    
    ax4.text(0.2, 0.5, assessment_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
    
    plt.tight_layout()

    # Save report
    report_path = os.path.join("output", f"xgboost_daily_report_v{version}.png")
    plt.savefig(report_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    log("info", f"      Training report saved: {report_path}")
    return report_path


if __name__ == "__main__":
    result = train_xgboost_daily()
    
    if result:
        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print(f"Model saved: {result['save_path']}")
        print(f"Test F1: {result['test_metrics']['f1']:.4f}")
        print(f"Test Precision: {result['test_metrics']['precision']:.4f}")
        print(f"Test Recall: {result['test_metrics']['recall']:.4f}")
        print("=" * 60)
    else:
        print("\nTraining failed!")
        sys.exit(1)
