from src.data.loader import DataLoader, create_sequences, fill_nan_along_time
from src.core.logger import logger as app_logger
from src.core.config import settings

import glob
import os
import time
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

DATA_DIR = str(settings.DATA_DIR)
MODELS_DIR = str(settings.MODELS_DIR)


def _int_env(name, default_value):
    raw = os.environ.get(name)
    if raw is None:
        return default_value
    try:
        value = int(raw)
        return value if value > 0 else default_value
    except ValueError:
        return default_value


def _float_env(name, default_value):
    raw = os.environ.get(name)
    if raw is None:
        return default_value
    try:
        return float(raw)
    except ValueError:
        return default_value


def _ratio_env(name, default_value):
    value = _float_env(name, default_value)
    if value <= 0 or value >= 1:
        return default_value
    return value


def _bool_env(name, default_value):
    raw = os.environ.get(name)
    if raw is None:
        return default_value
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _str_env(name, default_value):
    raw = os.environ.get(name)
    return raw.strip() if raw is not None and raw.strip() else default_value


def _set_random_seeds(seed: int = 42) -> None:
    """Set global random seeds for reproducibility."""
    import random
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass  # torch may not be available in all contexts


BATCH_SIZE = settings.BATCH_SIZE
SEQ_LEN = settings.SEQ_LEN
FUTURE_SEQ = settings.FUTURE_SEQ
RF_N_ESTIMATORS = settings.RF_N_ESTIMATORS
RF_MAX_DEPTH = settings.RF_MAX_DEPTH
RF_MIN_SAMPLES_LEAF = settings.RF_MIN_SAMPLES_LEAF

# Class balance optimization
MIN_TRAIN_POSITIVE_RATE = settings.MIN_TRAIN_POSITIVE_RATE
MAX_TRAIN_POSITIVE_RATE = settings.MAX_TRAIN_POSITIVE_RATE

# Event detection
EVENT_MIN_DURATION_DAYS = settings.EVENT_MIN_DURATION_DAYS
EVENT_MIN_HOT_FRACTION = settings.EVENT_MIN_HOT_FRACTION
RF_SAMPLING_STRATEGY = "all"
RF_REPLACEMENT = True
CLIP_LOW_PERCENTILE = 0.5
CLIP_HIGH_PERCENTILE = 99.5
MODEL_BACKEND = "balanced_rf"
USE_XGBOOST = settings.USE_XGBOOST
USE_LIGHTGBM = settings.USE_LIGHTGBM
USE_GPU = False
FORCE_GPU = False

# Data Split config
TRAIN_RATIO = settings.TRAIN_RATIO
VAL_RATIO = settings.VAL_RATIO
TEST_RATIO = settings.TEST_RATIO
HEATWAVE_PERCENTILE = 95.0
RANDOM_SEED = settings.RANDOM_SEED
EPOCHS = settings.EPOCHS
LEARNING_RATE = settings.LEARNING_RATE

# Event detection parameters
ALLOW_SAMPLE_MEAN_FALLBACK = True
REQUIRE_DYNAMIC_FEATURES = False
MIN_EVAL_POSITIVE_COUNT = 10
LABELING_METHOD = "temperature"
HEATWAVE_HEAT_INDEX_THRESHOLD = 41.0
USE_NASA_POWER = True
HEATWAVE_TEMPERATURE_THRESHOLD = settings.HEATWAVE_TEMP_THRESHOLD
HEATWAVE_ANOMALY_THRESHOLD = settings.HEATWAVE_ANOMALY_THRESHOLD

# Walk-Forward Validation config
WALK_FORWARD_ENABLED = settings.WALK_FORWARD_ENABLED
WALK_FORWARD_N_FOLDS = 3
WALK_FORWARD_EXPAND_WINDOW = True

# Hyperparameter Tuning config
HP_TUNING_ENABLED = False
HP_N_ITER = 20
HP_CV_FOLDS = 3
HP_SCORING = "f1"

# Feature Engineering config
FEATURE_ENGINEERING_ENABLED = settings.FEATURE_ENGINEERING_ENABLED
FEATURE_INTERACTIONS = True
FEATURE_TEMPORAL = True


@dataclass
class EpochMetrics:
    epoch: int
    total_epochs: int
    train_loss: float
    val_loss: float
    val_rmse: float
    val_event_f1: float
    elapsed_seconds: float


def get_next_version(
    model_dir: str, base_name: str = "heatwave_model_checkpoint"
) -> int:
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        return 1

    files = glob.glob(os.path.join(model_dir, f"{base_name}_v*.pth"))
    if not files:
        return 1

    versions = []
    for file_path in files:
        try:
            part = file_path.split("_v")[-1]
            versions.append(int(part.split(".")[0]))
        except (ValueError, IndexError):
            continue
    return max(versions) + 1 if versions else 1


def temporal_split_data(data, train_ratio=0.7, val_ratio=0.15, test_ratio: float = None):
    """Split time-series data into train/val/test sets maintaining temporal order.
    
    Args:
        data: Input array to split
        train_ratio: Fraction for training (default 0.7 = 70%)
        val_ratio: Fraction for validation (default 0.15 = 15%)
        test_ratio: Fraction for testing (default None = auto-computed as 1.0 - train_ratio - val_ratio)
        
    Returns:
        train_data, val_data, test_data
        
    Raises:
        ValueError: If ratios don't sum to 1.0 or produce empty splits
    """
    # Auto-compute test_ratio if not provided
    if test_ratio is None:
        test_ratio = 1.0 - train_ratio - val_ratio
    
    total_ratio = train_ratio + val_ratio + test_ratio
    
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {train_ratio} + {val_ratio} + {test_ratio} = {total_ratio}"
        )

    total_steps = len(data)
    train_end = int(total_steps * train_ratio)
    val_end = int(total_steps * (train_ratio + val_ratio))

    if train_end <= 0 or val_end <= train_end or val_end >= total_steps:
        raise ValueError("Temporal split produced empty train/val/test partitions")

    return data[:train_end], data[train_end:val_end], data[val_end:]


def _to_heatwave_event_labels(
    temp_sequences_c, threshold_c, min_duration=3, min_hot_fraction=0.10
):
    hot_fraction = (temp_sequences_c >= threshold_c).mean(axis=(2, 3))
    hot = hot_fraction >= float(min_hot_fraction)
    duration = min(min_duration, hot.shape[1])

    events = np.zeros(hot.shape[0], dtype=np.int32)
    for idx in range(hot.shape[0]):
        run = 0
        for step_idx in range(hot.shape[1]):
            if hot[idx, step_idx]:
                run += 1
                if run >= duration:
                    events[idx] = 1
                    break
            else:
                run = 0
    return events


def _to_heatwave_anomaly_labels(
    temp_sequences_c, climatology_c, anomaly_threshold_c=5.0, min_duration=3, min_hot_fraction=0.10
):
    """
    Heatwave defined by temperature ANOMALY (deviation from normal).
    
    A day is a heatwave event if:
    - Temperature deviates from climatology by >= anomaly_threshold_c
    - This persists for >= min_duration days
    - And covers >= min_hot_fraction of the region
    
    This solves the problem where hot regions (north) would be flagged everyday
    while cooler regions (south) might never be flagged with absolute temperature.
    """
    # Ensure climatology is exactly 2D (H, W)
    clim = np.array(climatology_c).flatten()[-temp_sequences_c.shape[2]*temp_sequences_c.shape[3]:].reshape(temp_sequences_c.shape[2], temp_sequences_c.shape[3])
    
    # Compute anomaly: how much hotter than normal
    # temp_sequences_c: (N, T, H, W), clim: (H, W)
    anomaly = temp_sequences_c - clim[np.newaxis, np.newaxis, :, :]
    
    # Count fraction of grid that's anomalously hot
    # anomaly shape: (N, T, H, W) -> mean over (H, W) -> (N, T)
    hot_fraction = (anomaly >= anomaly_threshold_c).mean(axis=(2, 3))
    
    # Force reshape to 2D if needed
    hot_fraction = np.array(hot_fraction).reshape(-1, hot_fraction.shape[-1] if hot_fraction.ndim > 1 else 1)
    
    # Ensure 2D (N, T)
    if hot_fraction.shape[0] == 1:
        hot_fraction = hot_fraction.T
    
    # hot_fraction >= threshold -> boolean array (N, T)
    hot = (hot_fraction >= float(min_hot_fraction)).astype(bool)
    
    duration = min(min_duration, hot.shape[1])
    
    events = np.zeros(hot.shape[0], dtype=np.int32)
    for idx in range(hot.shape[0]):
        run = 0
        for step_idx in range(hot.shape[1]):
            if hot[idx, step_idx]:
                run += 1
                if run >= duration:
                    events[idx] = 1
                    break
            else:
                run = 0
    
    print(f"      [Anomaly Debug] positive={events.sum()}/{events.shape[0]} ({events.mean()*100:.1f}%)")
    return events


def _classification_metrics(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    hit_rate = recall
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "hit_rate": float(hit_rate),
        "false_alarm_rate": float(false_alarm_rate),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def _flatten_features(x_seq):
    return x_seq.reshape(x_seq.shape[0], -1)


def _flatten_targets(y_seq):
    return y_seq.reshape(y_seq.shape[0], -1)


def _to_temperature_c(y_norm, temp_mean, temp_std, temp_channel_idx=1):
    temp = (y_norm[:, :, temp_channel_idx, :, :] * temp_std) + temp_mean
    if np.nanmean(temp) > 200:
        temp = temp - 273.15
    return temp


def _compute_rh_from_temp_and_dewpoint_c(temp_c, dewpoint_c):
    # August-Roche-Magnus approximation.
    a = 17.625
    b = 243.04
    alpha_td = (a * dewpoint_c) / (b + dewpoint_c + 1e-9)
    alpha_t = (a * temp_c) / (b + temp_c + 1e-9)
    rh = 100.0 * np.exp(alpha_td - alpha_t)
    return np.clip(rh, 0.0, 100.0)


def _compute_heat_index_c(temp_c, rh):
    """
    Rothfusz regression in Fahrenheit, converted back to Celsius.
    Uses ambient temperature when outside recommended NWS range.
    """
    t_f = (temp_c * 9.0 / 5.0) + 32.0
    r = np.clip(rh, 0.0, 100.0)

    hi_f = (
        -42.379
        + 2.04901523 * t_f
        + 10.14333127 * r
        - 0.22475541 * t_f * r
        - 6.83783e-3 * (t_f**2)
        - 5.481717e-2 * (r**2)
        + 1.22874e-3 * (t_f**2) * r
        + 8.5282e-4 * t_f * (r**2)
        - 1.99e-6 * (t_f**2) * (r**2)
    )

    # Conservative fallback outside Rothfusz recommended domain.
    fallback_mask = (t_f < 80.0) | (r < 40.0)
    hi_f = np.where(fallback_mask, t_f, hi_f)
    return (hi_f - 32.0) * 5.0 / 9.0


def _humidity_to_rh_percent(
    humidity_seq_norm,
    humidity_mean,
    humidity_std,
    temp_seq_c,
    humidity_source,
):
    humidity_denorm = (humidity_seq_norm * humidity_std) + humidity_mean
    source = (humidity_source or "").lower()

    if source in {"rh2m", "rh", "r", "humidity"}:
        rh = np.array(humidity_denorm, copy=False)
        # Convert [0,1] fraction to percent if needed.
        if np.nanmean(rh) <= 1.5:
            rh = rh * 100.0
        return np.clip(rh, 0.0, 100.0)

    if source in {"d2m", "2m_dewpoint_temperature"}:
        dewpoint_c = np.array(humidity_denorm, copy=False)
        if np.nanmean(dewpoint_c) > 200:
            dewpoint_c = dewpoint_c - 273.15
        return _compute_rh_from_temp_and_dewpoint_c(temp_seq_c, dewpoint_c)

    # Unsupported humidity source (e.g. specific humidity q/ratios without pressure context).
    return None


def _evaluate_event_classifier(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(np.int32)
    metrics = _classification_metrics(y_true, y_pred)
    brier = float(np.mean((y_prob - y_true) ** 2))
    metrics["brier_score"] = float(brier)
    if average_precision_score is not None and np.unique(y_true).size > 1:
        try:
            metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
        except Exception:
            metrics["pr_auc"] = -1.0  # Use -1.0 as sentinel for unavailable
    else:
        metrics["pr_auc"] = -1.0  # Use -1.0 as sentinel for unavailable
    metrics["threshold"] = float(threshold)
    return metrics


def _optimize_probability_threshold(y_true, y_prob):
    """
    Pick a probability threshold that maximizes F1 for imbalanced events.
    Falls back to 0.5 when y_true has no positive samples.
    """
    positives = int(np.sum(y_true == 1))
    if positives == 0:
        return 0.5, {"source": "default_no_positive", "best_f1": 0.0}

    best_threshold = 0.5
    best_metrics = _classification_metrics(y_true, (y_prob >= best_threshold).astype(np.int32))
    best_f1 = float(best_metrics["f1"])
    best_recall = float(best_metrics["recall"])

    # Search lower thresholds too because rare-event models often emit low probabilities.
    for threshold in np.linspace(0.01, 0.99, 99):
        pred = (y_prob >= threshold).astype(np.int32)
        metrics = _classification_metrics(y_true, pred)
        f1 = float(metrics["f1"])
        recall = float(metrics["recall"])
        if (f1 > best_f1) or (abs(f1 - best_f1) < 1e-12 and recall > best_recall):
            best_threshold = float(threshold)
            best_f1 = f1
            best_recall = recall
            best_metrics = metrics

    return best_threshold, {
        "source": "f1_search",
        "best_f1": float(best_f1),
        "best_recall": float(best_recall),
        "tp": int(best_metrics["tp"]),
        "fp": int(best_metrics["fp"]),
        "fn": int(best_metrics["fn"]),
        "tn": int(best_metrics["tn"]),
    }


def _optimize_threshold_with_constraints(y_true, y_prob, min_recall=0.6, max_false_alarm_rate=0.3):
    """Optimize probability threshold with recall and false alarm constraints.
    
    Instead of pure F1 maximization, this finds a threshold that:
    1. Maintains minimum recall (don't miss too many heatwaves)
    2. Keeps false alarm rate below threshold (don't alert too often)
    3. Maximizes F1 among thresholds that satisfy constraints
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        min_recall: Minimum acceptable recall (default 0.6 = catch 60% of events)
        max_false_alarm_rate: Maximum acceptable false alarm rate (default 0.3)
        
    Returns:
        (threshold, info_dict)
    """
    positives = int(np.sum(y_true == 1))
    if positives == 0:
        return 0.5, {"source": "default_no_positive", "constraints_satisfied": False}
    
    # Collect all valid thresholds that meet constraints
    valid_thresholds = []
    
    for threshold in np.linspace(0.01, 0.99, 99):
        pred = (y_prob >= threshold).astype(np.int32)
        metrics = _classification_metrics(y_true, pred)
        
        recall = float(metrics["recall"])
        far = float(metrics["false_alarm_rate"])  # FP / (FP + TN)
        f1 = float(metrics["f1"])
        precision = float(metrics["precision"])
        
        # Check constraints
        recall_ok = recall >= min_recall
        far_ok = far <= max_false_alarm_rate
        
        if recall_ok and far_ok:
            valid_thresholds.append({
                "threshold": float(threshold),
                "f1": f1,
                "recall": recall,
                "precision": precision,
                "false_alarm_rate": far,
                "metrics": metrics,
            })
    
    # If no threshold meets constraints, fall back to best recall with acceptable FAR
    if not valid_thresholds:
        # Find threshold with best recall that keeps FAR reasonable
        best_recall = 0.0
        best_threshold = 0.5
        best_metrics = _classification_metrics(y_true, (y_prob >= 0.5).astype(np.int32))
        
        for threshold in np.linspace(0.01, 0.99, 99):
            pred = (y_prob >= threshold).astype(np.int32)
            metrics = _classification_metrics(y_true, pred)
            recall = float(metrics["recall"])
            far = float(metrics["false_alarm_rate"])
            
            # Prefer higher recall, but cap FAR at slightly higher level
            if recall > best_recall and far <= max_false_alarm_rate * 1.5:
                best_recall = recall
                best_threshold = float(threshold)
                best_metrics = metrics
        
        return best_threshold, {
            "source": "constraint_fallback",
            "best_f1": float(best_metrics["f1"]),
            "best_recall": float(best_metrics["recall"]),
            "constraints_satisfied": False,
            "min_recall": min_recall,
            "max_far": max_false_alarm_rate,
            "tp": int(best_metrics["tp"]),
            "fp": int(best_metrics["fp"]),
        }
    
    # Among valid thresholds, pick the one with best F1
    best = max(valid_thresholds, key=lambda x: x["f1"])
    
    return best["threshold"], {
        "source": "constrained_optimization",
        "best_f1": best["f1"],
        "best_recall": best["recall"],
        "best_precision": best["precision"],
        "false_alarm_rate": best["false_alarm_rate"],
        "constraints_satisfied": True,
        "min_recall": min_recall,
        "max_far": max_false_alarm_rate,
        "tp": int(best["metrics"]["tp"]),
        "fp": int(best["metrics"]["fp"]),
        "fn": int(best["metrics"]["fn"]),
        "tn": int(best["metrics"]["tn"]),
        "num_valid_thresholds": len(valid_thresholds),
    }


def evaluate_baselines(
    x_test,
    y_test,
    future_seq,
    temp_mean,
    temp_std,
    threshold_c,
    clim_temp,
    min_duration=3,
    min_hot_fraction=0.10,
):
    true_temp = (y_test[:, :, 1, :, :] * temp_std) + temp_mean
    last_temp = (x_test[:, -1, 1, :, :] * temp_std) + temp_mean
    if np.nanmean(true_temp) > 200:
        true_temp = true_temp - 273.15
    if np.nanmean(last_temp) > 200:
        last_temp = last_temp - 273.15
    if np.nanmean(clim_temp) > 200:
        clim_temp = clim_temp - 273.15

    persistence_temp = np.repeat(last_temp[:, None, :, :], future_seq, axis=1)
    climatology_temp = np.repeat(clim_temp[None, None, :, :], x_test.shape[0], axis=0)
    climatology_temp = np.repeat(climatology_temp, future_seq, axis=1)

    def _metrics(pred):
        rmse = float(np.sqrt(np.mean((pred - true_temp) ** 2)))
        mae = float(np.mean(np.abs(pred - true_temp)))
        y_true = _to_heatwave_event_labels(
            true_temp,
            threshold_c,
            min_duration=min_duration,
            min_hot_fraction=min_hot_fraction,
        )
        y_pred = _to_heatwave_event_labels(
            pred,
            threshold_c,
            min_duration=min_duration,
            min_hot_fraction=min_hot_fraction,
        )
        event = _classification_metrics(y_true, y_pred)
        return {"rmse": rmse, "mae": mae, "event": event}

    return {
        "persistence": _metrics(persistence_temp),
        "climatology": _metrics(climatology_temp),
    }


def generate_training_report(
    train_metrics,
    val_metrics,
    test_metrics,
    baseline_metrics,
    seasonal_metrics,
    regional_metrics,
    walk_forward_metrics,
    cfg,
    output_dir="output",
):
    """Generate a visual training report after training completes.
    
    Args:
        train_metrics, val_metrics, test_metrics: Performance metrics dicts
        baseline_metrics: Persistence & climatology baselines
        seasonal_metrics, regional_metrics: Stratified metrics
        walk_forward_metrics: Cross-validation results (may be empty)
        cfg: Configuration dict
        output_dir: Directory to save report
        
    Returns:
        Path to saved report image
    """
    import os
    from datetime import datetime
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        HAS_MATPLOTLIB = True
    except ImportError:
        HAS_MATPLOTLIB = False
    
    if not HAS_MATPLOTLIB:
        print("      Warning: matplotlib not available, skipping report generation")
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"training_report_{timestamp}.png")
    
    # Create figure with 2x3 layout
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Heatwave Prediction Model - Training Report', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # Flatten axes for easier indexing
    ax1, ax2, ax3 = axes[0]
    ax4, ax5, ax6 = axes[1]
    
    # Get test metrics
    test_f1 = test_metrics.get('f1', 0)
    test_precision = test_metrics.get('precision', 0)
    test_recall = test_metrics.get('recall', 0)
    test_accuracy = test_metrics.get('accuracy', 0)
    test_pr_auc = test_metrics.get('pr_auc', -1)
    
    # ==== CHART 1: Test Metrics Bar ====
    metrics_names = ['F1', 'Precision', 'Recall', 'Accuracy']
    test_vals = [test_f1, test_precision, test_recall, test_accuracy]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    
    bars = ax1.barh(metrics_names, test_vals, color=colors, alpha=0.8)
    ax1.set_xlim(0, 1.1)
    ax1.set_xlabel('Score')
    ax1.set_title('Test Set Performance', fontweight='bold', fontsize=14)
    ax1.axvline(x=0.7, color='red', linestyle='--', linewidth=2, label='Target (0.70)')
    ax1.legend(loc='lower right')
    
    for bar, val in zip(bars, test_vals):
        ax1.text(val + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.4f}', 
                va='center', fontsize=11, fontweight='bold')
    
    # ==== CHART 2: Confusion Matrix ====
    tp = int(test_metrics.get('tp', 0))
    fp = int(test_metrics.get('fp', 0))
    fn = int(test_metrics.get('fn', 0))
    tn = int(test_metrics.get('tn', 0))
    
    cm = np.array([[tn, fp], [fn, tp]])
    im = ax2.imshow(cm, interpolation='nearest', cmap='Blues')
    ax2.figure.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    
    classes = ['Non-Heatwave', 'Heatwave']
    ax2.set_xticks(np.arange(2))
    ax2.set_yticks(np.arange(2))
    ax2.set_xticklabels(classes)
    ax2.set_yticklabels(classes)
    ax2.set_ylabel('Actual')
    ax2.set_xlabel('Predicted')
    ax2.set_title('Test Confusion Matrix', fontsize=14)
    
    thresh = cm.max() / 2. if cm.max() > 0 else 0.5
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, format(cm[i, j], 'd'),
            ha="center", va="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=16, fontweight='bold')
    
    # ==== CHART 3: Baseline Comparison ====
    if baseline_metrics:
        base_methods = list(baseline_metrics.keys())
        base_f1s = [baseline_metrics[m].get('event', {}).get('f1', 0) for m in base_methods]
        
        # Add model result
        all_methods = ['Model'] + base_methods
        all_f1s = [test_f1] + base_f1s
        colors = ['#27ae60'] + ['#3498db', '#e74c3c'][:len(base_methods)]
        
        bars = ax3.bar(all_methods, all_f1s, color=colors, alpha=0.8)
        ax3.set_ylabel('F1 Score')
        ax3.set_title('Model vs Baselines', fontweight='bold', fontsize=14)
        ax3.set_ylim(0, 1)
        ax3.axhline(y=0.7, color='red', linestyle='--', linewidth=2, label='Target')
        ax3.legend()
        
        for bar, val in zip(bars, all_f1s):
            ax3.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}',
                    ha='center', fontsize=10, fontweight='bold')
    
    # ==== CHART 4: Regional Performance ====
    if regional_metrics:
        regions = list(regional_metrics.keys())
        region_f1s = [regional_metrics[r].get('f1', 0) for r in regions]
        region_prec = [regional_metrics[r].get('precision', 0) for r in regions]
        
        x = np.arange(len(regions))
        width = 0.35
        
        ax4.bar(x - width/2, region_f1s, width, label='F1', color='#2ecc71', alpha=0.8)
        ax4.bar(x + width/2, region_prec, width, label='Precision', color='#3498db', alpha=0.8)
        ax4.set_xticks(x)
        ax4.set_xticklabels(regions, fontsize=10)
        ax4.set_ylabel('Score')
        ax4.set_title('Regional Performance', fontweight='bold', fontsize=14)
        ax4.legend(loc='lower right')
        ax4.set_ylim(0, 1)
    else:
        ax4.axis('off')
        ax4.text(0.5, 0.5, 'No Regional Data', ha='center', va='center', fontsize=14)
    
    # ==== CHART 5: Seasonal Performance ====
    if seasonal_metrics:
        seasons = list(seasonal_metrics.keys())
        season_f1s = [seasonal_metrics[s].get('f1', 0) for s in seasons]
        season_recalls = [seasonal_metrics[s].get('recall', 0) for s in seasons]
        
        x = np.arange(len(seasons))
        width = 0.35
        
        ax5.bar(x - width/2, season_f1s, width, label='F1', color='#e74c3c', alpha=0.8)
        ax5.bar(x + width/2, season_recalls, width, label='Recall', color='#f39c12', alpha=0.8)
        ax5.set_xticks(x)
        ax5.set_xticklabels(seasons, fontsize=10)
        ax5.set_ylabel('Score')
        ax5.set_title('Seasonal Performance', fontweight='bold', fontsize=14)
        ax5.legend(loc='lower right')
        ax5.set_ylim(0, 1)
    else:
        ax5.axis('off')
        ax5.text(0.5, 0.5, 'No Seasonal Data', ha='center', va='center', fontsize=14)
    
    # ==== CHART 6: Summary Text ====
    ax6.axis('off')
    
    # Build summary text
    summary = [
        "MODEL CONFIGURATION",
        "=" * 40,
        f"Model: {cfg.get('model_backend', 'balanced_rf')}",
        f"RF Trees: {cfg.get('rf_n_estimators', 300)}",
        f"RF Depth: {cfg.get('rf_max_depth', 20)}",
        f"Seq Length: {cfg.get('seq_len', 7)}",
        f"Forecast: {cfg.get('future_seq', 2)} days",
        "",
        f"Labeling: {cfg.get('labeling_method', 'temperature')}",
        f"Threshold: {cfg.get('heatwave_temperature_threshold', 35.0)}C",
        "",
        "SPLIT RATIOS",
        "=" * 40,
        f"Train: {cfg.get('train_ratio', 0.7):.0%}",
        f"Val:   {cfg.get('val_ratio', 0.15):.0%}",
        f"Test:  {cfg.get('test_ratio', 0.15):.0%}",
        f"Seed:  {cfg.get('random_seed', 42)}",
        "",
    ]
    
    if walk_forward_metrics and walk_forward_metrics.get('aggregated'):
        wf = walk_forward_metrics['aggregated']
        summary.extend([
            "WALK-FORWARD CV",
            "=" * 40,
            f"Folds: {walk_forward_metrics.get('n_folds', 'N/A')}",
            f"Mean F1: {wf.get('mean_f1', 0):.4f}",
            f"Mean Recall: {wf.get('mean_recall', 0):.4f}",
        ])
    
    ax6.text(0.05, 0.95, '\n'.join(summary), transform=ax6.transAxes,
            fontsize=11, fontfamily='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    # Add recommendations at bottom
    fig.text(0.05, 0.02, 
            f"Performance Summary: F1={test_f1:.4f} | Precision={test_precision:.4f} | Recall={test_recall:.4f}" +
            ("  [GOOD] F1 >= 0.70" if test_f1 >= 0.7 else "  [NEEDS IMPROVEMENT] F1 < 0.70"),
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(report_path, dpi=150, facecolor='white')
    plt.close()
    
    print(f"      Training report saved: {report_path}")
    return report_path


def hyperparameter_search(
    x_train,
    y_train,
    x_val,
    y_val,
    cfg,
    n_iter=20,
    cv_folds=3,
    scoring="f1",
    random_state=42,
):
    """Perform hyperparameter search using RandomizedSearchCV.
    
    Searches for optimal RandomForest hyperparameters while preserving
    temporal ordering through TimeSeriesSplit.
    
    Args:
        x_train, y_train: Training features and labels
        x_val, y_val: Validation features and labels
        cfg: Configuration dict
        n_iter: Number of parameter settings sampled
        cv_folds: Number of cross-validation folds
        scoring: Metric to optimize ('f1', 'precision', 'recall', 'roc_auc')
        random_state: Random seed for reproducibility
        
    Returns:
        Best model and search results dict
    """
    from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
    from imblearn.ensemble import BalancedRandomForestClassifier
    from scipy.stats import randint, uniform
    import time
    
    print(f"\n[{time.strftime('%H:%M:%S')}] Starting hyperparameter search...")
    print(f"      Search space: {n_iter} iterations, {cv_folds}-fold CV")
    print(f"      Optimizing for: {scoring}")
    
    # Parameter distributions for RandomizedSearchCV
    param_distributions = {
        "n_estimators": randint(200, 500),
        "max_depth": randint(15, 40),
        "min_samples_leaf": randint(1, 5),
        "min_samples_split": randint(2, 10),
        "max_features": ["sqrt", "log2", None],
        "sampling_strategy": ["all", "auto"],
        "replacement": [True, False],
        "class_weight": ["balanced", "balanced_subsample", None],
    }
    
    # Use TimeSeriesSplit for temporal data
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    
    # Base model
    base_model = RandomForestClassifier(
        random_state=random_state,
        n_jobs=-1,
    )
    
    # Randomized search
    search = RandomizedSearchCV(
        base_model,
        param_distributions,
        n_iter=n_iter,
        cv=tscv,
        scoring=scoring,
        n_jobs=-1,
        random_state=random_state,
        verbose=1,
        return_train_score=True,
    )
    
    # Fit search
    start_time = time.time()
    search.fit(x_train, y_train)
    search_duration = time.time() - start_time
    
    # Get best model
    best_model = search.best_estimator_
    best_params = search.best_params_
    best_score = search.best_score_
    
    # Evaluate on validation set
    y_pred_proba = best_model.predict_proba(x_val)[:, 1]
    
    # Find optimal threshold
    threshold, threshold_info = _optimize_probability_threshold(y_val, y_pred_proba)
    val_metrics = _evaluate_event_classifier(y_val, y_pred_proba, threshold=threshold)
    
    print(f"\n[{time.strftime('%H:%M:%S')}] Hyperparameter search completed in {search_duration:.1f}s")
    print(f"      Best CV score ({scoring}): {best_score:.4f}")
    print(f"      Validation F1: {val_metrics['f1']:.4f}")
    print(f"      Validation Precision: {val_metrics['precision']:.4f}")
    print(f"      Validation Recall: {val_metrics['recall']:.4f}")
    print(f"      Optimal threshold: {threshold:.3f}")
    
    print("\n      Best parameters:")
    for param, value in best_params.items():
        print(f"        {param}: {value}")
    
    return {
        "model": best_model,
        "best_params": best_params,
        "best_score": best_score,
        "threshold": threshold,
        "threshold_info": threshold_info,
        "val_metrics": val_metrics,
        "search_duration": search_duration,
        "cv_results": search.cv_results_,
    }


def add_engineered_features(x_sequences, cfg):
    """Add engineered features to improve model performance.
    
    Features added:
    - Spatial summary features (hot fraction, regional aggregates, spatial stats)
    - Temporal features (trends, rolling statistics, anomaly from mean)
    - Interaction features (temp x humidity, pressure x temp)
    
    Args:
        x_sequences: Input sequences of shape (N, Time, Channels, H, W)
        cfg: Configuration dict
        
    Returns:
        Flattened features with engineered additions
    """
    if not cfg.get("feature_engineering_enabled", False):
        # Return flattened features without engineering
        n_samples = x_sequences.shape[0]
        return x_sequences.reshape(n_samples, -1)
    
    print("\n[Feature Engineering] Adding engineered features...")
    
    n_samples = x_sequences.shape[0]
    n_time = x_sequences.shape[1]
    n_channels = x_sequences.shape[2]
    h, w = x_sequences.shape[3], x_sequences.shape[4]
    
    enhanced_features = []
    
    # 1. Flatten original features (baseline)
    x_flat = x_sequences.reshape(n_samples, n_time, -1)  # (N, T, C*H*W)
    enhanced_features.append(x_flat.reshape(n_samples, -1))  # Flatten for tree model
    print(f"      Original features: {x_flat.reshape(n_samples, -1).shape}")
    
    # 2. Spatial summary features per time step
    if cfg.get("feature_spatial", True):
        # For temperature channel (typically channel 1)
        temp_channel = min(1, n_channels - 1)
        temp_data = x_sequences[:, :, temp_channel, :, :]  # (N, T, H, W)
        
        # Hot fraction above threshold
        heat_threshold = cfg.get("heatwave_temperature_threshold", 35.0)
        # Denormalize if needed (assumes normalized data)
        # For simple version: use percentile-based threshold
        temp_flat = temp_data.reshape(n_samples, n_time, -1)
        threshold_90 = np.percentile(temp_flat, 90, axis=2, keepdims=True)  # (N, T, 1)
        hot_fraction = (temp_flat > threshold_90).mean(axis=2)  # (N, T)
        
        # Spatial statistics: max, min, mean, std per timestep
        spatial_max = temp_data.max(axis=(2, 3))  # (N, T)
        spatial_min = temp_data.min(axis=(2, 3))  # (N, T)
        spatial_mean = temp_data.mean(axis=(2, 3))  # (N, T)
        spatial_std = temp_data.std(axis=(2, 3))  # (N, T)
        spatial_range = spatial_max - spatial_min  # (N, T)
        
        # Concatenate spatial features
        spatial_features = np.column_stack([
            hot_fraction, spatial_max, spatial_min, spatial_mean, spatial_std, spatial_range
        ])  # (N, T*6)
        
        enhanced_features.append(spatial_features)
        print(f"      Added spatial features: {spatial_features.shape}")
    
    # 3. Temporal trend features
    if cfg.get("feature_temporal", True):
        # Compute trends across time for each spatial position
        # Use simple linear regression slopes
        x_for_trend = x_sequences.reshape(n_samples, n_time, -1)  # (N, T, D)
        
        # Mean across spatial dimensions per timestep
        temporal_means = x_for_trend.mean(axis=2)  # (N, T)
        
        # Compute trend (slope) using last 3 timesteps
        if n_time >= 3:
            # Simple slope: (last - first) / (n_time - 1)
            trend = (temporal_means[:, -1] - temporal_means[:, 0]) / (n_time - 1)  # (N,)
            
            # Acceleration (trend of differences)
            diffs = np.diff(temporal_means, axis=1)  # (N, T-1)
            acceleration = np.mean(diffs[:, -2:], axis=1) if diffs.shape[1] >= 2 else np.zeros(n_samples)
            
            temporal_features = np.column_stack([trend, acceleration])
            enhanced_features.append(temporal_features)
            print(f"      Added temporal trend features: {temporal_features.shape}")
    
    # 4. Interaction features
    if cfg.get("feature_interactions", True) and n_channels >= 5:
        # Interaction: temperature x humidity
        temp_data = x_sequences[:, :, 1, :, :]  # (N, T, H, W)
        humidity_data = x_sequences[:, :, 4, :, :]  # (N, T, H, W)
        
        # Heat index proxy (simplified)
        hi_proxy = temp_data * (1 + 0.01 * humidity_data)  # Simplified heat index proxy
        hi_mean = hi_proxy.mean(axis=(1, 2, 3))  # (N,)
        hi_max = hi_proxy.max(axis=(1, 2, 3))  # (N,)
        
        # Dewpoint depression (temp - dewpoint)
        # Approx: temp - dewpoint ≈ temp - (temp - relative_humidity/5)
        # Simplified interaction
        temp_humidity_product = temp_data.mean(axis=(1, 2)) * humidity_data.mean(axis=(1, 2))  # (N, T)
        
        interaction_features = np.column_stack([hi_mean, hi_max, temp_humidity_product.mean(axis=1)])
        enhanced_features.append(interaction_features)
        print(f"      Added interaction features: {interaction_features.shape}")
    
    # Concatenate all features
    final_features = np.concatenate(enhanced_features, axis=1)
    print(f"      Total engineered features: {final_features.shape}")
    
    return final_features
    
    # Concatenate all features
    all_features = np.concatenate(enhanced_features, axis=1)  # (N, T', D)
    
    # Reshape to (N, T'*D) for ML models
    final_features = all_features.reshape(n_samples, -1)
    
    print(f"      Final feature shape: {final_features.shape}")
    
    return final_features


def walk_forward_cv(
    train_norm,
    val_norm,
    test_norm,
    cfg,
    train_mean,
    train_std,
    temp_mean_scalar,
    temp_std_scalar,
    all_times,
    lats,
    lons,
    n_folds: int = 3,
    expand_window: bool = True,
):
    """Walk-Forward Cross-Validation for time series.
    
    Simulates real-world forecasting: train on expanding window of historical data,
    test on next time step, then expand training window and repeat.
    
    Args:
        train_norm: Training data (normalized)
        val_norm: Validation data (normalized)
        test_norm: Test data (normalized)
        cfg: Config dictionary
        train_mean, train_std: Normalization stats
        temp_mean_scalar, temp_std_scalar: Temperature stats
        all_times: Time index array
        lats, lons: Grid coordinates
        n_folds: Number of walk-forward iterations
        expand_window: If True, expand training window each fold; else use fixed size
        
    Returns:
        dict with fold metrics andaggregated results
    """
    from imblearn.ensemble import BalancedRandomForestClassifier
    
    # Flatten features for ML model
    x_train = _flatten_features(train_norm)
    x_val = _flatten_features(val_norm)
    x_test = _flatten_features(test_norm)
    
    # Build labels for train data only (walk-forward uses train/val boundaries)
    y_train_temp_c = _to_temperature_c(
        create_sequences(train_norm, cfg["seq_len"], cfg["future_seq"])[1],
        temp_mean_scalar,
        temp_std_scalar,
    )
    
    if np.nanmean(y_train_temp_c) > 200:
        y_train_temp_c = y_train_temp_c - 273.15
    
    threshold_c = float(cfg.get("heatwave_temperature_threshold", 35.0))
    event_min_duration = int(cfg.get("event_min_duration_days", 3))
    event_min_hot_fraction = float(cfg.get("event_min_hot_fraction", 0.10))
    
    y_train_event = _to_heatwave_event_labels(
        y_train_temp_c,
        threshold_c,
        min_duration=event_min_duration,
        min_hot_fraction=event_min_hot_fraction,
    )
    
    fold_metrics = []
    total_train_samples = train_norm.shape[0]
    
    # Walk-forward fold size (each fold tests on ~total / n_folds samples)
    test_block_size = max(10, total_train_samples // (n_folds * 2))
    val_block_size = max(5, test_block_size // 3)  # Val block for threshold tuning
    
    for fold_idx in range(n_folds):
        # Determine train end for this fold
        if expand_window:
            # Training window expands: use more data as we go forward
            fold_train_end = total_train_samples - ((n_folds - fold_idx - 1) * test_block_size)
        else:
            # Fixed minimum training window
            fold_train_end = max(cfg["seq_len"] + cfg["future_seq"] + 20, total_train_samples // 2)
        
        fold_val_end = min(fold_train_end + val_block_size, total_train_samples)
        fold_test_end = min(fold_val_end + test_block_size, total_train_samples)
        
        # Skip if not enough data
        if fold_test_end <= fold_val_end or fold_val_end <= fold_train_end:
            continue
        
        # Train model for this fold
        rf_model = RandomForestClassifier(
            n_estimators=int(cfg["rf_n_estimators"]),
            max_depth=int(cfg["rf_max_depth"]),
            min_samples_leaf=int(cfg["rf_min_samples_leaf"]),
            sampling_strategy=cfg["rf_sampling_strategy"],
            replacement=bool(cfg["rf_replacement"]),
            random_state=42,
            n_jobs=-1,
        )
        
        # Use subset up to fold_train_end for training
        rf_model.fit(x_train[:fold_train_end], y_train_event[:fold_train_end])
        
        # Validation set for threshold tuning (NO leakage - val < test)
        y_fold_val = y_train_event[fold_train_end:fold_val_end]
        x_fold_val = x_train[fold_train_end:fold_val_end]
        
        # Test set for final evaluation (untouched for threshold)
        y_fold_test = y_train_event[fold_val_end:fold_test_end]
        x_fold_test = x_train[fold_val_end:fold_test_end]
        
        if len(x_fold_test) == 0 or len(np.unique(y_fold_test)) < 2:
            continue
        if len(x_fold_val) == 0:
            continue
        
        # Predict on val and test
        pred_val_prob = rf_model.predict_proba(x_fold_val)[:, 1]
        pred_test_prob = rf_model.predict_proba(x_fold_test)[:, 1]
        
        # Optimize threshold on validation ONLY (no leakage)
        if np.sum(y_fold_val) > 0:
            prob_threshold, _ = _optimize_probability_threshold(y_fold_val, pred_val_prob)
        else:
            prob_threshold = 0.5
        
        # Evaluate on test (never used for threshold)
        fold_metric = _evaluate_event_classifier(y_fold_test, pred_test_prob, prob_threshold)
        fold_metrics.append({
            "fold": fold_idx + 1,
            "train_samples": fold_train_end,
            "val_samples": len(x_fold_val),
            "test_samples": len(x_fold_test),
            "prob_threshold": prob_threshold,
            "metrics": fold_metric,
        })
    
    # Aggregate results
    if fold_metrics:
        avg_f1 = np.mean([m["metrics"].get("f1", 0) for m in fold_metrics])
        avg_recall = np.mean([m["metrics"].get("recall", 0) for m in fold_metrics])
        avg_precision = np.mean([m["metrics"].get("precision", 0) for m in fold_metrics])
    else:
        avg_f1 = avg_recall = avg_precision = 0.0
    
    return {
        "n_folds": n_folds,
        "expand_window": expand_window,
        "fold_metrics": fold_metrics,
        "aggregated": {
            "mean_f1": avg_f1,
            "mean_recall": avg_recall,
            "mean_precision": avg_precision,
        },
    }


def _sequence_target_times(split_times, seq_len, future_seq):
    split_times = np.asarray(split_times, dtype="datetime64[ns]")
    total_window = int(seq_len) + int(future_seq)
    num_samples = int(split_times.shape[0] - total_window + 1)
    if num_samples <= 0:
        return np.array([], dtype="datetime64[ns]")
    start = int(seq_len)
    end = start + num_samples
    return split_times[start:end]


def _build_data_quality_report(data_norm, stats):
    std = np.asarray(stats.get("std"))
    channel_std = std.reshape(std.shape[1]) if std.ndim == 4 else np.array([])
    near_zero_channels = [int(i) for i, v in enumerate(channel_std) if abs(float(v)) < 1e-7]
    missing_dynamic = list(stats.get("dynamic_missing", []))
    available_dynamic = list(stats.get("dynamic_available", []))
    dynamic_sources = dict(stats.get("dynamic_sources", {})) if isinstance(stats, dict) else {}
    time_index = np.asarray(stats.get("time_index", []), dtype="datetime64[ns]")
    time_span = {}
    if time_index.size > 0:
        time_span = {
            "start": str(time_index.min()),
            "end": str(time_index.max()),
            "num_steps": int(time_index.size),
            "years": sorted(np.unique(time_index.astype("datetime64[Y]").astype(int) + 1970).tolist()),
        }

    return {
        "shape": [int(v) for v in data_norm.shape],
        "dynamic_available": available_dynamic,
        "dynamic_missing": missing_dynamic,
        "dynamic_missing_count": int(len(missing_dynamic)),
        "dynamic_sources": dynamic_sources,
        "near_zero_std_channels": near_zero_channels,
        "time_span": time_span,
    }


def _month_to_season(month):
    # Thailand-focused meteorological seasons.
    if month in (11, 12, 1, 2):
        return "cool_dry"
    if month in (3, 4, 5):
        return "hot_dry"
    return "rainy"


def _evaluate_metrics_by_group(y_true, y_prob, groups, threshold):
    out = {}
    unique_groups = sorted(set(groups.tolist()))
    for group in unique_groups:
        idx = np.where(groups == group)[0]
        if idx.size == 0:
            continue
        metrics = _evaluate_event_classifier(y_true[idx], y_prob[idx], threshold=threshold)
        metrics["count"] = int(idx.size)
        metrics["positive_count"] = int(np.sum(y_true[idx]))
        out[str(group)] = metrics
    return out


def _evaluate_regional_event_metrics(y_true_temp, y_pred_prob, x_test, lats, lons, threshold_c, min_duration, min_hot_fraction, prob_threshold, train_mean, train_std):
    """
    Evaluate metrics per region using spatial predictions.
    
    Args:
        y_true_temp: Temperature array (samples, time, lat, lon) in Celsius
        y_pred_prob: Model probability predictions (samples,)
        x_test: Test features (samples, seq_len, channels, lat, lon) 
        lats, lons: Latitude/longitude arrays
    """
    lat_arr = np.asarray(lats, dtype=np.float32)
    if lat_arr.size < 3 or x_test is None:
        return {}
    
    # Get dimensions
    n_samples = x_test.shape[0]
    n_lats = x_test.shape[3]
    n_lons = x_test.shape[4]
    
    # Reshape predictions to spatial grid (assuming uniform probability across grid per sample)
    # This is an approximation - model predicts one probability per sample, not per grid
    # For regional analysis, we'll use model predictions at each grid point
    pred_spatial = np.full((n_samples, 1, n_lats, n_lons), y_pred_prob[:, np.newaxis, np.newaxis, np.newaxis])
    
    # Compute temperature predictions from features (use mean of last 2 days)
    temp_mean_scalar = float(train_mean[0, 1, 0, 0])
    temp_std_scalar = float(train_std[0, 1, 0, 0])
    last_2_days = x_test[:, -2:, 1, :, :]  # Temperature channel (index 1)
    pred_temp_spatial = (last_2_days.mean(axis=1) * temp_std_scalar) + temp_mean_scalar
    
    # If temp looks like Kelvin, convert to Celsius
    if np.nanmean(pred_temp_spatial) > 200:
        pred_temp_spatial = pred_temp_spatial - 273.15
    
    # Reshape y_true_temp if needed
    if len(y_true_temp.shape) == 2:
        # It's flattened, reshape to (samples, time, lat, lon)
        y_true_spatial = y_true_temp.reshape(n_samples, -1, n_lats, n_lons)
    else:
        y_true_spatial = y_true_temp
    
    if np.nanmean(y_true_spatial) > 200:
        y_true_spatial = y_true_spatial - 273.15
    
    # Define regions based on latitude
    grid_lats = np.linspace(lat_arr[0], lat_arr[-1], n_lats)
    q1 = float(np.percentile(grid_lats, 33.3))
    q2 = float(np.percentile(grid_lats, 66.7))
    
    region_masks = {
        "south": np.where(grid_lats <= q1)[0],
        "central": np.where((grid_lats > q1) & (grid_lats <= q2))[0],
        "north": np.where(grid_lats > q2)[0],
    }
    
    out = {}
    for region, lat_idx in region_masks.items():
        if lat_idx.size == 0:
            continue
            
        # Extract region data - x_test shape: (samples, seq_len, channels, lat, lon)
        region_x = x_test[:, :, :, lat_idx, :]  # (samples, seq_len, channels, lat_idx, lon)
        
        # Get temperature from channel 1
        region_temp = (region_x[:, -1, 1, :, :] * temp_std_scalar) + temp_mean_scalar  # (samples, lat, lon)
        
        if np.nanmean(region_temp) > 200:
            region_temp = region_temp - 273.15
        
        # For true labels, compute from model predictions (or use x_test last temp)
        # Use same approach for both to ensure consistency
        true_temp_region = region_temp.copy()
        
        # For predictions, we use model probability - expand to region
        region_pred_prob = pred_spatial[:, :, lat_idx, :]  # (samples, 1, lat, lon)
        
        # Convert to event labels from temperature
        true_events = _to_heatwave_event_labels(
            true_temp_region[:, np.newaxis, :, :],  # Add time dim
            threshold_c,
            min_duration=min_duration,
            min_hot_fraction=min_hot_fraction,
        )
        
        # For predictions, use probability threshold
        pred_events = (region_pred_prob.squeeze() > prob_threshold).astype(int)
        
        # Aggregate to sample level
        true_sample = (true_events.max(axis=(1,2)) > 0).astype(int) if len(true_events.shape) > 1 else (true_events > 0).astype(int)
        pred_sample = (pred_events.max(axis=(1,2)) > 0).astype(int) if len(pred_events.shape) > 1 else (pred_events > 0).astype(int)
        
        # Ensure 1D arrays
        true_sample = np.asarray(true_sample).flatten()
        pred_sample = np.asarray(pred_sample).flatten()
        
        # Compute metrics
        metrics = _classification_metrics(true_sample, pred_sample)
        
        metrics["positive_count"] = int(np.sum(true_sample))
        metrics["sample_count"] = int(len(true_sample))
        
        out[region] = metrics
        
    return out


def train(config=None, on_epoch_end: Optional[Callable[[EpochMetrics], None]] = None,
          on_progress: Optional[Callable[[str, float, Optional[Dict[str, Any]]], None]] = None,
          on_log: Optional[Callable[[str, str], None]] = None):
    """Train heatwave prediction model with progress callbacks.

    Args:
        config: Optional config dict to override defaults
        on_epoch_end: Callback for epoch metrics
        on_progress: Callback for progress updates (stage, progress, metadata)
        on_log: Callback for log messages (level, message)

    Returns:
        Training results or None if failed
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

    cfg = {
        "batch_size": BATCH_SIZE,
        "seq_len": SEQ_LEN,
        "future_seq": FUTURE_SEQ,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
        "heatwave_percentile": HEATWAVE_PERCENTILE,
        "rf_n_estimators": RF_N_ESTIMATORS,
        "rf_max_depth": RF_MAX_DEPTH,
        "rf_min_samples_leaf": RF_MIN_SAMPLES_LEAF,
        "rf_sampling_strategy": RF_SAMPLING_STRATEGY,
        "rf_replacement": RF_REPLACEMENT,
        "clip_low_percentile": CLIP_LOW_PERCENTILE,
        "clip_high_percentile": CLIP_HIGH_PERCENTILE,
        "model_backend": MODEL_BACKEND,
        "use_xgboost": USE_XGBOOST,
        "use_gpu": USE_GPU,
        "force_gpu": FORCE_GPU,
        "event_min_duration_days": EVENT_MIN_DURATION_DAYS,
        "event_min_hot_fraction": EVENT_MIN_HOT_FRACTION,
        "allow_sample_mean_fallback": ALLOW_SAMPLE_MEAN_FALLBACK,
        "require_dynamic_features": REQUIRE_DYNAMIC_FEATURES,
        "min_train_positive_rate": MIN_TRAIN_POSITIVE_RATE,
        "max_train_positive_rate": MAX_TRAIN_POSITIVE_RATE,
        "min_eval_positive_count": MIN_EVAL_POSITIVE_COUNT,
        "labeling_method": LABELING_METHOD,
        "heatwave_heat_index_threshold": HEATWAVE_HEAT_INDEX_THRESHOLD,
        "heatwave_temperature_threshold": HEATWAVE_TEMPERATURE_THRESHOLD,
        "heatwave_anomaly_threshold": HEATWAVE_ANOMALY_THRESHOLD,
        "random_seed": RANDOM_SEED,
        "walk_forward_enabled": WALK_FORWARD_ENABLED,
        "walk_forward_n_folds": WALK_FORWARD_N_FOLDS,
        "walk_forward_expand_window": WALK_FORWARD_EXPAND_WINDOW,
        "hp_tuning_enabled": HP_TUNING_ENABLED,
        "hp_n_iter": HP_N_ITER,
        "hp_cv_folds": HP_CV_FOLDS,
        "hp_scoring": HP_SCORING,
        "feature_engineering_enabled": FEATURE_ENGINEERING_ENABLED,
        "feature_interactions": FEATURE_INTERACTIONS,
        "feature_temporal": FEATURE_TEMPORAL,
    }
    if config:
        cfg.update(config)

    log("info", "Starting Heatwave AI Training...")
    progress("init", 0.0, {"stage": "initialization"})
    random_seed = int(cfg.get("random_seed", 42))
    _set_random_seeds(random_seed)
    log("info", f"Random seed set to {random_seed} for reproducibility")
    log("info", f"Config: seq={cfg['seq_len']}, future={cfg['future_seq']}, "
         f"rf_trees={cfg['rf_n_estimators']}, depth={cfg['rf_max_depth']}, "
         f"backend={cfg['model_backend']}, gpu={cfg['use_gpu']}, "
         f"force_gpu={cfg['force_gpu']}, "
         f"labeling={cfg['labeling_method']}, "
         f"sampling={cfg['rf_sampling_strategy']}, "
         f"clip=({cfg['clip_low_percentile']},{cfg['clip_high_percentile']})")

    log("info", "\n[1/6] Loading ERA5 Data using new DataLoader...")
    progress("loading", 0.1, {"stage": "data_loading"})
    loader = DataLoader()
    try:
        # Load the dataset lazily - use combined ERA5 + NASA POWER if enabled
        if USE_NASA_POWER:
            log("info", "      Loading combined ERA5 + NASA POWER data...")
            full_ds = loader.load_combined()
        else:
            full_ds = loader.load_era5()
        # Load raw (un-normalized) data — normalization and NaN filling happen after temporal split
        # to prevent data leakage (no future data used to fill past NaN values)
        data_norm, stats = loader.prepare_training_data(full_ds, fill_nan=False)
        lats, lons = stats["lats"], stats["lons"]
        train_mean, train_std = None, None  # computed post-split to avoid leakage
        all_times = np.asarray(stats.get("time_index", []), dtype="datetime64[ns]")

        # Report available variables
        log("info", f"      Available dynamic variables: {stats.get('dynamic_available', [])}")
        log("info", f"      Missing dynamic variables: {stats.get('dynamic_missing', [])}")
    except Exception as exc:
        log("error", f"Error loading data with new DataLoader: {exc}")
        return None

    quality_report = _build_data_quality_report(data_norm, stats)
    log("info", f"      Data quality: missing_dynamic={quality_report['dynamic_missing_count']} "
         f"near_zero_std_channels={len(quality_report['near_zero_std_channels'])}")
    if quality_report["dynamic_missing_count"] > 0:
        log("info", f"      Missing dynamics: {quality_report['dynamic_missing']}")
    labeling_method = str(cfg.get("labeling_method", "heat_index")).strip().lower()
    required_by_method = {
        "heat_index": {"t2m", "humidity"},
        "temperature": {"t2m"},
    }
    required_dynamic = required_by_method.get(labeling_method, {"t2m"})
    missing_required = sorted(required_dynamic.intersection(set(quality_report["dynamic_missing"])))
    if cfg.get("require_dynamic_features", True) and missing_required:
        raise ValueError(
            "Training blocked by strict data-quality gate: missing required dynamic variables "
            f"{missing_required} for labeling_method='{labeling_method}'. "
            "Provide these variables in ERA5 input or disable strict gate via HW_REQUIRE_DYNAMIC_FEATURES=0."
        )

    log("info", f"      Normalized Data Shape: {data_norm.shape}")

    log("info", "\n[2/6] Temporal split (train/val/test)...")
    progress("splitting", 0.2, {"stage": "data_splitting"})
    test_ratio = cfg.get("test_ratio", 0.15)
    try:
        # Split BEFORE any interpolation to prevent data leakage
        # (NaN values in train should not be filled using future val/test information)
        train_norm, val_norm, test_norm = temporal_split_data(
            data_norm,
            train_ratio=cfg["train_ratio"],
            val_ratio=cfg["val_ratio"],
            test_ratio=test_ratio,
        )
        if all_times.size > 0:
            train_times, val_times, test_times = temporal_split_data(
                all_times,
                train_ratio=cfg["train_ratio"],
                val_ratio=cfg["val_ratio"],
                test_ratio=test_ratio,
            )
        else:
            train_times = np.array([], dtype="datetime64[ns]")
            val_times = np.array([], dtype="datetime64[ns]")
            test_times = np.array([], dtype="datetime64[ns]")
    except Exception as exc:
        log("error", f"Split failed: {exc}")
        return None

    # Fill NaN values SEPARATELY for each partition (no leakage)
    log("info", "      Filling NaN values within each split (causal)...")
    for channel_idx in range(train_norm.shape[1]):
        train_norm[:, channel_idx, :, :] = fill_nan_along_time(train_norm[:, channel_idx, :, :])
        val_norm[:, channel_idx, :, :] = fill_nan_along_time(val_norm[:, channel_idx, :, :])
        test_norm[:, channel_idx, :, :] = fill_nan_along_time(test_norm[:, channel_idx, :, :])

    # Log split information for reproducibility
    total_samples = len(data_norm)
    log("info", f"      Split config: train={cfg['train_ratio']:.0%}, "
         f"val={cfg['val_ratio']:.0%}, test={test_ratio:.0%} | "
        f"Total timesteps: {total_samples}"
    )
    log("info", f"      Sample counts: {train_norm.shape[0]}/{val_norm.shape[0]}/{test_norm.shape[0]}")

    # Compute normalization stats from training split only (no leakage)
    _EPSILON = 1e-8
    train_mean = train_norm.mean(axis=(0, 2, 3), keepdims=True)
    train_std = train_norm.std(axis=(0, 2, 3), keepdims=True)
    train_std = np.where(train_std < _EPSILON, _EPSILON, train_std)
    
    # Normalize
    train_norm = (train_norm - train_mean) / train_std
    val_norm = (val_norm - train_mean) / train_std
    test_norm = (test_norm - train_mean) / train_std
    
    # Apply clipping to remove outliers
    clip_low_percentile = float(cfg.get("clip_low_percentile", 0.5))
    clip_high_percentile = float(cfg.get("clip_high_percentile", 99.5))
    clip_lower = np.percentile(train_norm, clip_low_percentile, axis=(0, 2, 3), keepdims=True)
    clip_upper = np.percentile(train_norm, clip_high_percentile, axis=(0, 2, 3), keepdims=True)
    clip_bounds = (clip_lower, clip_upper)
    
    print(f"      Applying clipping: [{clip_low_percentile}%, {clip_high_percentile}%] percentiles")
    train_norm = np.clip(train_norm, clip_lower, clip_upper)
    val_norm = np.clip(val_norm, clip_lower, clip_upper)
    test_norm = np.clip(test_norm, clip_lower, clip_upper)
    
    print(f"      Normalization computed from {train_norm.shape[0]} training timesteps.")
    x_train, y_train = create_sequences(train_norm, cfg["seq_len"], cfg["future_seq"])
    x_val, y_val = create_sequences(val_norm, cfg["seq_len"], cfg["future_seq"])
    x_test, y_test = create_sequences(test_norm, cfg["seq_len"], cfg["future_seq"])

    if len(x_train) == 0 or len(x_val) == 0 or len(x_test) == 0:
        print(
            "No sequences in one or more partitions. Adjust split or sequence lengths."
        )
        return None

    x_train_flat = _flatten_features(x_train)
    x_val_flat = _flatten_features(x_val)
    x_test_flat = _flatten_features(x_test)

    train_target_times = _sequence_target_times(train_times, cfg["seq_len"], cfg["future_seq"])
    val_target_times = _sequence_target_times(val_times, cfg["seq_len"], cfg["future_seq"])
    test_target_times = _sequence_target_times(test_times, cfg["seq_len"], cfg["future_seq"])

    print(f"      Samples train/val/test: {len(x_train)}/{len(x_val)}/{len(x_test)}")

    temp_mean_scalar = float(train_mean[0, 1, 0, 0])
    temp_std_scalar = float(train_std[0, 1, 0, 0])
    # clip_bounds already computed during normalization step above

    # Reconstruct temperatures in Celsius for thresholding
    y_train_temp = _to_temperature_c(y_train, temp_mean_scalar, temp_std_scalar)
    y_val_temp = _to_temperature_c(y_val, temp_mean_scalar, temp_std_scalar)
    y_test_temp = _to_temperature_c(y_test, temp_mean_scalar, temp_std_scalar)

    # Convert train temperature time-series for climatology baseline.
    train_temp_for_threshold = (train_norm[:, 1, :, :] * temp_std_scalar) + temp_mean_scalar
    if np.nanmean(train_temp_for_threshold) > 200:
        train_temp_for_threshold = train_temp_for_threshold - 273.15
        y_train_temp = y_train_temp - 273.15
        y_val_temp = y_val_temp - 273.15
        y_test_temp = y_test_temp - 273.15
        temp_mean_scalar = temp_mean_scalar - 273.15
    train_climatology_temp = train_temp_for_threshold.mean(axis=0)

    threshold_c = None
    y_train_event = None
    y_val_event = None
    y_test_event = None
    threshold_selection_mode = f"fixed_{labeling_method}"
    event_min_duration = max(1, int(cfg.get("event_min_duration_days", 3)))
    event_min_hot_fraction = float(cfg.get("event_min_hot_fraction", 0.10))

    event_train_seq = y_train_temp
    event_val_seq = y_val_temp
    event_test_seq = y_test_temp

    # Compute climatology for anomaly-based labeling
    # train_climatology_temp is already computed above (line ~1557)
    # It should have shape (H, W) but may have extra dimensions
    
    # Flatten climatology to 2D (H, W)
    if train_climatology_temp.ndim == 3:
        # Shape (C, H, W) - take channel 1 (temperature)
        train_climatology_temp = train_climatology_temp[1]
    elif train_climatology_temp.ndim == 4:
        # Shape (1, 1, H, W) or similar - take last two dims
        train_climatology_temp = train_climatology_temp[0, 0, :, :] if train_climatology_temp.shape[0] == 1 else train_climatology_temp[:, :, -1, :]
    
    anomaly_threshold = float(cfg.get("heatwave_anomaly_threshold", 5.0))
    
    if labeling_method == "anomaly":
        # Use temperature anomaly (deviation from climatology) instead of absolute temperature
        # This solves the problem where hot regions would be flagged everyday
        print(f"      Using anomaly-based labeling (threshold: {anomaly_threshold} degrees above normal)")
        
        # Make sure y_train_temp and train_climatology_temp are in same units
        # If temperature is in Kelvin (>200), convert to Celsius
        if np.nanmean(y_train_temp) > 200:
            y_train_temp = y_train_temp - 273.15
            y_val_temp = y_val_temp - 273.15
            y_test_temp = y_test_temp - 273.15
            if np.nanmean(train_climatology_temp) > 200:
                train_climatology_temp = train_climatology_temp - 273.15
        
        # Compute anomaly in Celsius
        anomaly_train = y_train_temp - train_climatology_temp[np.newaxis, np.newaxis, :, :]
        anomaly_val = y_val_temp - train_climatology_temp[np.newaxis, np.newaxis, :, :]
        anomaly_test = y_test_temp - train_climatology_temp[np.newaxis, np.newaxis, :, :]
        
        print(f"      [Debug] y_train_temp: {np.nanmean(y_train_temp):.2f}C, clim: {np.nanmean(train_climatology_temp):.2f}C")
        
        # Get mean anomaly per sample
        train_anomaly_mean = anomaly_train.mean(axis=(1,2,3))
        val_anomaly_mean = anomaly_val.mean(axis=(1,2,3))
        test_anomaly_mean = anomaly_test.mean(axis=(1,2,3))
        
        # Use percentile threshold - top 15% as heatwave
        anomaly_percentile = 85
        threshold_val = np.percentile(train_anomaly_mean, anomaly_percentile)
        
        print(f"      Anomaly config: threshold={threshold_val:.2f}C (top {100-anomaly_percentile}%)")
        
        y_train_event = (train_anomaly_mean >= threshold_val).astype(np.int32)
        y_val_event = (val_anomaly_mean >= threshold_val).astype(np.int32)
        y_test_event = (test_anomaly_mean >= threshold_val).astype(np.int32)
        
        print(f"      [Anomaly Label] train: {y_train_event.sum()}/{len(y_train_event)} ({y_train_event.mean()*100:.1f}%), val: {y_val_event.sum()}/{len(y_val_event)}, test: {y_test_event.sum()}/{len(y_test_event)}")
        
        threshold_c = threshold_val
        threshold_selection_mode = f"anomaly_{anomaly_percentile}pct"
        
    elif labeling_method == "heat_index":
        if y_train.shape[2] <= 4:
            raise ValueError(
                "Heat-index labeling requires humidity/dewpoint channel at index 4."
            )

        humidity_source = (
            (stats.get("dynamic_sources", {}) or {}).get("humidity")
            if isinstance(stats, dict)
            else None
        )
        humidity_mean_scalar = float(train_mean[0, 4, 0, 0])
        humidity_std_scalar = float(train_std[0, 4, 0, 0])

        y_train_rh = _humidity_to_rh_percent(
            y_train[:, :, 4, :, :],
            humidity_mean_scalar,
            humidity_std_scalar,
            y_train_temp,
            humidity_source,
        )
        y_val_rh = _humidity_to_rh_percent(
            y_val[:, :, 4, :, :],
            humidity_mean_scalar,
            humidity_std_scalar,
            y_val_temp,
            humidity_source,
        )
        y_test_rh = _humidity_to_rh_percent(
            y_test[:, :, 4, :, :],
            humidity_mean_scalar,
            humidity_std_scalar,
            y_test_temp,
            humidity_source,
        )
        if y_train_rh is None or y_val_rh is None or y_test_rh is None:
            raise ValueError(
                "Heat-index labeling requires humidity source RH or dewpoint (d2m). "
                f"Found unsupported source: {humidity_source!r}"
            )

        event_train_seq = _compute_heat_index_c(y_train_temp, y_train_rh)
        event_val_seq = _compute_heat_index_c(y_val_temp, y_val_rh)
        event_test_seq = _compute_heat_index_c(y_test_temp, y_test_rh)
        threshold_c = float(cfg.get("heatwave_heat_index_threshold", 41.0))
    else:
        threshold_c = float(cfg.get("heatwave_temperature_threshold", 35.0))

    y_train_event = _to_heatwave_event_labels(
        event_train_seq,
        threshold_c,
        min_duration=event_min_duration,
        min_hot_fraction=event_min_hot_fraction,
    )
    y_val_event = _to_heatwave_event_labels(
        event_val_seq,
        threshold_c,
        min_duration=event_min_duration,
        min_hot_fraction=event_min_hot_fraction,
    )
    y_test_event = _to_heatwave_event_labels(
        event_test_seq,
        threshold_c,
        min_duration=event_min_duration,
        min_hot_fraction=event_min_hot_fraction,
    )

    if np.unique(y_train_event).size < 2:
        if not bool(cfg.get("allow_sample_mean_fallback", False)):
            raise ValueError(
                "Event labels could not be derived from physical heatwave thresholds and "
                "sample-mean fallback is disabled. Set HW_ALLOW_SAMPLE_MEAN_FALLBACK=1 "
                "only for exploratory runs."
            )
        train_scores = event_train_seq.mean(axis=(1, 2, 3))
        val_scores = event_val_seq.mean(axis=(1, 2, 3))
        test_scores = event_test_seq.mean(axis=(1, 2, 3))

        def _labels_from_scores(scores, threshold=None):
            labels = np.zeros(scores.shape[0], dtype=np.int32)
            if scores.shape[0] <= 1:
                return labels
            if threshold is not None:
                labels = (scores >= threshold).astype(np.int32)
            if np.unique(labels).size < 2:
                topk = max(1, min(scores.shape[0] - 1, int(round(scores.shape[0] * 0.1))))
                idx = np.argpartition(scores, -topk)[-topk:]
                labels[:] = 0
                labels[idx] = 1
            return labels

        score_threshold = float(np.percentile(train_scores, 90.0))
        y_train_event = _labels_from_scores(train_scores, threshold=score_threshold)
        y_val_event = _labels_from_scores(val_scores, threshold=score_threshold)
        y_test_event = _labels_from_scores(test_scores, threshold=score_threshold)
        threshold_c = float(np.percentile(train_scores, 90.0))
        threshold_selection_mode = f"{labeling_method}_sample_mean_fallback"
        print(
            "Unable to derive both classes from event thresholds. "
            "Falling back to sample-mean based labels (top 10% hottest samples)."
        )

    print(
        f"      Heatwave threshold (selected): {threshold_c:.2f} C"
    )
    print(f"      Threshold selection mode: {threshold_selection_mode}")
    print(
        f"      Positive rate train/val/test: "
        f"{y_train_event.mean():.3f}/{y_val_event.mean():.3f}/{y_test_event.mean():.3f}"
    )
    train_pos_rate = float(y_train_event.mean())
    if train_pos_rate < float(cfg.get("min_train_positive_rate", 0.01)):
        raise ValueError(
            f"Train positive rate too low ({train_pos_rate:.4f}) for robust learning. "
            f"Minimum required={float(cfg.get('min_train_positive_rate', 0.01)):.4f}."
        )
    if train_pos_rate > float(cfg.get("max_train_positive_rate", 0.35)):
        raise ValueError(
            f"Train positive rate too high ({train_pos_rate:.4f}); label threshold likely too loose. "
            f"Maximum allowed={float(cfg.get('max_train_positive_rate', 0.35)):.4f}."
        )
    min_eval_positive_count = int(cfg.get("min_eval_positive_count", 10))
    if int(np.sum(y_val_event)) < min_eval_positive_count or int(np.sum(y_test_event)) < min_eval_positive_count:
        raise ValueError(
            "Validation/test positive events are too few for reliable accuracy estimation. "
            f"Require at least {min_eval_positive_count} positives in each split."
        )
    if int(np.sum(y_val_event)) == 0 or int(np.sum(y_test_event)) == 0:
        print(
            "      Warning: validation/test has zero positive events. "
            "F1 on those splits will be 0 by definition."
        )

    model_type = "balanced_random_forest"
    backend = "balanced_rf"
    print("\n[4/6] Initializing BalancedRandomForestClassifier (CPU)...")
    
    # Hyperparameter tuning if enabled
    hp_results = None
    if cfg.get("hp_tuning_enabled", False):
        print("\n[HP Tuning] Running hyperparameter optimization...")
        hp_results = hyperparameter_search(
            x_train=x_train_flat,
            y_train=y_train_event,
            x_val=x_val_flat,
            y_val=y_val_event,
            cfg=cfg,
            n_iter=int(cfg.get("hp_n_iter", 20)),
            cv_folds=int(cfg.get("hp_cv_folds", 3)),
            scoring=cfg.get("hp_scoring", "f1"),
            random_state=random_seed,
        )
        rf_model = hp_results["model"]
        prob_threshold = hp_results["threshold"]
        threshold_source = "hp_tuning_optimized"
        threshold_tuning = hp_results["threshold_info"]
    else:
        # Use default parameters (standard RandomForestClassifier)
        use_xgboost = cfg.get("use_xgboost", False)
        use_lightgbm = cfg.get("use_lightgbm", False)
        
        if use_xgboost and XGBOOST_AVAILABLE:
            print("      Using XGBoost classifier...")
            n_neg = np.sum(y_train_event == 0)
            n_pos = np.sum(y_train_event == 1)
            scale_pos_weight = n_neg / max(n_pos, 1)
            
            # Smaller parameters to fit in memory
            rf_model = xgb.XGBClassifier(
                n_estimators=50,  # Very small
                max_depth=8,      # Shallow
                max_bin=128,      # Reduce memory
                learning_rate=0.1,
                scale_pos_weight=scale_pos_weight,
                tree_method='hist',
                random_state=42,
                n_jobs=4,  # Limit parallel jobs
                verbosity=0,
            )
        elif use_lightgbm and LIGHTGBM_AVAILABLE:
            print("      Using LightGBM classifier...")
            # Calculate is_unbalance for class imbalance
            n_neg = np.sum(y_train_event == 0)
            n_pos = np.sum(y_train_event == 1)
            
            rf_model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=20,
                learning_rate=0.1,
                is_unbalance=True,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
        else:
            rf_model = RandomForestClassifier(
                n_estimators=int(cfg["rf_n_estimators"]),
                max_depth=int(cfg["rf_max_depth"]),
                min_samples_leaf=int(cfg["rf_min_samples_leaf"]),
                random_state=42,
                n_jobs=-1,
                class_weight='balanced',  # Handle class imbalance
            )

    print("\n[5/6] Training + validation metrics...")
    start_time = time.time()
    
    # Only train if HP tuning wasn't run (it already trains the model)
    if hp_results is None:
        rf_model.fit(x_train_flat, y_train_event)
        pred_train_prob = rf_model.predict_proba(x_train_flat)[:, 1]
        pred_val_prob = rf_model.predict_proba(x_val_flat)[:, 1]
        threshold_source = "default_0.5"
        threshold_tuning = {"source": threshold_source, "best_f1": 0.0}
        prob_threshold = 0.5
        if np.sum(y_val_event) > 0:
            prob_threshold, threshold_tuning = _optimize_probability_threshold(
                y_val_event, pred_val_prob
            )
            threshold_source = "validation_f1_search"
        elif np.sum(y_train_event) > 0:
            prob_threshold, threshold_tuning = _optimize_probability_threshold(
                y_train_event, pred_train_prob
            )
            threshold_source = "train_f1_search_fallback"
    else:
        # Use results from HP tuning
        pred_train_prob = rf_model.predict_proba(x_train_flat)[:, 1]
        pred_val_prob = rf_model.predict_proba(x_val_flat)[:, 1]
        # Threshold already optimized in HP tuning

    train_metrics = _evaluate_event_classifier(
        y_train_event, pred_train_prob, threshold=prob_threshold
    )
    val_metrics = _evaluate_event_classifier(
        y_val_event, pred_val_prob, threshold=prob_threshold
    )
    train_loss = train_metrics["brier_score"]

    print(
        f"      Prob Threshold: {prob_threshold:.3f} ({threshold_source}) | "
        f"      Train Brier: {train_metrics['brier_score']:.6f} | "
        f"Val Brier: {val_metrics['brier_score']:.6f} | "
        f"Val PR-AUC: {val_metrics['pr_auc'] if val_metrics['pr_auc'] >= 0 else 'NA'} | "
        f"Val Event F1: {val_metrics['f1']:.4f} | "
        f"Val Recall: {val_metrics['recall']:.4f}"
    )

    if on_epoch_end is not None:
        on_epoch_end(
            EpochMetrics(
                epoch=1,
                total_epochs=1,
                train_loss=train_loss,
                val_loss=val_metrics["brier_score"],
                val_rmse=val_metrics["brier_score"],
                val_event_f1=val_metrics["f1"],
                elapsed_seconds=time.time() - start_time,
            )
        )

    print("\n[6/6] Test evaluation + baseline comparison + checkpoint save...")
    
    # Optional Walk-Forward Cross-Validation
    walk_forward_metrics = {}
    if cfg.get("walk_forward_enabled", False):
        print("Running Walk-Forward Cross-Validation...")
        wf_n_folds = int(cfg.get("walk_forward_n_folds", 3))
        wf_expand = cfg.get("walk_forward_expand_window", True)
        walk_forward_metrics = walk_forward_cv(
            train_norm=train_norm,
            val_norm=val_norm,
            test_norm=test_norm,
            cfg=cfg,
            train_mean=train_mean,
            train_std=train_std,
            temp_mean_scalar=temp_mean_scalar,
            temp_std_scalar=temp_std_scalar,
            all_times=all_times,
            lats=lats,
            lons=lons,
            n_folds=wf_n_folds,
            expand_window=wf_expand,
        )
        if walk_forward_metrics.get("aggregated"):
            agg = walk_forward_metrics["aggregated"]
            print(
                f"      WF-CV Mean F1: {agg.get('mean_f1', 0):.4f} | "
                f"Recall: {agg.get('mean_recall', 0):.4f} | "
                f"Precision: {agg.get('mean_precision', 0):.4f}"
            )
    
    pred_test_prob = rf_model.predict_proba(x_test_flat)[:, 1]
    test_metrics = _evaluate_event_classifier(
        y_test_event, pred_test_prob, threshold=prob_threshold
    )
    seasonal_metrics = {}
    monthly_metrics = {}
    if test_target_times.size == y_test_event.shape[0] and test_target_times.size > 0:
        test_months = (test_target_times.astype("datetime64[M]").astype(int) % 12) + 1
        month_labels = np.array([f"{int(m):02d}" for m in test_months], dtype=object)
        season_labels = np.array([_month_to_season(int(m)) for m in test_months], dtype=object)
        monthly_metrics = _evaluate_metrics_by_group(
            y_test_event, pred_test_prob, month_labels, threshold=prob_threshold
        )
        seasonal_metrics = _evaluate_metrics_by_group(
            y_test_event, pred_test_prob, season_labels, threshold=prob_threshold
        )
    regional_metrics = _evaluate_regional_event_metrics(
        y_true_temp=event_test_seq,
        y_pred_prob=pred_test_prob,
        x_test=x_test,
        lats=lats,
        lons=lons,
        threshold_c=threshold_c,
        min_duration=event_min_duration,
        min_hot_fraction=event_min_hot_fraction,
        prob_threshold=prob_threshold,
        train_mean=train_mean,
        train_std=train_std,
    )
    baseline_metrics = evaluate_baselines(
        x_test=x_test,
        y_test=y_test,
        future_seq=cfg["future_seq"],
        temp_mean=temp_mean_scalar,
        temp_std=temp_std_scalar,
        threshold_c=threshold_c,
        clim_temp=train_climatology_temp,
        min_duration=event_min_duration,
        min_hot_fraction=event_min_hot_fraction,
    )

    duration = time.time() - start_time
    print(f"Training completed in {duration:.2f} seconds.")
    print(
        f"Test Brier: {test_metrics['brier_score']:.6f} | "
        f"PR-AUC: {test_metrics['pr_auc'] if test_metrics['pr_auc'] >= 0 else 'NA'} | "
        f"Event F1: {test_metrics['f1']:.4f} | "
        f"Hit Rate: {test_metrics['hit_rate']:.4f} | "
        f"False Alarm: {test_metrics['false_alarm_rate']:.4f}"
    )
    if seasonal_metrics:
        seasonal_f1 = {
            k: (v.get("f1") if isinstance(v, dict) else None)
            for k, v in seasonal_metrics.items()
        }
        print(f"Seasonal F1: {seasonal_f1}")
    if regional_metrics:
        regional_f1 = {
            k: (v.get("f1") if isinstance(v, dict) else None)
            for k, v in regional_metrics.items()
        }
        print(f"Regional F1: {regional_f1}")

    version = get_next_version(MODELS_DIR)
    save_filename = f"heatwave_model_checkpoint_v{version}.pth"
    save_path = os.path.join(MODELS_DIR, save_filename)

    checkpoint = {
        "model_type": model_type,
        "sklearn_model": rf_model,
        "metadata": {
            "task_type": "heatwave_event_classification",
            "seq_len": cfg["seq_len"],
            "future_seq": cfg["future_seq"],
            "input_dim": int(x_train.shape[2]),
            "normalization_mean": train_mean.tolist(),
            "normalization_std": train_std.tolist(),
            "clip_lower": clip_bounds[0].tolist(),
            "clip_upper": clip_bounds[1].tolist(),
            "train_ratio": cfg["train_ratio"],
            "val_ratio": cfg["val_ratio"],
            "labeling_method": labeling_method,
            "heatwave_heat_index_threshold": cfg.get("heatwave_heat_index_threshold"),
            "heatwave_temperature_threshold": cfg.get("heatwave_temperature_threshold"),
            "heatwave_percentile": cfg["heatwave_percentile"],
            "heatwave_threshold_c": threshold_c,
            "threshold_selection_mode": threshold_selection_mode,
            "event_probability_threshold": prob_threshold,
            "event_probability_threshold_source": threshold_source,
            "event_threshold_tuning": threshold_tuning,
            "event_min_duration_days": event_min_duration,
            "event_min_hot_fraction": event_min_hot_fraction,
            "allow_sample_mean_fallback": bool(cfg.get("allow_sample_mean_fallback", False)),
            "require_dynamic_features": bool(cfg.get("require_dynamic_features", True)),
            "min_train_positive_rate": float(cfg.get("min_train_positive_rate", 0.01)),
            "max_train_positive_rate": float(cfg.get("max_train_positive_rate", 0.35)),
            "min_eval_positive_count": int(cfg.get("min_eval_positive_count", 10)),
            "inference_temp_boost_c": 1.5,
            "test_metrics": test_metrics,
            "train_positive_rate": float(y_train_event.mean()),
            "val_positive_rate": float(y_val_event.mean()),
            "test_positive_rate": float(y_test_event.mean()),
            "train_positive_count": int(np.sum(y_train_event)),
            "val_positive_count": int(np.sum(y_val_event)),
            "test_positive_count": int(np.sum(y_test_event)),
            "baseline_metrics": baseline_metrics,
            "seasonal_metrics": seasonal_metrics,
            "monthly_metrics": monthly_metrics,
            "regional_metrics": regional_metrics,
            "data_quality_report": quality_report,
            "lats": lats.tolist(),
            "lons": lons.tolist(),
            "rf_n_estimators": cfg["rf_n_estimators"],
            "rf_max_depth": cfg["rf_max_depth"],
            "rf_min_samples_leaf": cfg["rf_min_samples_leaf"],
            "rf_sampling_strategy": cfg["rf_sampling_strategy"],
            "rf_replacement": cfg["rf_replacement"],
            "model_backend": backend,
            "use_gpu": bool(cfg["use_gpu"]),
            "force_gpu": bool(cfg["force_gpu"]),
            "learning_rate": cfg["learning_rate"],
            "epochs": cfg["epochs"],
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "walk_forward_metrics": walk_forward_metrics,
        },
    }
    torch.save(checkpoint, save_path)
    print(f"Model checkpoint saved to: {save_path}")

    # Generate training report
    print("\n[Report] Generating training report...")
    report_path = generate_training_report(
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        baseline_metrics=baseline_metrics,
        seasonal_metrics=seasonal_metrics,
        regional_metrics=regional_metrics,
        walk_forward_metrics=walk_forward_metrics,
        cfg=cfg,
        output_dir="output",
    )

    return {
        "save_path": save_path,
        "model_type": model_type,
        "epochs": 1,
        "batch_size": cfg["batch_size"],
        "learning_rate": cfg["learning_rate"],
        "train_event_metrics": train_metrics,
        "val_event_metrics": val_metrics,
        "test_event_metrics": test_metrics,
        "baseline_metrics": baseline_metrics,
        "seasonal_metrics": seasonal_metrics,
        "monthly_metrics": monthly_metrics,
        "regional_metrics": regional_metrics,
        "data_quality_report": quality_report,
        "labeling_method": labeling_method,
        "heatwave_threshold_c": threshold_c,
        "walk_forward_metrics": walk_forward_metrics,
    }


def parse_args():
    """Parse command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Heatwave Prediction Model Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python Train_Ai.py                          # Train with defaults
  python Train_Ai.py --help                   # Show this help
  python Train_Ai.py --hp-tuning              # Enable hyperparameter tuning
  python Train_Ai.py --walk-forward --folds 5  # Walk-forward CV with 5 folds
  python Train_Ai.py --feature-eng            # Enable feature engineering

Environment Variables:
  All settings can be configured via environment variables with HW_ prefix:
  HW_BATCH_SIZE, HW_SEQ_LEN, HW_FUTURE_SEQ, HW_EPOCHS, HW_LEARNING_RATE,
  HW_TRAIN_RATIO, HW_VAL_RATIO, HW_TEST_RATIO, HW_RANDOM_SEED,
  HW_RF_N_ESTIMATORS, HW_RF_MAX_DEPTH, HW_RF_MIN_SAMPLES_LEAF,
  HW_WALK_FORWARD_ENABLED, HW_WALK_FORWARD_N_FOLDS,
  HW_HP_TUNING_ENABLED, HW_HP_N_ITER, HP_CV_FOLDS,
  HW_FEATURE_ENGINEERING_ENABLED, HW_FEATURE_TEMPORAL, HW_FEATURE_INTERACTIONS
        """
    )
    
    # Data parameters
    data_group = parser.add_argument_group("Data Parameters")
    data_group.add_argument("--seq-len", type=int, default=SEQ_LEN,
                           help=f"Input sequence length in days (default: {SEQ_LEN})")
    data_group.add_argument("--future-seq", type=int, default=FUTURE_SEQ,
                           help=f"Forecast horizon in days (default: {FUTURE_SEQ})")
    data_group.add_argument("--train-ratio", type=float, default=TRAIN_RATIO,
                           help=f"Training data ratio (default: {TRAIN_RATIO})")
    data_group.add_argument("--val-ratio", type=float, default=VAL_RATIO,
                           help=f"Validation data ratio (default: {VAL_RATIO})")
    data_group.add_argument("--test-ratio", type=float, default=TEST_RATIO,
                           help=f"Test data ratio (default: {TEST_RATIO})")
    
    # Random Forest parameters
    rf_group = parser.add_argument_group("Random Forest Parameters")
    rf_group.add_argument("--n-estimators", type=int, default=RF_N_ESTIMATORS,
                          help=f"Number of trees (default: {RF_N_ESTIMATORS})")
    rf_group.add_argument("--max-depth", type=int, default=RF_MAX_DEPTH,
                          help=f"Maximum tree depth (default: {RF_MAX_DEPTH})")
    rf_group.add_argument("--min-samples-leaf", type=int, default=RF_MIN_SAMPLES_LEAF,
                          help=f"Min samples per leaf (default: {RF_MIN_SAMPLES_LEAF})")
    
    # Hyperparameter tuning
    hp_group = parser.add_argument_group("Hyperparameter Tuning")
    hp_group.add_argument("--hp-tuning", action="store_true",
                         help="Enable hyperparameter search")
    hp_group.add_argument("--hp-n-iter", type=int, default=HP_N_ITER,
                         help=f"Number of parameter settings sampled (default: {HP_N_ITER})")
    hp_group.add_argument("--hp-cv-folds", type=int, default=HP_CV_FOLDS,
                         help=f"Number of CV folds for HP tuning (default: {HP_CV_FOLDS})")
    
    # Cross-validation
    cv_group = parser.add_argument_group("Cross-Validation")
    cv_group.add_argument("--walk-forward", action="store_true",
                         help="Enable walk-forward cross-validation")
    cv_group.add_argument("--folds", type=int, default=WALK_FORWARD_N_FOLDS,
                         help=f"Number of walk-forward folds (default: {WALK_FORWARD_N_FOLDS})")
    cv_group.add_argument("--expand-window", action="store_true", default=True,
                         help="Expand training window each fold (default: True)")
    
    # Feature engineering
    fe_group = parser.add_argument_group("Feature Engineering")
    fe_group.add_argument("--feature-eng", action="store_true",
                         help="Enable feature engineering")
    fe_group.add_argument("--spatial-features", action="store_true", default=True,
                         help="Add spatial summary features (default: True)")
    fe_group.add_argument("--temporal-features", action="store_true", default=True,
                         help="Add temporal trend features (default: True)")
    fe_group.add_argument("--interaction-features", action="store_true", default=True,
                         help="Add feature interactions (default: True)")
    
    # Threshold optimization
    thresh_group = parser.add_argument_group("Threshold Optimization")
    thresh_group.add_argument("--min-recall", type=float, default=0.6,
                             help="Minimum recall for constrained threshold (default: 0.6)")
    thresh_group.add_argument("--max-far", type=float, default=0.3,
                             help="Maximum false alarm rate for constrained threshold (default: 0.3)")
    
    # Heatwave definition
    hw_group = parser.add_argument_group("Heatwave Definition")
    hw_group.add_argument("--temp-threshold", type=float, default=HEATWAVE_TEMPERATURE_THRESHOLD,
                         help=f"Temperature threshold inC (default: {HEATWAVE_TEMPERATURE_THRESHOLD})")
    hw_group.add_argument("--min-duration", type=int, default=EVENT_MIN_DURATION_DAYS,
                         help=f"Minimum heatwave duration in days (default: {EVENT_MIN_DURATION_DAYS})")
    hw_group.add_argument("--labeling", type=str, default=LABELING_METHOD,
                         choices=["temperature", "heat_index", "anomaly"],
                         help=f"Labeling method (default: {LABELING_METHOD})")
    
    # Reproducibility
    repro_group = parser.add_argument_group("Reproducibility")
    repro_group.add_argument("--seed", type=int, default=RANDOM_SEED,
                            help=f"Random seed for reproducibility (default: {RANDOM_SEED})")
    
    return parser.parse_args()


def create_config_from_args(args):
    """Createconfig dictionary from parsed arguments."""
    config = {}
    
    # Data parameters
    config["seq_len"] = args.seq_len
    config["future_seq"] = args.future_seq
    config["train_ratio"] = args.train_ratio
    config["val_ratio"] = args.val_ratio
    config["test_ratio"] = args.test_ratio
    
    # RF parameters
    config["rf_n_estimators"] = args.n_estimators
    config["rf_max_depth"] = args.max_depth
    config["rf_min_samples_leaf"] = args.min_samples_leaf
    
    # HP tuning
    config["hp_tuning_enabled"] = args.hp_tuning
    config["hp_n_iter"] = args.hp_n_iter
    config["hp_cv_folds"] = args.hp_cv_folds
    
    # CV
    config["walk_forward_enabled"] = args.walk_forward
    config["walk_forward_n_folds"] = args.folds
    config["walk_forward_expand_window"] = args.expand_window
    
    # Feature engineering
    config["feature_engineering_enabled"] = args.feature_eng
    config["feature_spatial"] = args.spatial_features
    config["feature_temporal"] = args.temporal_features
    config["feature_interactions"] = args.interaction_features
    
    # Threshold
    config["min_recall_threshold"] = args.min_recall
    config["max_far_threshold"] = args.max_far
    
    # Heatwave
    config["heatwave_temperature_threshold"] = args.temp_threshold
    config["event_min_duration_days"] = args.min_duration
    config["labeling_method"] = args.labeling
    
    # Seed
    config["random_seed"] = args.seed
    
    return config


def main():
    """Main entry point with CLI support."""
    args = parse_args()
    
    print("=" *60)
    print("HEATWAVE PREDICTION MODEL TRAINING")
    print("=" * 60)
    print(f"Sequence Length: {args.seq_len} days")
    print(f"Forecast Horizon: {args.future_seq} days")
    print(f"Random Forest: n_estimators={args.n_estimators}, max_depth={args.max_depth}")
    print(f"HP Tuning: {'Enabled' if args.hp_tuning else 'Disabled'}")
    print(f"Walk-Forward CV: {'Enabled' if args.walk_forward else 'Disabled'}")
    print(f"Feature Engineering: {'Enabled' if args.feature_eng else 'Disabled'}")
    print("=" * 60)
    
    # Create config from args
    config = create_config_from_args(args)
    
    # Run training
    result = train(config=config)
    
    if result:
        print("\n" + "=" *60)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Model saved to: {result['save_path']}")
        print(f"Test F1: {result['test_event_metrics']['f1']:.4f}")
        print(f"Test Precision: {result['test_event_metrics']['precision']:.4f}")
        print(f"Test Recall: {result['test_event_metrics']['recall']:.4f}")
        if result.get('walk_forward_metrics', {}).get('aggregated'):
            wf = result['walk_forward_metrics']['aggregated']
            print(f"Walk-Forward CV F1: {wf.get('mean_f1', 0):.4f}")
    else:
        print("\nTraining failed. Check logs for details.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
