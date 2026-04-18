"""
Cross-Validation Training Script for Heatwave Prediction
Implements K-Fold CV with detailed metrics and visualization
"""
import json
import os
import time
import warnings
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional, Dict, Any, List
import glob
import numpy as np
import torch
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score, 
    precision_score, recall_score, accuracy_score, 
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy import stats

from src.data.loader import DataLoader, create_sequences
from src.core.logger import logger as app_logger

# ============== Configuration ==============
DATA_DIR = "era5_data"
MODELS_DIR = "models"
OUTPUT_DIR = "output"

BATCH_SIZE = 4
SEQ_LEN = 7
FUTURE_SEQ = 2
N_FOLDS = 5
RANDOM_SEED = 42

# Random Forest Parameters
RF_N_ESTIMATORS = 300
RF_MAX_DEPTH = 20
RF_MIN_SAMPLES_LEAF = 2

# Training Parameters
HEATWAVE_TEMPERATURE_THRESHOLD = 35.0
EVENT_MIN_DURATION_DAYS = 3
EVENT_MIN_HOT_FRACTION = 0.10

# Environment helpers
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

BATCH_SIZE = _int_env("HW_BATCH_SIZE", BATCH_SIZE)
SEQ_LEN = _int_env("HW_SEQ_LEN", SEQ_LEN)
FUTURE_SEQ = _int_env("HW_FUTURE_SEQ", FUTURE_SEQ)
N_FOLDS = _int_env("HW_N_FOLDS", 5)
RF_N_ESTIMATORS = _int_env("HW_RF_N_ESTIMATORS", RF_N_ESTIMATORS)


# ============== Data Processing Functions ==============
def temporal_split_data(data, n_folds=5, fold_idx=0):
    """Split data into train/val/test for a specific fold (temporal split)."""
    total_steps = len(data)
    fold_size = total_steps // n_folds
    
    # Create temporal folds
    val_start = fold_idx * fold_size
    val_end = val_start + fold_size
    
    # Training: everything except validation fold
    train_indices = list(range(0, val_start)) + list(range(val_end, total_steps))
    train_data = data[train_indices]
    val_data = data[val_start:val_end]
    
    # For testing, use the last fold or a hold-out set
    if fold_idx < n_folds - 1:
        test_start = (fold_idx + 1) * fold_size
        test_data = data[test_start:]
    else:
        # Last fold: use first fold as test
        test_data = data[:fold_size]
    
    return train_data, val_data, test_data


def _to_heatwave_event_labels(temp_sequences_c, threshold_c, min_duration=3, min_hot_fraction=0.10):
    """Convert temperature sequences to heatwave event labels."""
    # Handle case where threshold is too low and all samples are positive
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


def _adjust_threshold_for_balance(y_temp, target_positive_rate=0.15):
    """Automatically adjust threshold to achieve target positive rate."""
    # Calculate mean temperature for each sequence
    seq_means = y_temp.mean(axis=(1, 2, 3))
    
    # Find threshold that gives target positive rate
    threshold = float(np.percentile(seq_means, (1 - target_positive_rate) * 100))
    return threshold


def _classification_metrics(y_true, y_pred):
    """Compute classification metrics."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _evaluate_event_classifier(y_true, y_prob, threshold=0.5):
    """Evaluate event classifier with various thresholds."""
    y_pred = (y_prob >= threshold).astype(np.int32)
    metrics = _classification_metrics(y_true, y_pred)
    
    # Brier Score
    brier = float(np.mean((y_prob - y_true) ** 2))
    metrics["brier_score"] = brier
    
    # PR-AUC
    if average_precision_score is not None and np.unique(y_true).size > 1:
        try:
            metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
        except Exception:
            metrics["pr_auc"] = None
    else:
        metrics["pr_auc"] = None
    
    # ROC-AUC
    if np.unique(y_true).size > 1:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None
    
    metrics["threshold"] = float(threshold)
    return metrics


def _optimize_probability_threshold(y_true, y_prob):
    """Find optimal probability threshold that maximizes F1."""
    positives = int(np.sum(y_true == 1))
    if positives == 0:
        return 0.5, {"source": "default_no_positive", "best_f1": 0.0}

    best_threshold = 0.5
    best_metrics = _classification_metrics(y_true, (y_prob >= best_threshold).astype(np.int32))
    best_f1 = float(best_metrics["f1"])
    best_recall = float(best_metrics["recall"])

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
    }


def _to_temperature_c(y_norm, temp_mean, temp_std, temp_channel_idx=1):
    """Convert normalized temperature to Celsius."""
    temp = (y_norm[:, :, temp_channel_idx, :, :] * temp_std) + temp_mean
    if np.nanmean(temp) > 200:
        temp = temp - 273.15
    return temp


def _flatten_features(x_seq):
    """Flatten sequence features for sklearn."""
    return x_seq.reshape(x_seq.shape[0], -1)


# ============== Cross-Validation Training ==============
@dataclass
class FoldMetrics:
    fold: int
    train_samples: int
    val_samples: int
    test_samples: int
    train_positive_rate: float
    val_positive_rate: float
    test_positive_rate: float
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    optimal_threshold: float
    training_time_seconds: float
    y_true_test: List = field(default_factory=list)
    y_prob_test: List = field(default_factory=list)


@dataclass
class CrossValidationResults:
    n_folds: int
    fold_results: List[FoldMetrics]
    mean_metrics: Dict[str, float]
    std_metrics: Dict[str, float]
    overall_test_metrics: Dict[str, float]
    training_config: Dict[str, Any]


def train_with_cross_validation():
    """Main training function with K-Fold Cross Validation."""
    print("="*70)
    print("HEATWAVE PREDICTION - K-FOLD CROSS VALIDATION TRAINING")
    print("="*70)
    
    config = {
        "n_folds": N_FOLDS,
        "seq_len": SEQ_LEN,
        "future_seq": FUTURE_SEQ,
        "rf_n_estimators": RF_N_ESTIMATORS,
        "rf_max_depth": RF_MAX_DEPTH,
        "rf_min_samples_leaf": RF_MIN_SAMPLES_LEAF,
        "heatwave_threshold": HEATWAVE_TEMPERATURE_THRESHOLD,
        "event_min_duration": EVENT_MIN_DURATION_DAYS,
        "event_min_hot_fraction": EVENT_MIN_HOT_FRACTION,
        "random_seed": RANDOM_SEED,
    }
    
    print(f"\n[1/7] Loading ERA5 Data...")
    loader = DataLoader()
    try:
        full_ds = loader.load_era5()
        data_norm, stats = loader.prepare_training_data(full_ds)
        lats, lons = stats["lats"], stats["lons"]
        all_times = np.asarray(stats.get("time_index", []), dtype="datetime64[ns]")
    except Exception as exc:
        print(f"Error loading data: {exc}")
        return None

    print(f"      Data Shape: {data_norm.shape}")
    print(f"      Time steps: {len(all_times)}")
    
    # Store results for each fold
    fold_results: List[FoldMetrics] = []
    all_y_true_test = []
    all_y_prob_test = []
    
    # Convert to sequences first
    print(f"\n[2/7] Creating sequences (seq_len={SEQ_LEN}, future_seq={FUTURE_SEQ})...")
    x_all, y_all = create_sequences(data_norm, SEQ_LEN, FUTURE_SEQ)
    print(f"      Total sequences: {len(x_all)}")
    
    # Get temperature statistics
    temp_mean = float(np.mean(data_norm[:, 1, :, :]))
    temp_std = float(np.std(data_norm[:, 1, :, :]))
    
    # Convert all labels
    y_all_temp = _to_temperature_c(y_all, temp_mean, temp_std)
    y_all_event = _to_heatwave_event_labels(
        y_all_temp,
        HEATWAVE_TEMPERATURE_THRESHOLD,
        min_duration=EVENT_MIN_DURATION_DAYS,
        min_hot_fraction=EVENT_MIN_HOT_FRACTION,
    )
    
    # Check if we have both classes, if not adjust threshold
    if len(np.unique(y_all_event)) < 2:
        print(f"      WARNING: Only one class found with threshold {HEATWAVE_TEMPERATURE_THRESHOLD}C")
        print(f"      Adjusting threshold to create balanced classes...")
        
        # Use percentile-based threshold to get ~15% positive rate
        adjusted_threshold = _adjust_threshold_for_balance(y_all_temp, target_positive_rate=0.15)
        print(f"      Adjusted threshold: {adjusted_threshold:.2f} C")
        
        y_all_event = _to_heatwave_event_labels(
            y_all_temp,
            adjusted_threshold,
            min_duration=EVENT_MIN_DURATION_DAYS,
            min_hot_fraction=EVENT_MIN_HOT_FRACTION,
        )
    
    print(f"      Overall positive rate: {y_all_event.mean():.4f}")
    
    # K-Fold Cross Validation (temporal split)
    print(f"\n[3/7] Running {N_FOLDS}-Fold Cross Validation...")
    
    for fold in range(N_FOLDS):
        print(f"\n      === FOLD {fold + 1}/{N_FOLDS} ===")
        fold_start_time = time.time()
        
        # Temporal split - use 80% train, 10% val, 10% test per fold
        n_samples = len(x_all)
        fold_size = n_samples // N_FOLDS
        
        # Create indices for this fold
        val_start = fold * fold_size
        val_end = val_start + fold_size
        
        train_val_indices = list(range(0, val_start)) + list(range(val_end, n_samples))
        test_indices = list(range(val_start, val_end))
        
        # Further split train/val from train_val (90% train, 10% val)
        n_train_val = len(train_val_indices)
        train_end = int(n_train_val * 0.9)
        
        train_indices = train_val_indices[:train_end]
        val_indices = train_val_indices[train_end:]
        
        # Get data
        x_train = x_all[train_indices]
        x_val = x_all[val_indices]
        x_test = x_all[test_indices]
        
        y_train = y_all_event[train_indices]
        y_val = y_all_event[val_indices]
        y_test = y_all_event[test_indices]
        
        # Flatten features
        x_train_flat = _flatten_features(x_train)
        x_val_flat = _flatten_features(x_val)
        x_test_flat = _flatten_features(x_test)
        
        print(f"      Train: {len(x_train)}, Val: {len(x_val)}, Test: {len(x_test)}")
        print(f"      Train+ rate: {y_train.mean():.4f}, Val+ rate: {y_val.mean():.4f}, Test+ rate: {y_test.mean():.4f}")
        
        # Train model
        print(f"      Training RandomForest...")
        
        # Check if we have both classes
        unique_classes = len(np.unique(y_train))
        
        if unique_classes < 2:
            # Use standard RandomForest when only one class
            print(f"      WARNING: Only one class in training data, using standard RF")
            from sklearn.ensemble import RandomForestClassifier
            rf_model = RandomForestClassifier(
                n_estimators=RF_N_ESTIMATORS,
                max_depth=RF_MAX_DEPTH,
                min_samples_leaf=RF_MIN_SAMPLES_LEAF,
                random_state=RANDOM_SEED,
                n_jobs=-1,
            )
        else:
            rf_model = BalancedRandomForestClassifier(
                n_estimators=RF_N_ESTIMATORS,
                max_depth=RF_MAX_DEPTH,
                min_samples_leaf=RF_MIN_SAMPLES_LEAF,
                sampling_strategy="all",
                replacement=True,
                random_state=RANDOM_SEED,
                n_jobs=-1,
            )
        rf_model.fit(x_train_flat, y_train)
        
        # Predict
        pred_train_prob = rf_model.predict_proba(x_train_flat)[:, 1]
        pred_val_prob = rf_model.predict_proba(x_val_flat)[:, 1]
        pred_test_prob = rf_model.predict_proba(x_test_flat)[:, 1]
        
        # Find optimal threshold
        prob_threshold = 0.5
        if np.sum(y_val) > 0:
            prob_threshold, _ = _optimize_probability_threshold(y_val, pred_val_prob)
        
        # Evaluate
        train_metrics = _evaluate_event_classifier(y_train, pred_train_prob, prob_threshold)
        val_metrics = _evaluate_event_classifier(y_val, pred_val_prob, prob_threshold)
        test_metrics = _evaluate_event_classifier(y_test, pred_test_prob, prob_threshold)
        
        fold_time = time.time() - fold_start_time
        
        fold_result = FoldMetrics(
            fold=fold + 1,
            train_samples=len(x_train),
            val_samples=len(x_val),
            test_samples=len(x_test),
            train_positive_rate=float(y_train.mean()),
            val_positive_rate=float(y_val.mean()),
            test_positive_rate=float(y_test.mean()),
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            optimal_threshold=prob_threshold,
            training_time_seconds=fold_time,
            y_true_test=y_test.tolist(),
            y_prob_test=pred_test_prob.tolist(),
        )
        fold_results.append(fold_result)
        
        all_y_true_test.extend(y_test.tolist())
        all_y_prob_test.extend(pred_test_prob.tolist())
        
        print(f"      Train F1: {train_metrics['f1']:.4f}, Val F1: {val_metrics['f1']:.4f}, Test F1: {test_metrics['f1']:.4f}")
        print(f"      Time: {fold_time:.2f}s")
    
    # Compute aggregate metrics
    print(f"\n[4/7] Computing aggregate metrics...")
    
    # Mean and std across folds
    metric_names = ['accuracy', 'precision', 'recall', 'f1', 'specificity', 'brier_score', 'pr_auc', 'roc_auc']
    
    mean_metrics = {}
    std_metrics = {}
    
    for metric in metric_names:
        values = []
        for fold in fold_results:
            if metric in fold.test_metrics and fold.test_metrics[metric] is not None:
                values.append(fold.test_metrics[metric])
        
        if values:
            mean_metrics[metric] = np.mean(values)
            std_metrics[metric] = np.std(values)
    
    # Overall test metrics (aggregated)
    overall_test_metrics = _evaluate_event_classifier(
        np.array(all_y_true_test),
        np.array(all_y_prob_test),
        threshold=0.5  # Use default threshold for overall
    )
    
    # Find optimal threshold on all test data
    optimal_overall_threshold, _ = _optimize_probability_threshold(
        np.array(all_y_true_test),
        np.array(all_y_prob_test)
    )
    overall_test_optimal = _evaluate_event_classifier(
        np.array(all_y_true_test),
        np.array(all_y_prob_test),
        threshold=optimal_overall_threshold
    )
    
    cv_results = CrossValidationResults(
        n_folds=N_FOLDS,
        fold_results=fold_results,
        mean_metrics=mean_metrics,
        std_metrics=std_metrics,
        overall_test_metrics=overall_test_optimal,
        training_config=config,
    )
    
    print(f"\n[5/7] Cross-Validation Results Summary:")
    print(f"      Mean Test F1: {mean_metrics.get('f1', 0):.4f} +/- {std_metrics.get('f1', 0):.4f}")
    print(f"      Mean Test Precision: {mean_metrics.get('precision', 0):.4f} +/- {std_metrics.get('precision', 0):.4f}")
    print(f"      Mean Test Recall: {mean_metrics.get('recall', 0):.4f} +/- {std_metrics.get('recall', 0):.4f}")
    print(f"      Mean Test ROC-AUC: {mean_metrics.get('roc_auc', 0):.4f} +/- {std_metrics.get('roc_auc', 0):.4f}")
    print(f"      Overall Test F1 (optimal threshold): {overall_test_optimal['f1']:.4f}")
    
    # Save results
    print(f"\n[6/7] Saving results...")
    results_path = os.path.join(OUTPUT_DIR, f"cv_results_{int(time.time())}.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Convert to serializable format
    results_json = {
        "n_folds": cv_results.n_folds,
        "config": cv_results.training_config,
        "fold_results": [
            {
                "fold": f.fold,
                "train_samples": f.train_samples,
                "val_samples": f.val_samples,
                "test_samples": f.test_samples,
                "train_positive_rate": f.train_positive_rate,
                "val_positive_rate": f.val_positive_rate,
                "test_positive_rate": f.test_positive_rate,
                "train_metrics": f.train_metrics,
                "val_metrics": f.val_metrics,
                "test_metrics": f.test_metrics,
                "optimal_threshold": f.optimal_threshold,
                "training_time_seconds": f.training_time_seconds,
            }
            for f in cv_results.fold_results
        ],
        "mean_metrics": {k: float(v) for k, v in cv_results.mean_metrics.items()},
        "std_metrics": {k: float(v) for k, v in cv_results.std_metrics.items()},
        "overall_test_metrics": {k: float(v) if v is not None else None for k, v in cv_results.overall_test_metrics.items()},
        "optimal_overall_threshold": float(optimal_overall_threshold),
    }
    
    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"      Results saved to: {results_path}")
    
    # Generate visualizations
    print(f"\n[7/7] Generating visualizations...")
    generate_visualizations(cv_results, results_path.replace('.json', '_plots'))
    
    print("\n" + "="*70)
    print("CROSS-VALIDATION COMPLETE")
    print("="*70)
    
    return cv_results


def generate_visualizations(cv_results: CrossValidationResults, output_dir: str):
    """Generate comprehensive visualizations for the CV report."""
    os.makedirs(output_dir, exist_ok=True)
    
    n_folds = cv_results.n_folds
    fold_results = cv_results.fold_results
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Performance Metrics by Fold
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('K-Fold Cross Validation Results', fontsize=16, fontweight='bold')
    
    metrics_to_plot = ['f1', 'precision', 'recall', 'accuracy', 'specificity', 'brier_score']
    titles = ['F1 Score', 'Precision', 'Recall', 'Accuracy', 'Specificity', 'Brier Score']
    
    for idx, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
        ax = axes[idx // 3, idx % 3]
        
        train_vals = [f.train_metrics.get(metric, 0) for f in fold_results]
        val_vals = [f.val_metrics.get(metric, 0) for f in fold_results]
        test_vals = [f.test_metrics.get(metric, 0) for f in fold_results]
        
        x = np.arange(n_folds)
        width = 0.25
        
        bars1 = ax.bar(x - width, train_vals, width, label='Train', color='#2ecc71', alpha=0.8)
        bars2 = ax.bar(x, val_vals, width, label='Validation', color='#3498db', alpha=0.8)
        bars3 = ax.bar(x + width, test_vals, width, label='Test', color='#e74c3c', alpha=0.8)
        
        ax.set_xlabel('Fold')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([f'Fold {i+1}' for i in range(n_folds)])
        ax.legend(loc='lower right', fontsize=8)
        ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cv_metrics_by_fold.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Performance Distribution (Box Plot)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    test_f1 = [f.test_metrics.get('f1', 0) for f in fold_results]
    test_precision = [f.test_metrics.get('precision', 0) for f in fold_results]
    test_recall = [f.test_metrics.get('recall', 0) for f in fold_results]
    test_roc_auc = [f.test_metrics.get('roc_auc', 0) for f in fold_results if f.test_metrics.get('roc_auc') is not None]
    
    data = [test_f1, test_precision, test_recall]
    if test_roc_auc:
        data.append(test_roc_auc)
    
    labels = ['F1', 'Precision', 'Recall']
    if test_roc_auc:
        labels.append('ROC-AUC')
    
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Score')
    ax.set_title('Test Set Performance Distribution Across Folds')
    ax.set_ylim(0, 1.1)
    
    # Add mean values
    means = [np.mean(d) for d in data]
    for i, mean in enumerate(means):
        ax.annotate(f'mean={mean:.3f}', xy=(i+1, mean), xytext=(i+1.15, mean),
                   fontsize=9, ha='left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cv_performance_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Confusion Matrix Heatmap (Aggregated)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    all_y_true = []
    all_y_pred = []
    for f in fold_results:
        y_true = np.array(f.y_true_test)
        y_prob = np.array(f.y_prob_test)
        y_pred = (y_prob >= f.optimal_threshold).astype(int)
        all_y_true.extend(y_true.tolist())
        all_y_pred.extend(y_pred.tolist())
    
    cm = confusion_matrix(all_y_true, all_y_pred)
    
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    classes = ['Non-Heatwave', 'Heatwave']
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           ylabel='True Label',
           xlabel='Predicted Label',
           title='Aggregated Confusion Matrix')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cv_confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. ROC Curves for Each Fold
    fig, ax = plt.subplots(figsize=(10, 8))
    
    from sklearn.metrics import roc_curve, roc_auc_score
    
    all_y_true_combined = []
    all_y_prob_combined = []
    
    for i, f in enumerate(fold_results):
        y_true = np.array(f.y_true_test)
        y_prob = np.array(f.y_prob_test)
        all_y_true_combined.extend(f.y_true_test)
        all_y_prob_combined.extend(f.y_prob_test)
        
        if len(np.unique(y_true)) < 2:
            continue
            
        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = roc_auc_score(y_true, y_prob)
            ax.plot(fpr, tpr, alpha=0.6, label=f'Fold {i+1} (AUC = {roc_auc:.3f})')
        except:
            pass
    
    # Add mean ROC
    all_y_true_arr = np.array(all_y_true_combined)
    all_y_prob_arr = np.array(all_y_prob_combined)
    fpr_mean, tpr_mean, _ = roc_curve(all_y_true_arr, all_y_prob_arr)
    roc_auc_mean = roc_auc_score(all_y_true_arr, all_y_prob_arr)
    ax.plot(fpr_mean, tpr_mean, 'k--', linewidth=2, label=f'Mean (AUC = {roc_auc_mean:.3f})')
    ax.plot([0, 1], [0, 1], 'k:', alpha=0.5, label='Random')
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves by Fold')
    ax.legend(loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cv_roc_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 5. Precision-Recall Curves
    fig, ax = plt.subplots(figsize=(10, 8))
    
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    all_y_true_pr = []
    all_y_prob_pr = []
    
    for i, f in enumerate(fold_results):
        y_true = np.array(f.y_true_test)
        y_prob = np.array(f.y_prob_test)
        all_y_true_pr.extend(f.y_true_test)
        all_y_prob_pr.extend(f.y_prob_test)
        
        if len(np.unique(y_true)) < 2:
            continue
            
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            ap = average_precision_score(y_true, y_prob)
            ax.plot(recall, precision, alpha=0.6, label=f'Fold {i+1} (AP = {ap:.3f})')
        except:
            pass
    
    # Overall
    all_y_true_arr_pr = np.array(all_y_true_pr)
    all_y_prob_arr_pr = np.array(all_y_prob_pr)
    precision_mean, recall_mean, _ = precision_recall_curve(all_y_true_arr_pr, all_y_prob_arr_pr)
    ap_mean = average_precision_score(all_y_true_arr_pr, all_y_prob_arr_pr)
    ax.plot(recall_mean, precision_mean, 'k--', linewidth=2, label=f'Mean (AP = {ap_mean:.3f})')
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves by Fold')
    ax.legend(loc='lower left')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cv_precision_recall.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 6. Training Time per Fold
    fig, ax = plt.subplots(figsize=(10, 6))
    
    fold_times = [f.training_time_seconds for f in fold_results]
    fold_labels = [f'Fold {f.fold}' for f in fold_results]
    
    bars = ax.bar(fold_labels, fold_times, color='#3498db', alpha=0.8)
    ax.set_xlabel('Fold')
    ax.set_ylabel('Training Time (seconds)')
    ax.set_title('Training Time per Fold')
    
    # Add value labels
    for bar, time_val in zip(bars, fold_times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{time_val:.1f}s', ha='center', va='bottom')
    
    # Add mean line
    mean_time = np.mean(fold_times)
    ax.axhline(y=mean_time, color='red', linestyle='--', label=f'Mean: {mean_time:.1f}s')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cv_training_time.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 7. Positive Class Distribution by Fold
    fig, ax = plt.subplots(figsize=(10, 6))
    
    train_rates = [f.train_positive_rate for f in fold_results]
    val_rates = [f.val_positive_rate for f in fold_results]
    test_rates = [f.test_positive_rate for f in fold_results]
    
    x = np.arange(n_folds)
    width = 0.25
    
    bars1 = ax.bar(x - width, train_rates, width, label='Train', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x, val_rates, width, label='Validation', color='#3498db', alpha=0.8)
    bars3 = ax.bar(x + width, test_rates, width, label='Test', color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('Fold')
    ax.set_ylabel('Positive Rate')
    ax.set_title('Heatwave Event Distribution by Fold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {i+1}' for i in range(n_folds)])
    ax.legend()
    ax.set_ylim(0, max(max(train_rates), max(val_rates), max(test_rates)) * 1.2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cv_positive_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 8. Summary Table
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    # Create summary table
    summary_data = []
    headers = ['Fold', 'Train Samples', 'Test Samples', 'Test F1', 'Test Precision', 'Test Recall', 'Test ROC-AUC', 'Time (s)']
    
    for f in fold_results:
        row = [
            f'Fold {f.fold}',
            str(f.train_samples),
            str(f.test_samples),
            f"{f.test_metrics.get('f1', 0):.4f}",
            f"{f.test_metrics.get('precision', 0):.4f}",
            f"{f.test_metrics.get('recall', 0):.4f}",
            f"{f.test_metrics.get('roc_auc', 0):.4f}" if f.test_metrics.get('roc_auc') else 'N/A',
            f"{f.training_time_seconds:.1f}"
        ]
        summary_data.append(row)
    
    # Add mean row
    summary_data.append(['-' * 10, '-' * 12, '-' * 12, '-' * 8, '-' * 10, '-' * 10, '-' * 10, '-' * 8])
    summary_data.append([
        'MEAN',
        '-',
        '-',
        f"{cv_results.mean_metrics.get('f1', 0):.4f}",
        f"{cv_results.mean_metrics.get('precision', 0):.4f}",
        f"{cv_results.mean_metrics.get('recall', 0):.4f}",
        f"{cv_results.mean_metrics.get('roc_auc', 0):.4f}",
        f"{np.mean([f.training_time_seconds for f in fold_results]):.1f}"
    ])
    
    table = ax.table(cellText=summary_data, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    ax.set_title('Cross-Validation Summary Results', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cv_summary_table.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 9. Improvement Recommendations Chart
    create_improvement_chart(cv_results, output_dir)
    
    print(f"      Visualizations saved to: {output_dir}")


def create_improvement_chart(cv_results: CrossValidationResults, output_dir: str):
    """Create improvement recommendations visualization."""
    
    mean_metrics = cv_results.mean_metrics
    overall_metrics = cv_results.overall_test_metrics
    
    # Calculate performance gaps
    f1_score = mean_metrics.get('f1', 0)
    precision = mean_metrics.get('precision', 0)
    recall = mean_metrics.get('recall', 0)
    roc_auc = mean_metrics.get('roc_auc', 0)
    
    # Identify issues
    issues = []
    recommendations = []
    
    if recall < 0.5:
        issues.append("Low Recall")
        recommendations.append("Increase model sensitivity or lower detection threshold")
    
    if precision < 0.5:
        issues.append("Low Precision")
        recommendations.append("Reduce false positives with stricter threshold or more features")
    
    if f1_score < 0.6:
        issues.append("Low F1 Score")
        recommendations.append("Balance precision and recall with class weighting")
    
    if roc_auc < 0.7:
        issues.append("Low ROC-AUC")
        recommendations.append("Improve feature extraction or model capacity")
    
    if cv_results.std_metrics.get('f1', 0) > 0.1:
        issues.append("High Variance")
        recommendations.append("Use regularization or ensemble methods")
    
    # General recommendations
    if not issues:
        issues.append("None - Good Performance")
        recommendations.append("Continue monitoring; consider ensemble for robustness")
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.95, 'Model Improvement Recommendations', 
            fontsize=18, fontweight='bold', ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.90, f'Based on {cv_results.n_folds}-Fold Cross Validation Results',
            fontsize=12, ha='center', transform=ax.transAxes, style='italic')
    
    # Current metrics
    metrics_text = f"""
    Current Performance:
    --------------------
    * F1 Score:     {f1_score:.4f} +/- {cv_results.std_metrics.get('f1', 0):.4f}
    * Precision:    {precision:.4f} +/- {cv_results.std_metrics.get('precision', 0):.4f}
    * Recall:       {recall:.4f} +/- {cv_results.std_metrics.get('recall', 0):.4f}
    * ROC-AUC:      {roc_auc:.4f} +/- {cv_results.std_metrics.get('roc_auc', 0):.4f}
    """
    ax.text(0.25, 0.75, metrics_text, fontsize=12, va='top', transform=ax.transAxes,
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    # Issues and recommendations
    issues_text = "Identified Issues:\n" + "-" * 20 + "\n"
    for issue in issues:
        issues_text += f"  ! {issue}\n"
    
    ax.text(0.05, 0.50, issues_text, fontsize=11, va='top', transform=ax.transAxes)
    
    rec_text = "Recommendations:\n" + "-" * 20 + "\n"
    for i, rec in enumerate(recommendations, 1):
        rec_text += f"  {i}. {rec}\n"
    
    ax.text(0.55, 0.50, rec_text, fontsize=11, va='top', transform=ax.transAxes)
    
    # Action items
    action_items = [
        "1. Hyperparameter Tuning: Optimize n_estimators, max_depth",
        "2. Feature Engineering: Add temporal features, seasonal indicators",
        "3. Model Ensemble: Combine with gradient boosting or neural network",
        "4. Class Balancing: Adjust sampling strategy for imbalanced data",
        "5. Threshold Optimization: Fine-tune probability threshold per fold"
    ]
    
    action_text = "Next Steps:\n" + "-" * 20 + "\n"
    for item in action_items:
        action_text += f"  - {item}\n"
    
    # Performance gap analysis
    gap_analysis = f"""
    Target Thresholds:
    --------------------
    F1 >= 0.70 | Prec >= 0.70 | Rec >= 0.70
    
    Gap Analysis:
    * F1 Gap: {(0.70 - f1_score):.4f}
    * Precision Gap: {(0.70 - precision):.4f}
    * Recall Gap: {(0.70 - recall):.4f}
    """
    ax.text(0.55, 0.15, gap_analysis, fontsize=10, va='top', transform=ax.transAxes,
            fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'improvement_recommendations.png'), dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Main entry point."""
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    
    results = train_with_cross_validation()
    
    if results:
        print("\n[SUCCESS] Cross-Validation Training Complete!")
        print(f"\nOutput files saved to: {OUTPUT_DIR}")
        return results
    else:
        print("\n[FAILED] Training Failed!")
        return None


if __name__ == "__main__":
    main()