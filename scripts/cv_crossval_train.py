#!/usr/bin/env python3
"""k-fold cross-validation training wrapper for Heatwave project.

Uses existing cv_splits in tmp/cv_splits and saves per-fold results to tmp/cv_results/fold{0..k-1}.

This script trains a BalancedRandomForestClassifier for each fold using the project's DataLoader
and sequence helpers. It normalizes per-fold using the union of training frames (no leakage).

Note: For speed this defaults to n_estimators=100; override with --n-estimators.
"""

import os
import sys
import json
import time
import argparse
import datetime
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr

import numpy as np
import random

# Import project code
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from src.data.loader import DataLoader, create_sequences
except Exception as e:
    print(f"Error importing project loader: {e}")
    raise

# sklearn / imblearn
try:
    from imblearn.ensemble import BalancedRandomForestClassifier
except Exception:
    BalancedRandomForestClassifier = None

# Helpers (ported/adapted from Train_Ai.py)

def _flatten_features(x_seq: np.ndarray) -> np.ndarray:
    return x_seq.reshape(x_seq.shape[0], -1)


def _to_temperature_c(y_norm, temp_mean, temp_std, temp_channel_idx=1):
    temp = (y_norm[:, :, temp_channel_idx, :, :] * temp_std) + temp_mean
    if np.nanmean(temp) > 200:
        temp = temp - 273.15
    return temp


def _to_heatwave_event_labels(temp_sequences_c, threshold_c, min_duration=3, min_hot_fraction=0.10):
    # temp_sequences_c: (N, future_seq, H, W)
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
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "hit_rate": hit_rate,
        "false_alarm_rate": false_alarm_rate,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _evaluate_event_classifier(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(np.int32)
    metrics = _classification_metrics(y_true, y_pred)
    brier = float(np.mean((y_prob - y_true) ** 2))
    metrics["brier_score"] = brier
    metrics["threshold"] = float(threshold)
    return metrics


def _optimize_probability_threshold(y_true, y_prob):
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

    return best_threshold, {"source": "f1_search", "best_f1": float(best_f1), "best_recall": float(best_recall)}


def load_indices(csv_file: Path):
    import csv
    out = []
    with csv_file.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                out.append(int(r.get("index", r.get("idx", ""))))
            except Exception:
                continue
    return np.array(out, dtype=np.int64)


def train_cv(cv_dir: Path, results_dir: Path, seq_len: int = 7, future_seq: int = 2, n_estimators: int = 100, max_depth: int = 20, min_samples_leaf: int = 2, random_state: int = 42, force_recreate: bool = False):
    cv_dir = Path(cv_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load folds metadata
    folds_meta = cv_dir / "folds.json"
    if not folds_meta.exists():
        raise FileNotFoundError(f"folds.json not found in {cv_dir}")
    folds = json.loads(folds_meta.read_text(encoding='utf-8'))
    k = int(folds.get("k", 5))

    # Load raw dataset
    loader = DataLoader()
    print(f"Loading ERA5 data from {loader.data_dir} ...")
    ds = loader.load_era5()
    data_array, stats = loader.prepare_training_data(ds)
    T = data_array.shape[0]
    total_window = seq_len + future_seq
    num_samples = T - total_window + 1
    if num_samples <= 0:
        raise ValueError("Sequence configuration yields no samples. Check seq_len/future_seq and data length.")

    print(f"Data frames: {T}, seq_len={seq_len}, future_seq={future_seq}, num_samples={num_samples}")

    per_fold_results = {}

    # Iterate folds
    for fmeta in folds.get("folds", []):
        fold = int(fmeta.get("fold"))
        print("\n" + "="*60)
        print(f"Starting fold {fold}")
        train_idx_file = ROOT / fmeta.get("train_file")
        val_idx_file = ROOT / fmeta.get("val_file")
        if not train_idx_file.exists() or not val_idx_file.exists():
            print(f"Missing index files for fold {fold}: {train_idx_file}, {val_idx_file}")
            per_fold_results[str(fold)] = {"error": "missing index files"}
            continue

        fold_out = results_dir / f"fold{fold}"
        fold_out.mkdir(parents=True, exist_ok=True)
        log_file = fold_out / "train.log"

        start_time = time.time()
        try:
            # Redirect prints to per-fold log
            with open(log_file, 'w', encoding='utf-8') as lf, redirect_stdout(lf), redirect_stderr(lf):
                print(f"Fold {fold} started at {datetime.datetime.utcnow().isoformat()} UTC")
                print(f"Loading indices from {train_idx_file} and {val_idx_file}")
                train_time_indices = load_indices(train_idx_file)
                val_time_indices = load_indices(val_idx_file)

                # Map time indices -> sample indices (sample i corresponds to target at time i+seq_len)
                train_samples = train_time_indices - seq_len
                val_samples = val_time_indices - seq_len
                # Filter valid sample indices
                train_samples = train_samples[(train_samples >= 0) & (train_samples < num_samples)]
                val_samples = val_samples[(val_samples >= 0) & (val_samples < num_samples)]
                print(f"Train samples: {len(train_samples)}, Val samples: {len(val_samples)}")

                if len(train_samples) == 0 or len(val_samples) == 0:
                    raise ValueError("Empty train or validation sample set after mapping indices")

                # Build union of frames used in training sequences for normalization
                train_frame_set = set()
                for s in train_samples.tolist():
                    for fr in range(int(s), int(s + seq_len)):
                        train_frame_set.add(fr)
                train_frame_list = sorted(train_frame_set)
                print(f"Training frames for normalization: {len(train_frame_list)} frames (unique)")

                train_frames = data_array[train_frame_list, ...]
                train_mean = train_frames.mean(axis=(0, 2, 3), keepdims=True)
                train_std = train_frames.std(axis=(0, 2, 3), keepdims=True)
                eps = 1e-8
                train_std = np.where(train_std < eps, eps, train_std)

                # Normalize entire dataset using training stats (no leakage)
                data_norm = (data_array - train_mean) / train_std

                # Create sequences from normalized data
                X_all, Y_all = create_sequences(data_norm, seq_len, future_seq)

                # Select samples for train/val
                X_train = X_all[train_samples]
                Y_train = Y_all[train_samples]
                X_val = X_all[val_samples]
                Y_val = Y_all[val_samples]

                print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")

                # Flatten features
                x_train_flat = _flatten_features(X_train)
                x_val_flat = _flatten_features(X_val)

                # Compute event labels from denormalized temperatures
                temp_mean_scalar = float(train_mean[0, 1, 0, 0])
                temp_std_scalar = float(train_std[0, 1, 0, 0])
                y_train_temp = _to_temperature_c(Y_train, temp_mean_scalar, temp_std_scalar, temp_channel_idx=1)
                y_val_temp = _to_temperature_c(Y_val, temp_mean_scalar, temp_std_scalar, temp_channel_idx=1)

                threshold_c = 35.0
                y_train_event = _to_heatwave_event_labels(y_train_temp, threshold_c)
                y_val_event = _to_heatwave_event_labels(y_val_temp, threshold_c)

                print(f"Train positive rate: {y_train_event.mean():.4f}, Val positive rate: {y_val_event.mean():.4f}")

                # Initialize classifier
                if BalancedRandomForestClassifier is None:
                    raise RuntimeError("imbalanced-learn not available (BalancedRandomForestClassifier)")

                clf = BalancedRandomForestClassifier(
                    n_estimators=int(n_estimators),
                    max_depth=None if int(max_depth) <= 0 else int(max_depth),
                    min_samples_leaf=int(min_samples_leaf),
                    sampling_strategy="all",
                    replacement=True,
                    random_state=int(random_state),
                    n_jobs=-1,
                )

                # Fit
                print(f"Fitting BalancedRandomForestClassifier (n_estimators={n_estimators})...")
                clf.fit(x_train_flat, y_train_event)

                pred_train_prob = clf.predict_proba(x_train_flat)[:, 1]
                pred_val_prob = clf.predict_proba(x_val_flat)[:, 1]

                prob_threshold, tuning = _optimize_probability_threshold(y_val_event, pred_val_prob) if np.sum(y_val_event) > 0 else (0.5, {"source":"default"})

                train_metrics = _evaluate_event_classifier(y_train_event, pred_train_prob, threshold=prob_threshold)
                val_metrics = _evaluate_event_classifier(y_val_event, pred_val_prob, threshold=prob_threshold)

                print(f"Fold {fold} results: Train Brier={train_metrics['brier_score']:.6f}, Val Brier={val_metrics['brier_score']:.6f}, Val F1={val_metrics['f1']:.4f}")

                # Save model checkpoint
                save_ckpt = {
                    "model_type": "balanced_random_forest",
                    "sklearn_model": clf,
                    "metadata": {
                        "seq_len": seq_len,
                        "future_seq": future_seq,
                        "normalization_mean": train_mean.tolist(),
                        "normalization_std": train_std.tolist(),
                        "threshold_c": threshold_c,
                        "n_estimators": int(n_estimators),
                    },
                }
                model_path = fold_out / "model.pt"
                try:
                    import torch
                    torch.save(save_ckpt, str(model_path))
                except Exception as e:
                    print(f"Warning: failed to torch.save sklearn checkpoint: {e}. Will try pickle fallback.")
                    import pickle
                    with open(fold_out / "model.pkl", 'wb') as pf:
                        pickle.dump(save_ckpt, pf)

                # Save metrics.json
                metrics = {
                    "train_loss": train_metrics.get('brier_score'),
                    "val_loss": val_metrics.get('brier_score'),
                    "val_metric": val_metrics.get('f1'),
                    "epochs_trained": 1,
                    "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                }
                (fold_out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding='utf-8')

                print(f"Saved model and metrics to {fold_out}")
                elapsed = time.time() - start_time
                print(f"Fold {fold} finished in {elapsed:.1f}s")

                # Collect per-fold results for summary
                per_fold_results[str(fold)] = {
                    "train_loss": metrics["train_loss"],
                    "val_loss": metrics["val_loss"],
                    "val_f1": metrics["val_metric"],
                    "n_train": int(len(train_samples)),
                    "n_val": int(len(val_samples)),
                    "model_path": str(model_path.resolve()),
                    "log_path": str(log_file.resolve()),
                }

        except Exception as e:
            # Write exception details to fold log and record error in results
            with open(log_file, 'a', encoding='utf-8') as lf:
                lf.write('\n' + '='*20 + '\n')
                lf.write(f"ERROR during fold {fold}: {e}\n")
            per_fold_results[str(fold)] = {"error": str(e)}
            print(f"Fold {fold} failed: {e}")
            continue

    # After folds: aggregate summary
    summary = {"folds": per_fold_results}
    # compute aggregate stats for val_f1, val_loss, train_loss
    import math
    def _collect_metric(key):
        vals = []
        for k, v in per_fold_results.items():
            if isinstance(v, dict) and key in v and v.get(key) is not None:
                vals.append(float(v[key]))
        if not vals:
            return {"mean": None, "std": None}
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=0))
        return {"mean": mean, "std": std}

    val_f1_stats = _collect_metric('val_f1')
    val_loss_stats = _collect_metric('val_loss')
    train_loss_stats = _collect_metric('train_loss')

    summary['aggregates'] = {
        'val_f1': val_f1_stats,
        'val_loss': val_loss_stats,
        'train_loss': train_loss_stats,
    }

    (results_dir / 'cv_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f"Wrote CV summary to {(results_dir / 'cv_summary.json').resolve()}")

    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cv-dir', type=str, default='tmp/cv_splits')
    parser.add_argument('--results-dir', type=str, default='tmp/cv_results')
    parser.add_argument('--seq-len', type=int, default=7)
    parser.add_argument('--future-seq', type=int, default=2)
    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--max-depth', type=int, default=20)
    parser.add_argument('--min-samples-leaf', type=int, default=2)
    parser.add_argument('--random-seed', type=int, default=42)
    args = parser.parse_args()

    # Set seeds
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    try:
        import torch
        torch.manual_seed(args.random_seed)
    except Exception:
        pass

    summary = train_cv(args.cv_dir, args.results_dir, seq_len=args.seq_len, future_seq=args.future_seq, n_estimators=args.n_estimators, max_depth=args.max_depth, min_samples_leaf=args.min_samples_leaf, random_state=args.random_seed)
    print("Done. Summary:")
    print(json.dumps(summary, indent=2))
