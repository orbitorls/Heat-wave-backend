#!/usr/bin/env python
"""evaluate_model.py — Standalone model evaluation tool for Heatwave ConvLSTM.

Usage
-----
# Evaluate latest checkpoint automatically:
    python evaluate_model.py

# Evaluate a specific checkpoint:
    python evaluate_model.py --checkpoint models/heatwave_convlstm_v3.pth

# Use specific data directory and sequence length:
    python evaluate_model.py --data-dir era5_data --seq-len 7 --future-seq 2

# Save detailed results to JSON:
    python evaluate_model.py --output-json output/eval_results.json

# Verbose mode (print per-sample stats):
    python evaluate_model.py --verbose
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

# ── ensure repo root is importable ──────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.core.logger import logger
from src.data.loader import clean_data, compute_normalization_stats, create_sequences, normalize_data
from src.models.convlstm import HeatwaveConvLSTM, PhysicsInformedLoss
from src.models.manager import ModelManager

try:
    from src.evaluation.metrics import per_lead_time_metrics, print_metrics_report
    _metrics_available = True
except ImportError:
    _metrics_available = False

# ────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ────────────────────────────────────────────────────────────────────────────

TEMP_CHANNEL = 1          # index of the t2m channel in the (C, H, W) stack
HEATWAVE_THRESHOLD_C = 35.0  # °C — threshold for heatwave classification


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def max_error(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.max(np.abs(pred - target)))


def r_squared(pred: np.ndarray, target: np.ndarray) -> float:
    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2)
    if ss_tot < 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def heatwave_detection_metrics(
    pred_temp: np.ndarray,
    target_temp: np.ndarray,
    threshold: float = HEATWAVE_THRESHOLD_C,
) -> Dict[str, float]:
    """Binary classification metrics for heatwave grid cells.

    Args:
        pred_temp:   (N, H, W) or (H, W) array of predicted temperatures (°C)
        target_temp: same shape, ground-truth temperatures (°C)
        threshold:   temperature above which a cell is considered a heatwave

    Returns:
        dict with precision, recall, f1, accuracy keys
    """
    pred_flat = pred_temp.ravel()
    tgt_flat = target_temp.ravel()

    tp = np.sum((pred_flat >= threshold) & (tgt_flat >= threshold))
    fp = np.sum((pred_flat >= threshold) & (tgt_flat < threshold))
    fn = np.sum((pred_flat < threshold) & (tgt_flat >= threshold))
    tn = np.sum((pred_flat < threshold) & (tgt_flat < threshold))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")
    recall    = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    f1        = (
        float(2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else float("nan")
    )
    accuracy  = float((tp + tn) / (tp + fp + fn + tn)) if (tp + fp + fn + tn) > 0 else float("nan")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
    }


# ────────────────────────────────────────────────────────────────────────────
# Checkpoint loading
# ────────────────────────────────────────────────────────────────────────────


def load_convlstm_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[HeatwaveConvLSTM, dict, Optional[np.ndarray], Optional[np.ndarray]]:
    """Load a HeatwaveConvLSTM checkpoint.

    Returns:
        model, metadata, norm_mean, norm_std
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "model_state_dict" not in checkpoint:
        raise ValueError(
            f"Checkpoint does not contain 'model_state_dict'. "
            f"Keys found: {list(checkpoint.keys())}"
        )

    metadata = checkpoint.get("metadata", {})
    input_dim  = metadata.get("input_dim", 8)
    hidden_dim = metadata.get("hidden_dim", [16, 16])
    kernel_size = metadata.get("kernel_size", [(3, 3), (3, 3)])
    num_layers  = metadata.get("num_layers", 2)

    model = HeatwaveConvLSTM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        kernel_size=kernel_size,
        num_layers=num_layers,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    norm_mean_raw = metadata.get("normalization_mean") or checkpoint.get("normalization_mean")
    norm_std_raw  = metadata.get("normalization_std")  or checkpoint.get("normalization_std")
    norm_mean = np.asarray(norm_mean_raw) if norm_mean_raw is not None else None
    norm_std  = np.asarray(norm_std_raw)  if norm_std_raw  is not None else None

    return model, metadata, norm_mean, norm_std


# ────────────────────────────────────────────────────────────────────────────
# Denormalization
# ────────────────────────────────────────────────────────────────────────────


def denorm_temp(
    grid: np.ndarray,
    norm_mean: Optional[np.ndarray],
    norm_std: Optional[np.ndarray],
    channel_idx: int = TEMP_CHANNEL,
) -> np.ndarray:
    """Denormalize a temperature grid. Converts Kelvin to Celsius if > 200."""
    if norm_mean is None or norm_std is None:
        return grid
    mean_val = float(np.take(norm_mean.squeeze(), channel_idx))
    std_val  = float(np.take(norm_std.squeeze(),  channel_idx))
    result = grid * std_val + mean_val
    if np.nanmean(result) > 200:
        result = result - 273.15
    return result


# ────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ────────────────────────────────────────────────────────────────────────────


def evaluate(
    model: HeatwaveConvLSTM,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    norm_mean: Optional[np.ndarray],
    norm_std: Optional[np.ndarray],
    future_seq: int,
    device: torch.device,
    verbose: bool = False,
) -> Dict:
    """Run evaluation on the test split and collect metrics.

    Args:
        X_test:    (N, seq_len, C, H, W) input sequences
        Y_test:    (N, future_seq, C, H, W) ground-truth targets
        norm_mean: normalization mean (or None)
        norm_std:  normalization std  (or None)
        future_seq: number of future steps
        device:    torch device
        verbose:   if True, print per-batch progress

    Returns:
        dict with all computed metrics
    """
    loss_fn = PhysicsInformedLoss()

    all_preds_temp   = []
    all_targets_temp = []
    total_losses_mse = []
    total_losses_phy = []
    inference_times  = []

    n_samples = X_test.shape[0]
    batch_size = min(8, n_samples)

    model.eval()
    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end   = min(start + batch_size, n_samples)
            batch_x = torch.from_numpy(X_test[start:end]).float().to(device)
            batch_y = torch.from_numpy(Y_test[start:end]).float().to(device)

            t0 = time.perf_counter()
            pred = model(batch_x, future_seq=future_seq)
            elapsed = time.perf_counter() - t0
            inference_times.append(elapsed / (end - start))  # per sample

            _, mse_loss, phy_loss = loss_fn(pred, batch_y)
            total_losses_mse.append(mse_loss.item())
            total_losses_phy.append(phy_loss.item())

            # Extract temperature channel (index TEMP_CHANNEL)
            pred_cpu   = pred.cpu().numpy()
            target_cpu = batch_y.cpu().numpy()

            # Shape: (batch, future_seq, C, H, W) → take temp channel
            pred_temp   = denorm_temp(pred_cpu[:, :, TEMP_CHANNEL, :, :],   norm_mean, norm_std)
            target_temp = denorm_temp(target_cpu[:, :, TEMP_CHANNEL, :, :], norm_mean, norm_std)

            all_preds_temp.append(pred_temp)
            all_targets_temp.append(target_temp)

            if verbose:
                print(
                    f"  Batch {start:4d}-{end:4d} | "
                    f"MSE={mse_loss.item():.4f} | "
                    f"PHY={phy_loss.item():.4f} | "
                    f"AvgTime={elapsed/(end-start)*1000:.1f}ms/sample"
                )

    all_preds_temp   = np.concatenate(all_preds_temp,   axis=0)
    all_targets_temp = np.concatenate(all_targets_temp, axis=0)

    # Regression metrics on temperature
    temp_mae  = mae(all_preds_temp, all_targets_temp)
    temp_rmse = rmse(all_preds_temp, all_targets_temp)
    temp_max  = max_error(all_preds_temp, all_targets_temp)
    temp_r2   = r_squared(all_preds_temp.ravel(), all_targets_temp.ravel())

    # Heatwave detection metrics
    hw_metrics = heatwave_detection_metrics(all_preds_temp, all_targets_temp)

    # Per-lead-time metrics (shape: N, future_seq, H, W)
    lead_time_metrics: Dict = {}
    if _metrics_available and all_preds_temp.ndim >= 2:
        lead_time_metrics = per_lead_time_metrics(all_targets_temp, all_preds_temp)

    # Temperature range info
    pred_min  = float(np.nanmin(all_preds_temp))
    pred_max  = float(np.nanmax(all_preds_temp))
    tgt_min   = float(np.nanmin(all_targets_temp))
    tgt_max   = float(np.nanmax(all_targets_temp))

    results = {
        "n_test_samples": n_samples,
        "future_seq": future_seq,
        "temperature_metrics": {
            "mae_celsius":        round(temp_mae,  4),
            "rmse_celsius":       round(temp_rmse, 4),
            "max_error_celsius":  round(temp_max,  4),
            "r_squared":          round(temp_r2,   4) if not np.isnan(temp_r2) else None,
        },
        "loss_metrics": {
            "mean_mse_loss":      round(float(np.mean(total_losses_mse)), 6),
            "mean_physics_loss":  round(float(np.mean(total_losses_phy)), 6),
        },
        "heatwave_detection": hw_metrics,
        "temperature_range": {
            "pred_min_celsius":   round(pred_min, 2),
            "pred_max_celsius":   round(pred_max, 2),
            "target_min_celsius": round(tgt_min,  2),
            "target_max_celsius": round(tgt_max,  2),
        },
        "performance": {
            "mean_inference_ms_per_sample": round(float(np.mean(inference_times)) * 1000, 2),
        },
        "lead_time_metrics": lead_time_metrics,
    }
    return results


# ────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ────────────────────────────────────────────────────────────────────────────


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained HeatwaveConvLSTM model on test data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Path to model checkpoint (.pth). Defaults to latest in models/.",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("era5_data"),
        help="Directory containing ERA5 NetCDF files.",
    )
    parser.add_argument(
        "--seq-len", type=int, default=7,
        help="Input sequence length (time steps).",
    )
    parser.add_argument(
        "--future-seq", type=int, default=2,
        help="Number of future time steps to predict.",
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.15,
        help="Fraction of data to use as test set.",
    )
    parser.add_argument(
        "--output-json", type=Path, default=None,
        help="Save evaluation results to this JSON file.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-batch progress.",
    )
    return parser.parse_args()


def _print_results(results: Dict) -> None:
    w = 55
    print("\n" + "=" * w)
    print("  MODEL EVALUATION RESULTS")
    print("=" * w)
    print(f"  Test samples:   {results['n_test_samples']}")
    print(f"  Future steps:   {results['future_seq']}")

    print("\n  ── Temperature Metrics ──────────────────────────")
    tm = results["temperature_metrics"]
    print(f"  MAE  (°C):      {tm['mae_celsius']}")
    print(f"  RMSE (°C):      {tm['rmse_celsius']}")
    print(f"  Max Error (°C): {tm['max_error_celsius']}")
    print(f"  R²:             {tm['r_squared']}")

    print("\n  ── Loss Metrics ─────────────────────────────────")
    lm = results["loss_metrics"]
    print(f"  MSE Loss:       {lm['mean_mse_loss']}")
    print(f"  Physics Loss:   {lm['mean_physics_loss']}")

    print("\n  ── Heatwave Detection (≥35°C) ───────────────────")
    hw = results["heatwave_detection"]
    print(f"  Precision:      {hw['precision']:.4f}" if hw["precision"] is not None and not np.isnan(hw["precision"]) else "  Precision:      N/A")
    print(f"  Recall:         {hw['recall']:.4f}"    if hw["recall"]    is not None and not np.isnan(hw["recall"])    else "  Recall:         N/A")
    print(f"  F1 Score:       {hw['f1']:.4f}"        if hw["f1"]        is not None and not np.isnan(hw["f1"])        else "  F1 Score:       N/A")
    print(f"  Accuracy:       {hw['accuracy']:.4f}"  if hw["accuracy"]  is not None and not np.isnan(hw["accuracy"])  else "  Accuracy:       N/A")
    print(f"  TP={hw['true_positives']}  FP={hw['false_positives']}  FN={hw['false_negatives']}  TN={hw['true_negatives']}")

    print("\n  ── Temperature Range ────────────────────────────")
    tr = results["temperature_range"]
    print(f"  Predicted:      [{tr['pred_min_celsius']}, {tr['pred_max_celsius']}] °C")
    print(f"  Actual:         [{tr['target_min_celsius']}, {tr['target_max_celsius']}] °C")

    print("\n  ── Performance ──────────────────────────────────")
    perf = results["performance"]
    print(f"  Inference:      {perf['mean_inference_ms_per_sample']} ms/sample")

    lead_metrics = results.get("lead_time_metrics", {})
    if lead_metrics and _metrics_available:
        print_metrics_report(lead_metrics, title="Per-Lead-Time Temperature Metrics (°C)")

    print("=" * w + "\n")


def main() -> None:
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Resolve checkpoint ───────────────────────────────────────────────
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        mm = ModelManager()
        checkpoint_path = mm.get_latest_checkpoint()
        if checkpoint_path is None:
            logger.error("No checkpoint found in models/. Train a model first.")
            sys.exit(1)
    print(f"Checkpoint: {checkpoint_path}")

    # ── Load model ───────────────────────────────────────────────────────
    print("Loading model...")
    try:
        model, metadata, norm_mean, norm_std = load_convlstm_from_checkpoint(checkpoint_path, device)
    except Exception as exc:
        logger.error(f"Failed to load checkpoint: {exc}")
        sys.exit(1)

    model_type = metadata.get("model_type", "unknown")
    if "sklearn" in model_type or "rf" in model_type.lower():
        logger.error(
            "Checkpoint is a Balanced Random Forest model (sklearn). "
            "evaluate_model.py supports ConvLSTM checkpoints only."
        )
        sys.exit(1)

    print(
        f"Model loaded — layers={model.num_layers}, "
        f"hidden={model.hidden_dim}, input_dim={model.input_dim}"
    )

    # ── Load & preprocess data ───────────────────────────────────────────
    print(f"Loading ERA5 data from: {args.data_dir} ...")
    try:
        from src.data.loader import DataLoader

        loader = DataLoader()
        loader.data_dir = args.data_dir
        ds = loader.load_era5()
        raw, stats = loader.prepare_training_data(ds)
    except Exception as exc:
        logger.error(f"Data loading failed: {exc}")
        sys.exit(1)

    print(f"Raw data shape: {raw.shape}  (Time, C, H, W)")

    # Clean
    cleaned, _ = clean_data(raw)

    # Split into train / val / test
    n = cleaned.shape[0]
    val_start  = int(n * (1.0 - args.test_ratio * 2))
    test_start = int(n * (1.0 - args.test_ratio))

    train_data = cleaned[:val_start]
    test_data  = cleaned[test_start:]
    print(f"Test split: {test_start}:{n} ({len(test_data)} time steps)")

    # Normalization — use training stats if available from checkpoint
    if norm_mean is not None and norm_std is not None:
        print("Using normalization stats from checkpoint.")
    else:
        print("Computing normalization stats from training data.")
        norm_mean, norm_std = compute_normalization_stats(train_data)

    test_norm = normalize_data(test_data, norm_mean, norm_std)

    # Build sequences
    X_test, Y_test = create_sequences(test_norm, seq_len=args.seq_len, pred_len=args.future_seq)
    if X_test.size == 0:
        logger.error(
            f"Not enough test data to build sequences "
            f"(need ≥ {args.seq_len + args.future_seq} steps, got {len(test_data)})."
        )
        sys.exit(1)

    print(f"Test sequences: {X_test.shape[0]}  (seq_len={args.seq_len}, future_seq={args.future_seq})")

    # ── Evaluate ─────────────────────────────────────────────────────────
    print("\nRunning evaluation...")
    results = evaluate(
        model=model,
        X_test=X_test,
        Y_test=Y_test,
        norm_mean=norm_mean,
        norm_std=norm_std,
        future_seq=args.future_seq,
        device=device,
        verbose=args.verbose,
    )

    results["checkpoint"] = str(checkpoint_path)
    results["data_dir"]   = str(args.data_dir)

    _print_results(results)

    # ── Save JSON ────────────────────────────────────────────────────────
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"Results saved to: {args.output_json}")


if __name__ == "__main__":
    main()
