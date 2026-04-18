#!/usr/bin/env python3
"""
Cross-validation training for ConvLSTM (GPU-only).
Loads existing cv splits from tmp/cv_splits and writes per-fold results to tmp/cv_results/fold{i}.

Defaults: epochs=10, patience=3, batch_size prefers existing Train_ConvLSTM.py (4) otherwise 32.

This script enforces CUDA-only execution and will raise if CUDA is not available.
"""

import os
import sys
import json
import time
import argparse
import datetime
import random
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr

import numpy as np

# Add repo root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset

# Project imports
try:
    from src.data.loader import DataLoader as ERA5Loader, create_sequences
    from src.models.convlstm import HeatwaveConvLSTM, PhysicsInformedLoss
except Exception as e:
    raise


def load_indices(csv_file: Path):
    import csv
    out = []
    with csv_file.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                out.append(int(r.get('index', r.get('idx', ''))))
            except Exception:
                continue
    return np.array(out, dtype=np.int64)


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_cv(cv_dir: Path, results_dir: Path, seq_len: int = 7, future_seq: int = 2,
             epochs: int = 10, patience: int = 3, batch_size: int = 4, lr: float = 1e-4,
             num_workers: int = 4, preload_checkpoint: str = 'models\\heatwave_model_checkpoint_v26.pth'):

    cv_dir = Path(cv_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Ensure CUDA-only
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA not available. Aborting: GPU-only enforcement.')
    device = torch.device('cuda')

    # Load folds metadata
    folds_meta = cv_dir / 'folds.json'
    if not folds_meta.exists():
        raise FileNotFoundError(f'folds.json not found in {cv_dir}')
    folds = json.loads(folds_meta.read_text(encoding='utf-8'))

    k = int(folds.get('k', 5))

    # Load raw dataset once
    loader = ERA5Loader()
    print(f"Loading ERA5 data from {loader.data_dir} ...")
    ds = loader.load_era5()
    data_array, stats = loader.prepare_training_data(ds)

    T = data_array.shape[0]
    total_window = seq_len + future_seq
    num_samples = T - total_window + 1
    if num_samples <= 0:
        raise ValueError('Sequence configuration yields no samples. Check seq_len/future_seq and data length.')

    per_fold_results = {}

    # temp channel index for temperature (as used elsewhere)
    TEMP_CHANNEL_IDX = 1

    # Prepare device tensors for normalization later
    for fmeta in folds.get('folds', []):
        fold = int(fmeta.get('fold'))
        print('\n' + '=' * 60)
        print(f'Starting fold {fold}')
        train_idx_file = ROOT / fmeta.get('train_file')
        val_idx_file = ROOT / fmeta.get('val_file')
        if not train_idx_file.exists() or not val_idx_file.exists():
            print(f'Missing index files for fold {fold}: {train_idx_file}, {val_idx_file}')
            per_fold_results[str(fold)] = { 'error': 'missing index files' }
            continue

        fold_out = results_dir / f'fold{fold}'
        fold_out.mkdir(parents=True, exist_ok=True)
        log_file = fold_out / 'train.log'

        start_time = time.time()
        try:
            with open(log_file, 'w', encoding='utf-8') as lf, redirect_stdout(lf), redirect_stderr(lf):
                print(f'Fold {fold} started at {datetime.datetime.utcnow().isoformat()} UTC')
                print(f'Loading indices from {train_idx_file} and {val_idx_file}')
                train_time_indices = load_indices(train_idx_file)
                val_time_indices = load_indices(val_idx_file)

                # Map time indices -> sample indices (sample i corresponds to target at time i+seq_len)
                train_samples = train_time_indices - seq_len
                val_samples = val_time_indices - seq_len

                # Filter valid sample indices
                train_samples = train_samples[(train_samples >= 0) & (train_samples < num_samples)]
                val_samples = val_samples[(val_samples >= 0) & (val_samples < num_samples)]
                print(f'Train samples: {len(train_samples)}, Val samples: {len(val_samples)}')
                if len(train_samples) == 0 or len(val_samples) == 0:
                    raise ValueError('Empty train or validation sample set after mapping indices')

                # Build union of frames used in training sequences for normalization
                train_frame_set = set()
                for s in train_samples.tolist():
                    for fr in range(int(s), int(s + seq_len)):
                        train_frame_set.add(fr)
                train_frame_list = sorted(train_frame_set)
                print(f'Training frames for normalization: {len(train_frame_list)} frames (unique)')

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

                print(f'X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}')

                # Convert to tensors
                x_train_t = torch.from_numpy(X_train).float()
                y_train_t = torch.from_numpy(Y_train).float()
                x_val_t = torch.from_numpy(X_val).float()
                y_val_t = torch.from_numpy(Y_val).float()

                # DataLoaders with pin_memory and num_workers>0
                train_dataset = TensorDataset(x_train_t, y_train_t)
                val_dataset = TensorDataset(x_val_t, y_val_t)

                # Ensure num_workers > 0
                if num_workers <= 0:
                    num_workers = max(1, (os.cpu_count() or 2) - 1)

                train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
                val_loader = TorchDataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

                # Initialize model
                input_dim = int(data_array.shape[1])
                HIDDEN_DIM = [32, 32]
                KERNEL_SIZE = [(3, 3)] * len(HIDDEN_DIM)
                NUM_LAYERS = len(HIDDEN_DIM)

                model = HeatwaveConvLSTM(input_dim=input_dim, hidden_dim=HIDDEN_DIM, kernel_size=KERNEL_SIZE, num_layers=NUM_LAYERS)

                # Enforce CUDA-only device assignment
                if not torch.cuda.is_available():
                    raise RuntimeError('CUDA became unavailable during run - aborting (GPU-only)')
                model = model.to(device)

                criterion = PhysicsInformedLoss(lambda_phy=0.1).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                # Optionally load pretrained checkpoint (map to cuda device)
                pretrained_used = False
                ckpt_path = Path(preload_checkpoint)
                if ckpt_path.exists():
                    try:
                        ckpt = torch.load(str(ckpt_path), map_location='cuda')
                        # Various checkpoint formats supported
                        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                            model.load_state_dict(ckpt['model_state_dict'], strict=False)
                            pretrained_used = True
                        elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
                            model.load_state_dict(ckpt['state_dict'], strict=False)
                            pretrained_used = True
                        elif isinstance(ckpt, dict) and 'sklearn_model' in ckpt:
                            print('Warning: checkpoint appears to be sklearn model (not compatible)')
                        else:
                            # Try loading as state_dict directly
                            try:
                                model.load_state_dict(ckpt, strict=False)
                                pretrained_used = True
                            except Exception:
                                print('Pretrained checkpoint present but could not be loaded into ConvLSTM (skipping).')
                    except Exception as e:
                        print(f'Error loading pretrained checkpoint: {e} (continuing from scratch)')

                if pretrained_used:
                    print(f'Loaded pretrained weights from {ckpt_path}')
                else:
                    print('No compatible pretrained checkpoint found; training from scratch.')

                # Training loop with early stopping on val RMSE (temperature channel)
                best_val_rmse = float('inf')
                best_epoch = 0
                epochs_trained = 0
                patience_counter = 0
                train_loss_history = []
                val_loss_history = []

                # For denormalization we need train_mean/train_std as torch tensors on device
                mean_t = torch.from_numpy(train_mean).float().to(device)
                std_t = torch.from_numpy(train_std).float().to(device)

                for epoch in range(epochs):
                    epoch_start = time.time()
                    model.train()
                    running_train_loss = 0.0
                    batches = 0
                    for batch_x, batch_y in train_loader:
                        # Move to GPU only, enforce non-blocking with pin_memory
                        batch_x = batch_x.to(device, non_blocking=True)
                        batch_y = batch_y.to(device, non_blocking=True)

                        optimizer.zero_grad()
                        output = model(batch_x, future_seq=future_seq)
                        loss, mse_loss, phy_loss = criterion(output, batch_y)
                        loss.backward()
                        optimizer.step()

                        running_train_loss += loss.item()
                        batches += 1

                    avg_train_loss = running_train_loss / max(1, batches)
                    train_loss_history.append(float(avg_train_loss))

                    # Validation
                    model.eval()
                    running_val_loss = 0.0
                    val_batches = 0
                    # For RMSE calculation
                    val_sq_sum = 0.0
                    val_count = 0
                    with torch.no_grad():
                        for batch_x, batch_y in val_loader:
                            batch_x = batch_x.to(device, non_blocking=True)
                            batch_y = batch_y.to(device, non_blocking=True)
                            output = model(batch_x, future_seq=future_seq)
                            loss_val, _, _ = criterion(output, batch_y)
                            running_val_loss += loss_val.item()
                            val_batches += 1

                            # Denormalize temperature channel and compute RMSE
                            # output, batch_y shapes: (B, future_seq, C, H, W)
                            out_denorm = (output * std_t) + mean_t
                            y_denorm = (batch_y * std_t) + mean_t
                            # select temperature channel
                            out_temp = out_denorm[:, :, TEMP_CHANNEL_IDX, :, :]
                            y_temp = y_denorm[:, :, TEMP_CHANNEL_IDX, :, :]
                            diff = out_temp - y_temp
                            val_sq_sum += float((diff ** 2).sum().item())
                            val_count += int(diff.numel())

                    avg_val_loss = running_val_loss / max(1, val_batches)
                    val_loss_history.append(float(avg_val_loss))
                    val_rmse = float((val_sq_sum / val_count) ** 0.5) if val_count > 0 else float('nan')

                    epoch_time = time.time() - epoch_start
                    print(f'Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Val RMSE(temp): {val_rmse:.6f} | Time: {epoch_time:.1f}s')

                    epochs_trained += 1

                    # Check for improvement
                    if not (val_rmse is None or (isinstance(val_rmse, float) and (val_rmse != val_rmse))):
                        improved = val_rmse < best_val_rmse
                    else:
                        improved = False

                    if improved:
                        best_val_rmse = val_rmse
                        best_epoch = epoch + 1
                        patience_counter = 0
                        # Save best model
                        model_ckpt = {
                            'model_state_dict': model.state_dict(),
                            'epoch': epoch + 1,
                            'val_rmse': best_val_rmse,
                            'normalization_mean': train_mean.tolist(),
                            'normalization_std': train_std.tolist(),
                            'seq_len': seq_len,
                            'future_seq': future_seq,
                            'model_type': 'convlstm',
                        }
                        model_path = fold_out / 'model.pt'
                        torch.save(model_ckpt, str(model_path))
                        print(f'Saved best model to {model_path}')
                    else:
                        patience_counter += 1

                    if patience_counter >= patience:
                        print(f'Early stopping (patience {patience}) at epoch {epoch+1}')
                        break

                # After training
                metrics = {
                    'train_loss_history': train_loss_history,
                    'val_loss_history': val_loss_history,
                    'best_val_metric': best_val_rmse if best_val_rmse != float('inf') else None,
                    'epochs_trained': epochs_trained,
                    'timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
                }
                (fold_out / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')

                print(f"Saved metrics to {(fold_out / 'metrics.json').resolve()}")
                elapsed = time.time() - start_time
                print(f"Fold {fold} finished in {elapsed:.1f}s")

                per_fold_results[str(fold)] = {
                    'best_val_metric': metrics['best_val_metric'],
                    'train_loss_history': train_loss_history,
                    'val_loss_history': val_loss_history,
                    'epochs_trained': epochs_trained,
                    'model_path': str((fold_out / 'model.pt').resolve()) if (fold_out / 'model.pt').exists() else None,
                    'log_path': str(log_file.resolve()),
                }

        except Exception as e:
            # Write exception details to fold log and record error in results
            with open(log_file, 'a', encoding='utf-8') as lf:
                lf.write('\n' + '='*20 + '\n')
                lf.write(f'ERROR during fold {fold}: {e}\n')
            per_fold_results[str(fold)] = {'error': str(e)}
            print(f'Fold {fold} failed: {e}')
            continue

    # After folds: aggregate summary
    summary = {'folds': per_fold_results}

    def _collect_metric(key):
        vals = []
        for k, v in per_fold_results.items():
            if isinstance(v, dict) and key in v and v.get(key) is not None:
                try:
                    vals.append(float(v[key]))
                except Exception:
                    continue
        if not vals:
            return {'mean': None, 'std': None}
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=0))
        return {'mean': mean, 'std': std}

    best_val_stats = _collect_metric('best_val_metric')
    summary['aggregates'] = { 'best_val_metric': best_val_stats }

    (results_dir / 'cv_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'Wrote CV summary to {(results_dir / "cv_summary.json").resolve()}')

    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cv-dir', type=str, default='tmp/cv_splits')
    parser.add_argument('--results-dir', type=str, default='tmp/cv_results')
    parser.add_argument('--seq-len', type=int, default=7)
    parser.add_argument('--future-seq', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--preload-checkpoint', type=str, default='models\\heatwave_model_checkpoint_v26.pth')

    args = parser.parse_args()

    # Set seeds
    set_seeds(42)

    try:
        summary = train_cv(args.cv_dir, args.results_dir, seq_len=args.seq_len, future_seq=args.future_seq,
                           epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
                           lr=args.lr, num_workers=args.num_workers, preload_checkpoint=args.preload_checkpoint)
        print('Done. Summary:')
        print(json.dumps(summary, indent=2))
    except Exception as e:
        print(f'ERROR: {e}')
        raise
