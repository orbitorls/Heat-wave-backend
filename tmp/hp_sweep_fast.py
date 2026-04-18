"""Faster HP sweep: load dataset once and run multiple trials on fold0.
Saves outputs to tmp/cv_results/hp_sweep.
"""
from pathlib import Path
import sys
import os
import json
import time
import datetime
import random
import math
from contextlib import redirect_stdout, redirect_stderr

# Add repo root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset

# Project imports
from src.data.loader import DataLoader as ERA5Loader, create_sequences
from src.models.convlstm import HeatwaveConvLSTM, PhysicsInformedLoss


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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(trials=6, seed=42, seq_len=7, future_seq=2, epochs=6, patience=2, num_workers=2):
    set_seeds(seed)
    results_dir = Path('tmp') / 'cv_results' / 'hp_sweep'
    results_dir.mkdir(parents=True, exist_ok=True)

    # Ensure folds meta
    cv_dir = Path('tmp') / 'cv_splits'
    folds_meta = cv_dir / 'folds.json'
    if not folds_meta.exists():
        raise FileNotFoundError(f'folds.json not found in {cv_dir}')
    folds = json.loads(folds_meta.read_text(encoding='utf-8'))
    fmeta = None
    for f in folds.get('folds', []):
        if int(f.get('fold', -1)) == 0:
            fmeta = f
            break
    if fmeta is None:
        raise FileNotFoundError('fold 0 not found in folds.json')

    train_idx_file = ROOT / fmeta.get('train_file')
    val_idx_file = ROOT / fmeta.get('val_file')

    # Load indices
    train_time_indices = load_indices(train_idx_file)
    val_time_indices = load_indices(val_idx_file)

    # Load dataset once
    print('Loading ERA5 dataset (once) ...')
    loader = ERA5Loader()
    ds = loader.load_era5()
    data_array, stats = loader.prepare_training_data(ds)

    T = data_array.shape[0]
    total_window = seq_len + future_seq
    num_samples = T - total_window + 1
    if num_samples <= 0:
        raise ValueError('Sequence configuration yields no samples. Check seq_len/future_seq and data length.')

    # Map time indices -> sample indices
    train_samples = train_time_indices - seq_len
    val_samples = val_time_indices - seq_len
    train_samples = train_samples[(train_samples >= 0) & (train_samples < num_samples)]
    val_samples = val_samples[(val_samples >= 0) & (val_samples < num_samples)]
    if len(train_samples) == 0 or len(val_samples) == 0:
        raise ValueError('Empty train or validation sample set after mapping indices')

    # Build union of frames for normalization
    train_frame_set = set()
    for s in train_samples.tolist():
        for fr in range(int(s), int(s + seq_len)):
            train_frame_set.add(fr)
    train_frame_list = sorted(train_frame_set)
    train_frames = data_array[train_frame_list, ...]
    train_mean = train_frames.mean(axis=(0, 2, 3), keepdims=True)
    train_std = train_frames.std(axis=(0, 2, 3), keepdims=True)
    eps = 1e-8
    train_std = np.where(train_std < eps, eps, train_std)

    # Normalize dataset
    data_norm = (data_array - train_mean) / train_std

    # Create sequences and select
    X_all, Y_all = create_sequences(data_norm, seq_len, future_seq)
    X_train = X_all[train_samples]
    Y_train = Y_all[train_samples]
    X_val = X_all[val_samples]
    Y_val = Y_all[val_samples]

    # Convert to tensors
    x_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(Y_train).float()
    x_val_t = torch.from_numpy(X_val).float()
    y_val_t = torch.from_numpy(Y_val).float()

    print(f'Prepared data: X_train {X_train.shape}, X_val {X_val.shape}')

    # Check CUDA
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA not available. Aborting: GPU-only enforcement.')
    device = torch.device('cuda')

    # Setup static tensors
    mean_t = torch.from_numpy(train_mean).float().to(device)
    std_t = torch.from_numpy(train_std).float().to(device)

    TEMP_CHANNEL_IDX = 1

    results = []

    # Pre-create small DataLoaders per batch_size inside loop
    for i in range(trials):
        # sample hyperparams
        lr = float(math.exp(random.uniform(math.log(1e-5), math.log(1e-3))))
        weight_decay = float(random.uniform(0.0, 1e-4))
        batch_size = random.choice([32, 64])

        trial_json = results_dir / f'trial_{i}.json'
        trial_log = results_dir / f'trial_{i}_train.log'
        trial_model = results_dir / f'trial_{i}_model.pt'

        with open(trial_log, 'w', encoding='utf-8') as lf, redirect_stdout(lf), redirect_stderr(lf):
            print(f"=== Trial {i} | lr={lr:.2e} wd={weight_decay:.1e} bs={batch_size} ===")
            start = time.time()

            train_dataset = TensorDataset(x_train_t, y_train_t)
            val_dataset = TensorDataset(x_val_t, y_val_t)
            train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=max(1, int(num_workers)))
            val_loader = TorchDataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=max(1, int(num_workers)))

            # Model
            input_dim = int(data_array.shape[1])
            HIDDEN_DIM = [32, 32]
            KERNEL_SIZE = [(3, 3)] * len(HIDDEN_DIM)
            NUM_LAYERS = len(HIDDEN_DIM)

            model = HeatwaveConvLSTM(input_dim=input_dim, hidden_dim=HIDDEN_DIM, kernel_size=KERNEL_SIZE, num_layers=NUM_LAYERS)
            model = model.to(device)

            criterion = PhysicsInformedLoss(lambda_phy=0.1).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            scaler = torch.cuda.amp.GradScaler()

            best_val_rmse = float('inf')
            patience_counter = 0
            epochs_trained = 0

            final_val_rmse = float('nan')
            final_val_mae = float('nan')
            final_val_r2 = float('nan')

            for epoch in range(epochs):
                epoch_start = time.time()
                model.train()
                running_train_loss = 0.0
                batches = 0
                for batch_x, batch_y in train_loader:
                    batch_x = batch_x.to(device, non_blocking=True)
                    batch_y = batch_y.to(device, non_blocking=True)
                    optimizer.zero_grad()
                    with torch.cuda.amp.autocast():
                        output = model(batch_x, future_seq=future_seq)
                        loss, mse_loss, phy_loss = criterion(output, batch_y)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    running_train_loss += float(loss.item())
                    batches += 1

                avg_train_loss = running_train_loss / max(1, batches)

                # Validation
                model.eval()
                running_val_loss = 0.0
                val_batches = 0
                val_sq_sum = 0.0
                val_abs_sum = 0.0
                val_count = 0
                sum_y = 0.0
                sum_y_sq = 0.0

                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(device, non_blocking=True)
                        batch_y = batch_y.to(device, non_blocking=True)
                        with torch.cuda.amp.autocast():
                            output = model(batch_x, future_seq=future_seq)
                            loss_val, _, _ = criterion(output, batch_y)
                        running_val_loss += float(loss_val.item())
                        val_batches += 1

                        out_denorm = (output * std_t) + mean_t
                        y_denorm = (batch_y * std_t) + mean_t

                        out_temp = out_denorm[:, :, TEMP_CHANNEL_IDX, :, :]
                        y_temp = y_denorm[:, :, TEMP_CHANNEL_IDX, :, :]
                        diff = out_temp - y_temp

                        val_sq_sum += float((diff ** 2).sum().item())
                        val_abs_sum += float(diff.abs().sum().item())
                        val_count += int(diff.numel())
                        sum_y += float(y_temp.sum().item())
                        sum_y_sq += float((y_temp ** 2).sum().item())

                avg_val_loss = running_val_loss / max(1, val_batches)
                val_rmse = (val_sq_sum / val_count) ** 0.5 if val_count > 0 else float('nan')
                val_mae = val_abs_sum / val_count if val_count > 0 else float('nan')
                mean_y = sum_y / val_count if val_count > 0 else 0.0
                ss_tot = sum_y_sq - val_count * mean_y * mean_y if val_count > 0 else 0.0
                val_r2 = 1.0 - (val_sq_sum / ss_tot) if ss_tot > 0 else float('nan')

                epochs_trained += 1
                print(f'Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Val RMSE(temp): {val_rmse:.6f} | Val MAE(temp): {val_mae:.6f}')

                improved = False
                if not (val_rmse is None or (isinstance(val_rmse, float) and (val_rmse != val_rmse))):
                    improved = val_rmse < best_val_rmse

                if improved:
                    best_val_rmse = val_rmse
                    patience_counter = 0
                    torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch + 1, 'val_rmse': best_val_rmse}, str(trial_model))
                    print(f'Saved best model to {trial_model}')
                else:
                    patience_counter += 1

                final_val_rmse = val_rmse
                final_val_mae = val_mae
                final_val_r2 = val_r2

                if patience_counter >= patience:
                    print(f'Early stopping (patience {patience}) at epoch {epoch+1}')
                    break

            elapsed = time.time() - start
            print(f'Trial {i} finished in {elapsed:.1f}s, epochs_trained={epochs_trained}')

            out = {
                'lr': lr,
                'weight_decay': weight_decay,
                'batch_size': batch_size,
                'val_RMSE': final_val_rmse,
                'val_MAE': final_val_mae,
                'val_R2': final_val_r2,
                'epochs_trained': epochs_trained,
                'timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
            }

            trial_json.write_text(json.dumps(out, indent=2), encoding='utf-8')
            print(f'Wrote trial JSON to {trial_json}')
            results.append({'trial': i, **out})

    # pick best by lowest val_RMSE
    best = None
    for r in results:
        val = r.get('val_RMSE')
        if val is None:
            continue
        if isinstance(val, float) and (val != val):
            continue
        if best is None or (val < best.get('val_RMSE')):
            best = r

    best_path = results_dir / 'best_config.json'
    if best is not None:
        best_path.write_text(json.dumps(best, indent=2), encoding='utf-8')
        print(f'WROTE BEST CONFIG: {best_path}')
    else:
        best_path.write_text(json.dumps({'error': 'no successful trials', 'results': results}, indent=2), encoding='utf-8')
        print('No successful trials; wrote error best_config.json')

    print('Done')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=6)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(trials=args.trials, seed=args.seed)
