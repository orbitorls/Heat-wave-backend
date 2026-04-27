import argparse
import os
import sys
import json
from pathlib import Path
import csv

# Ensure repo root is importable
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from src.data.loader import DataLoader
    from src.core.config import settings
except Exception as e:
    print(f"Error importing project modules: {e}")
    raise

import numpy as np


def safe_str(x):
    try:
        return str(x)
    except Exception:
        return repr(x)


def main():
    parser = argparse.ArgumentParser(description="Create k-fold CV splits from ERA5 dataset time indices")
    parser.add_argument("--k", type=int, default=5, help="Number of folds")
    parser.add_argument("--output_dir", type=str, default="tmp/cv_splits", help="Output directory for splits (relative to repo root)")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility when shuffling")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before splitting (default: False)")
    args = parser.parse_args()

    out_dir = Path(REPO_ROOT) / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader()
    # Ensure loader.data_dir points to configured DATA_DIR
    try:
        from src.core.config import settings as cfg
        loader.data_dir = Path(cfg.DATA_DIR)
    except Exception:
        pass

    print(f"Loading dataset from {loader.data_dir} ...")
    ds = loader.load_era5()

    # Extract time index
    try:
        time_vals = ds.time.values
    except Exception:
        # fallback: try coords
        time_vals = ds.coords.get('time').values

    time_strs = [safe_str(t) for t in time_vals]
    n = len(time_strs)

    print(f"Found {n} time samples")

    # Save time index mapping
    time_index_file = out_dir / "time_index.csv"
    with time_index_file.open("w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["index", "time"])
        for i, t in enumerate(time_strs):
            writer.writerow([i, t])

    # Prepare folds
    indices = np.arange(n)
    if args.shuffle:
        rng = np.random.RandomState(args.random_state)
        rng.shuffle(indices)

    # Try sklearn KFold first
    try:
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=args.k, shuffle=args.shuffle, random_state=args.random_state if args.shuffle else None)
        splits = list(kf.split(indices))
        # splits is a list of (train_idx, val_idx) pairs where indices are positions in 'indices'
        # We need to map them to actual sample indices
        folds = []
        for fold_i, (train_pos, val_pos) in enumerate(splits):
            train_idx = indices[train_pos].tolist()
            val_idx = indices[val_pos].tolist()
            folds.append((train_idx, val_idx))
    except Exception:
        # Fallback: simple numpy split
        parts = np.array_split(indices, args.k)
        folds = []
        for i in range(args.k):
            val_idx = parts[i].tolist()
            train_idx = np.hstack([parts[j] for j in range(args.k) if j != i]).tolist()
            folds.append((train_idx, val_idx))

    metadata = {
        "k": args.k,
        "shuffle": bool(args.shuffle),
        "random_state": int(args.random_state),
        "folds": []
    }

    for i, (train_idx, val_idx) in enumerate(folds):
        train_file = out_dir / f"train_indices_fold{i}.csv"
        val_file = out_dir / f"val_indices_fold{i}.csv"

        with train_file.open("w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["index", "time"])
            for idx in train_idx:
                writer.writerow([int(idx), time_strs[int(idx)]])

        with val_file.open("w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["index", "time"])
            for idx in val_idx:
                writer.writerow([int(idx), time_strs[int(idx)]])

        metadata["folds"].append({
            "fold": i,
            "train_file": str(train_file.relative_to(REPO_ROOT)),
            "val_file": str(val_file.relative_to(REPO_ROOT)),
            "train_count": len(train_idx),
            "val_count": len(val_idx)
        })

    # Save metadata
    meta_file = out_dir / "folds.json"
    with meta_file.open("w", encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print(f"Wrote {len(folds)} folds to {out_dir}")
    for f in metadata['folds']:
        print(f"Fold {f['fold']}: train={f['train_count']} val={f['val_count']}")


if __name__ == '__main__':
    main()
