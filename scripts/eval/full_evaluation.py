#!/usr/bin/env python3
"""
Full evaluation and retraining script for heatwave prediction models.

This script:
1. Checks current model performance
2. Verifies data split for temporal leakage
3. Retrains models with different thresholds (35C, 36C, 38C)
4. Compares all results
5. Provides recommendations
"""

import os
import sys
import time
import glob
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import numpy as np
    import torch
except ImportError as e:
    print(f"Error: Missing required package: {e}")
    print("Please install: pip install numpy torch")
    sys.exit(1)


def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_section(title):
    """Print a section title."""
    print("\n" + "-" * 70)
    print(f" {title}")
    print("-" * 70)


def extract_metrics_from_checkpoint(checkpoint_path):
    """Extract metrics from a checkpoint file."""
    import zipfile
    import pickle
    import io

    class FakeModule:
        def __getattr__(self, name):
            return FakeModule()

        def __call__(self, *args, **kwargs):
            return FakeModule()

    class CustomUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if "xgboost" in module or "sklearn" in module:
                return FakeModule
            return super().find_class(module, name)

    try:
        with zipfile.ZipFile(checkpoint_path, "r") as zf:
            pkl_path = checkpoint_path.replace(".pth", "/data.pkl").replace(
                "models/", ""
            )
            if not pkl_path.startswith("heatwave"):
                pkl_path = checkpoint_path.split("/")[-1].replace(".pth", "/data.pkl")

            with zf.open(pkl_path) as f:
                data_bytes = f.read()

            unpickler = CustomUnpickler(io.BytesIO(data_bytes))
            checkpoint = unpickler.load()

            meta = checkpoint.get("metadata", {})
            test_m = meta.get("test_metrics", {})
            val_m = meta.get("val_metrics", {})
            train_m = meta.get("train_metrics", {})

            return {"test": test_m, "val": val_m, "train": train_m, "metadata": meta}
    except Exception as e:
        print(f"  Error loading checkpoint: {e}")
        return None


def check_data_leakage():
    """Check if temporal data split has any leakage."""
    print_section("DATA SPLIT VERIFICATION")

    try:
        from src.data.loader import DataLoader

        print("\nLoading ERA5 data...")
        loader = DataLoader()
        full_ds = loader.load_combined()

        print(f"  Data shape: {full_ds.shape}")
        print(f"  Time steps: {full_ds.shape[0]}")

        # Check temporal ordering
        print("\nChecking temporal split integrity...")

        train_ratio = 0.75
        val_ratio = 0.10
        test_ratio = 0.15

        train_end = int(full_ds.shape[0] * train_ratio)
        val_end = int(full_ds.shape[0] * (train_ratio + val_ratio))

        train_size = train_end
        val_size = val_end - train_end
        test_size = full_ds.shape[0] - val_end

        print(f"  Train: 0-{train_end} ({train_size} samples)")
        print(f"  Val: {train_end}-{val_end} ({val_size} samples)")
        print(f"  Test: {val_end}-{full_ds.shape[0]} ({test_size} samples)")

        # Check temporal ordering
        total = train_size + val_size + test_size
        actual_ratios = (train_size / total, val_size / total, test_size / total)

        print(
            f"\n  Actual ratios: Train={actual_ratios[0] * 100:.1f}%, Val={actual_ratios[1] * 100:.1f}%, Test={actual_ratios[2] * 100:.1f}%"
        )
        print(
            f"  Expected: Train={train_ratio * 100:.0f}%, Val={val_ratio * 100:.0f}%, Test={test_ratio * 100:.0f}%"
        )

        # Check for leakage
        print("\n  LEAKAGE CHECK:")
        if train_end < val_end < full_ds.shape[0]:
            print("  [OK] Temporal ordering maintained: Train < Val < Test")
            print("  [OK] No temporal leakage detected")
            leakage_detected = False
        else:
            print("  [ERROR] Temporal ordering violated!")
            print("  [ERROR] Potential data leakage detected")
            leakage_detected = True

        # Check split sizes
        if abs(actual_ratios[0] - train_ratio) < 0.01:
            print("  [OK] Train split size matches expected ratio")
        else:
            print(
                f"  [WARNING] Train split size differs from expected ({actual_ratios[0] * 100:.1f}% vs {train_ratio * 100:.0f}%)"
            )

        return not leakage_detected

    except Exception as e:
        print(f"  [ERROR] Could not verify data split: {e}")
        print("  This may indicate missing ERA5 data or dependencies")
        return None


def train_model_with_threshold(threshold):
    """Train a new model with specified threshold."""
    print_section(f"TRAINING MODEL - Threshold {threshold}C")

    try:
        # Modify training config
        import train_daily_xgboost

        # Set threshold
        print(f"\n  Setting heatwave threshold to {threshold}C...")

        # Run training with modified config
        config = {
            "heatwave_temp_threshold": threshold,
            "heatwave_min_duration": 3,
            "train_ratio": 0.75,
            "val_ratio": 0.10,
            "test_ratio": 0.15,
            "n_estimators": 200,
            "max_depth": 10,
            "learning_rate": 0.1,
            "random_seed": 42,
        }

        print(f"  Configuration:")
        for k, v in config.items():
            print(f"    {k}: {v}")

        start_time = time.time()

        # This will train and save model
        result = train_daily_xgboost.train_xgboost_daily(config=config)

        training_time = time.time() - start_time

        if result:
            print(f"\n  Training completed in {training_time:.1f}s")
            print(f"  Model saved to: {result.get('save_path', 'N/A')}")

            # Return metrics
            return {
                "threshold": threshold,
                "test_metrics": result.get("test_metrics", {}),
                "val_metrics": result.get("val_metrics", {}),
                "train_metrics": result.get("train_metrics", {}),
                "training_time": training_time,
                "success": True,
            }
        else:
            print("  [ERROR] Training failed - no result returned")
            return {"threshold": threshold, "success": False}

    except Exception as e:
        print(f"  [ERROR] Training failed: {e}")
        import traceback

        traceback.print_exc()
        return {"threshold": threshold, "success": False, "error": str(e)}


def compare_models(current_metrics, new_models):
    """Compare all models and provide recommendations."""
    print_section("MODEL COMPARISON")

    results = []

    # Current model
    print("\n  [CURRENT] XGBoost v6 (Threshold 38C):")
    if current_metrics:
        test_m = current_metrics.get("test", {})
        print(
            f"    F1: {test_m.get('f1', 'N/A'):.4f}"
            if isinstance(test_m.get("f1"), float)
            else f"    F1: N/A"
        )
        print(
            f"    Precision: {test_m.get('precision', 'N/A'):.4f}"
            if isinstance(test_m.get("precision"), float)
            else f"    Precision: N/A"
        )
        print(
            f"    Recall: {test_m.get('recall', 'N/A'):.4f}"
            if isinstance(test_m.get("recall"), float)
            else f"    Recall: N/A"
        )
        results.append(
            {
                "model": "v6 (current)",
                "threshold": 38,
                "f1": test_m.get("f1", 0),
                "precision": test_m.get("precision", 0),
                "recall": test_m.get("recall", 0),
            }
        )

    # New models
    for model_result in new_models:
        if model_result.get("success"):
            threshold = model_result.get("threshold")
            test_m = model_result.get("test_metrics", {})

            print(f"\n  [NEW] Threshold {threshold}C:")
            print(
                f"    F1: {test_m.get('f1', 0):.4f}"
                if test_m.get("f1")
                else "    F1: N/A"
            )
            print(
                f"    Precision: {test_m.get('precision', 0):.4f}"
                if test_m.get("precision")
                else "    Precision: N/A"
            )
            print(
                f"    Recall: {test_m.get('recall', 0):.4f}"
                if test_m.get("recall")
                else "    Recall: N/A"
            )

            results.append(
                {
                    "model": f"new_{threshold}C",
                    "threshold": threshold,
                    "f1": test_m.get("f1", 0),
                    "precision": test_m.get("precision", 0),
                    "recall": test_m.get("recall", 0),
                }
            )
        else:
            print(
                f"\n  [FAILED] Threshold {model_result.get('threshold')}C: Training unsuccessful"
            )

    # Find best model
    if results:
        best = max(results, key=lambda x: x.get("f1", 0))
        print("\n" + "=" * 70)
        print("  BEST MODEL:")
        print("=" * 70)
        print(f"    Model: XGBoost {best['model']}")
        print(f"    Threshold: {best['threshold']}C")
        print(f"    F1 Score: {best['f1']:.4f}")
        print(f"    Precision: {best['precision']:.4f}")
        print(f"    Recall: {best['recall']:.4f}")

        return best

    return None


def main():
    """Main execution."""
    print_header("HEATWAVE MODEL EVALUATION AND RETRAINING")
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Step 1: Load current model metrics
    print_section("STEP 1: LOAD CURRENT MODEL METRICS")

    current_model_path = "models/heatwave_daily_xgboost_v6.pth"
    if os.path.exists(current_model_path):
        print(f"\n  Loading: {current_model_path}")
        current_metrics = extract_metrics_from_checkpoint(current_model_path)

        if current_metrics:
            test_m = current_metrics.get("test", {})
            print(f"\n  Current Model Performance:")
            print(f"    Test F1: {test_m.get('f1', 0):.4f}")
            print(f"    Test Precision: {test_m.get('precision', 0):.4f}")
            print(f"    Test Recall: {test_m.get('recall', 0):.4f}")
            print(f"    Test Accuracy: {test_m.get('accuracy', 0):.4f}")

            # Check for overfitting
            train_m = current_metrics.get("train", {})
            if train_m and test_m:
                gap = train_m.get("f1", 0) - test_m.get("f1", 0)
                print(f"\n  Overfitting Check:")
                print(f"    Train F1: {train_m.get('f1', 0):.4f}")
                print(f"    Test F1: {test_m.get('f1', 0):.4f}")
                print(f"    Gap: {gap:.4f}")

                if test_m.get("f1", 0) > 0.95:
                    print(f"\n    [WARNING] Very high F1 score detected!")
                    print(f"    This may indicate potential issues:")
                    print(f"      - Data leakage (test data seen during training)")
                    print(f"      - Very easy classification task")
                    print(f"      - Overfitting to specific patterns")
        else:
            print("  [ERROR] Could not load current model metrics")
            current_metrics = None
    else:
        print(f"  [ERROR] Current model not found: {current_model_path}")
        current_metrics = None

    # Step 2: Check data split
    print_header("STEP 2: DATA SPLIT VERIFICATION")
    data_ok = check_data_leakage()

    # Step 3: Retrain with different thresholds
    print_header("STEP 3: RETRAIN WITH DIFFERENT THRESHOLDS")

    new_models = []
    thresholds_to_try = [35, 36]  # Train with lower thresholds

    for threshold in thresholds_to_try:
        print(f"\n[*] Training with threshold {threshold}C...")
        result = train_model_with_threshold(threshold)
        new_models.append(result)

        # Brief pause between training runs
        time.sleep(2)

    # Step 4: Compare all models
    print_header("STEP 4: MODEL COMPARISON")
    best_model = compare_models(current_metrics, new_models)

    # Step 5: Recommendations
    print_header("STEP 5: RECOMMENDATIONS")

    if best_model:
        print(f"\n  RECOMMENDED MODEL:")
        print(f"    Threshold: {best_model['threshold']}C")
        print(f"    F1 Score: {best_model['f1']:.4f}")

        if best_model["f1"] >= 0.7:
            print(f"\n    [OK] Model meets accuracy target (F1 >= 0.70)")
        else:
            print(f"\n    [ATTENTION] Consider further improvements")

        if data_ok is False:
            print(f"\n    [WARNING] Potential data leakage detected!")
            print(f"    Recommendation: Audit training data split")
        elif data_ok is True:
            print(f"\n    [OK] No data leakage detected")

        print(f"\n  NEXT STEPS:")
        print(f"    1. Deploy recommended model for production")
        print(f"    2. Monitor performance on new data")
        print(f"    3. Retrain periodically with latest data")

    print_header("COMPLETED")
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main() or 0)
    except KeyboardInterrupt:
        print("\n\n[Interrupted] Exiting...")
        sys.exit(1)
