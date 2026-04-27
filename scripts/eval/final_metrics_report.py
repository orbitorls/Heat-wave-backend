#!/usr/bin/env python3
"""Extract all metrics from checkpoint (no unicode)."""

import zipfile
import pickle
import io
import sys


class FakeModule:
    """Fake module for unpickling."""

    def __getattr__(self, name):
        return FakeModule()

    def __call__(self, *args, **kwargs):
        return FakeModule()


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if "xgboost" in module or "sklearn" in module:
            return FakeModule
        return super().find_class(module, name)


def main():
    checkpoint_path = "models/heatwave_daily_xgboost_v6.pth"

    print("=" * 70)
    print("MODEL ACCURACY REPORT - XGBoost v6")
    print("=" * 70)

    try:
        with zipfile.ZipFile(checkpoint_path, "r") as zf:
            pkl_path = "heatwave_daily_xgboost_v6/data.pkl"
            with zf.open(pkl_path) as f:
                data_bytes = f.read()

            unpickler = CustomUnpickler(io.BytesIO(data_bytes))
            checkpoint = unpickler.load()

            if not isinstance(checkpoint, dict):
                print(f"Error: Unexpected type {type(checkpoint)}")
                return 1

            meta = checkpoint.get("metadata", {})
            test_m = meta.get("test_metrics", {})
            val_m = meta.get("val_metrics", {})
            train_m = meta.get("train_metrics", {})

            # TEST METRICS
            print("\n### TEST SET PERFORMANCE (UNSEEN DATA) ###")
            print("-" * 70)
            if test_m:
                for k in ["accuracy", "precision", "recall", "f1"]:
                    if k in test_m:
                        val = test_m[k]
                        if isinstance(val, (int, float)):
                            print(f"  Test {k:12s}: {val:.4f} ({val * 100:.2f}%)")
                test_f1 = test_m.get("f1", 0)
            else:
                print("  No test metrics available")
                test_f1 = 0

            # TRAINING METRICS
            print("\n### TRAINING SET PERFORMANCE ###")
            print("-" * 70)
            if train_m:
                for k in ["accuracy", "precision", "recall", "f1"]:
                    if k in train_m:
                        val = train_m[k]
                        if isinstance(val, (int, float)):
                            print(f"  Train {k:12s}: {val:.4f} ({val * 100:.2f}%)")

            # VALIDATION METRICS
            print("\n### VALIDATION SET PERFORMANCE ###")
            print("-" * 70)
            if val_m:
                for k in ["accuracy", "precision", "recall", "f1"]:
                    if k in val_m:
                        val = val_m[k]
                        if isinstance(val, (int, float)):
                            print(f"  Val {k:14s}: {val:.4f} ({val * 100:.2f}%)")

            # OVERFITTING CHECK
            if train_m and test_m:
                train_f1 = train_m.get("f1", 0)
                test_f1 = test_m.get("f1", 0)
                gap = train_f1 - test_f1

                print("\n### OVERFITTING CHECK ###")
                print("-" * 70)
                print(f"  Train F1: {train_f1:.4f}")
                print(f"  Test F1:  {test_f1:.4f}")
                print(f"  Gap:      {gap:.4f}")

                if abs(gap) < 0.05:
                    print("  Status: Good generalization (gap < 5%)")
                elif abs(gap) < 0.10:
                    print("  Status: Moderate overfitting (gap 5-10%)")
                else:
                    print("  Status: SEVERE OVERFITTING (gap > 10%)")

            # CLASS BALANCE
            print("\n### CLASS BALANCE ###")
            print("-" * 70)
            for k in ["train_positive_rate", "val_positive_rate", "test_positive_rate"]:
                if k in meta:
                    val = meta[k]
                    if isinstance(val, (int, float)):
                        print(f"  {k}: {val * 100:.2f}%")

            # CONFIGURATION
            print("\n### MODEL CONFIGURATION ###")
            print("-" * 70)
            for k in [
                "heatwave_temp_threshold",
                "heatwave_min_duration",
                "n_estimators",
                "max_depth",
                "learning_rate",
            ]:
                if k in meta:
                    print(f"  {k}: {meta[k]}")

            # FEATURE IMPORTANCE
            importance = meta.get("feature_importance", {})
            if importance:
                print("\n### TOP 10 FEATURES ###")
                print("-" * 70)
                for i, (feat, imp) in enumerate(
                    sorted(importance.items(), key=lambda x: -x[1])[:10], 1
                ):
                    print(f"  {i:2d}. {feat:20s}: {imp:.4f} ({imp * 100:.1f}%)")

            # OVERALL ASSESSMENT
            print("\n### OVERALL ASSESSMENT ###")
            print("=" * 70)

            if test_f1 >= 0.95:
                print(f"F1 Score: {test_f1:.4f} - EXCELLENT (>= 0.95)")
                print("\nWARNING: Very high F1 score detected!")
                print("This might indicate:")
                print("  1. Overfitting to test data")
                print("  2. Data leakage (test data seen during training)")
                print("  3. Very easy classification task")
                print("  4. Class imbalance issues")
                print("\nRecommendation: Investigate training/validation split")
            elif test_f1 >= 0.7:
                print(f"F1 Score: {test_f1:.4f} - GOOD (>= 0.70)")
                print("\nModel meets accuracy target.")
                print("Ready for deployment.")
            elif test_f1 >= 0.5:
                print(f"F1 Score: {test_f1:.4f} - MODERATE (0.5 - 0.7)")
                print("\nConsider improvements:")
                print("  - Hyperparameter tuning")
                print("  - More training data")
                print("  - Feature engineering")
            else:
                print(f"F1 Score: {test_f1:.4f} - NEEDS IMPROVEMENT (<0.5)")
                print("\nRetraining recommended:")
                print("  - Adjust temperature threshold")
                print("  - Increase positive samples")
                print("  - Enable feature engineering")

            # CREATED DATE
            if "created_at" in meta:
                print(f"\n\nModel created: {meta['created_at']}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
