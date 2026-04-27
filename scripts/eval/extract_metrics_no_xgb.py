#!/usr/bin/env python3
"""Extract metrics from checkpoint without xgboost dependency."""

import zipfile
import pickle
import io
import sys


class FakeXGBoost:
    """Fake module to allow unpickling."""

    def __getattr__(self, name):
        return FakeXGBoost()

    def __call__(self, *args, **kwargs):
        return FakeXGBoost()


class CustomUnpickler(pickle.Unpickler):
    """Custom unpickler that replaces xgboost with fake module."""

    def find_class(self, module, name):
        # Replace xgboost classes with fake
        if "xgboost" in module or "sklearn" in module:
            return FakeXGBoost
        return super().find_class(module, name)


def main():
    checkpoint_path = "models/heatwave_daily_xgboost_v6.pth"

    print("=" * 70)
    print("MODEL METRICS EXTRACTION (WITHOUT XGBOOST)")
    print("=" * 70)

    if not hasattr(sys, "path"):
        sys.path = []

    try:
        # Read checkpoint as zipfile
        with zipfile.ZipFile(checkpoint_path, "r") as zf:
            # Get data.pkl
            pkl_path = "heatwave_daily_xgboost_v6/data.pkl"

            with zf.open(pkl_path) as f:
                data_bytes = f.read()

            # Use custom unpickler
            unpickler = CustomUnpickler(io.BytesIO(data_bytes))
            checkpoint = unpickler.load()

            if not isinstance(checkpoint, dict):
                print(f"\nUnexpected checkpoint type: {type(checkpoint)}")
                print(f"Available attributes: {dir(checkpoint)[:20]}")
                return

            print("\n[Checkpoint loaded successfully!]")
            print(f"Checkpoint keys: {list(checkpoint.keys())}")

            # Get metadata
            meta = checkpoint.get("metadata", {})
            if not meta:
                print("\nNo metadata found in checkpoint")
                print(f"Available keys: {list(checkpoint.keys())}")
                return

            # Test metrics
            test_m = meta.get("test_metrics", {})
            val_m = meta.get("val_metrics", {})
            train_m = meta.get("train_metrics", {})

            if test_m:
                print("\n" + "=" * 70)
                print("=== TEST SET METRICS (UNSEEN DATA) ===")
                print("=" * 70)
                for k in [
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "pr_auc",
                    "brier_score",
                ]:
                    if k in test_m:
                        val = test_m[k]
                        if isinstance(val, (int, float)):
                            print(f"  {k.capitalize():15s}: {val:.4f}")

                # Assessment
                test_f1 = test_m.get("f1", 0)
                print("\n" + "=" * 70)
                print("ASSESSMENT:")
                print("=" * 70)

                if test_f1 >= 0.7:
                    status = "✓ GOOD"
                    recommendation = "Model is ready for use"
                elif test_f1 >= 0.5:
                    status = "~ MODERATE"
                    recommendation = "Consider optimizations"
                else:
                    status = "✗ NEEDS IMPROVEMENT"
                    recommendation = "Retraining recommended"

                print(f"  Status: {status}")
                print(f"  Test F1 Score: {test_f1:.4f}")
                print(f"\n  Recommendation: {recommendation}")

            # Train/Val metrics
            if train_m:
                print("\n" + "=" * 70)
                print("TRAINING METRICS:")
                print("=" * 70)
                for k in ["accuracy", "precision", "recall", "f1"]:
                    if k in train_m:
                        val = train_m[k]
                        if isinstance(val, (int, float)):
                            print(f"  Train {k}: {val:.4f}")

            if val_m:
                print("\n" + "=" * 70)
                print("VALIDATION METRICS:")
                print("=" * 70)
                for k in ["accuracy", "precision", "recall", "f1"]:
                    if k in val_m:
                        val = val_m[k]
                        if isinstance(val, (int, float)):
                            print(f"  Val {k}: {val:.4f}")

            # Config
            print("\n" + "=" * 70)
            print("CONFIGURATION:")
            print("=" * 70)
            for k in [
                "heatwave_temp_threshold",
                "heatwave_min_duration",
                "n_estimators",
                "max_depth",
                "learning_rate",
            ]:
                if k in meta:
                    print(f"  {k}: {meta[k]}")

            # Class balance
            print("\n" + "=" * 70)
            print("CLASS BALANCE:")
            print("=" * 70)
            for k in ["train_positive_rate", "val_positive_rate", "test_positive_rate"]:
                if k in meta:
                    val = meta[k]
                    if isinstance(val, (int, float)):
                        print(f"  {k}: {val * 100:.2f}%")

            # Feature importance
            importance = meta.get("feature_importance", {})
            if importance:
                print("\n" + "=" * 70)
                print("TOP 5 FEATURES:")
                print("=" * 70)
                for feat, imp in sorted(importance.items(), key=lambda x: -x[1])[:5]:
                    print(f"  {feat}: {imp:.4f}")

            # Training info
            print("\n" + "=" * 70)
            print("TRAINING INFO:")
            print("=" * 70)
            for k in ["created_at", "training_time_seconds", "model_type"]:
                if k in meta:
                    print(f"  {k}: {meta[k]}")

            # Summary
            print("\n" + "=" * 70)
            print("SUMMARY:")
            print("=" * 70)
            test_f1 = test_m.get("f1", 0) if test_m else 0
            test_precision = test_m.get("precision", 0) if test_m else 0
            test_recall = test_m.get("recall", 0) if test_m else 0
            test_accuracy = test_m.get("accuracy", 0) if test_m else 0

            print(f"  F1 Score:    {test_f1:.4f}")
            print(f"  Precision:   {test_precision:.4f}")
            print(f"  Recall:      {test_recall:.4f}")
            print(f"  Accuracy:    {test_accuracy:.4f}")

            if test_f1 >= 0.7:
                print("\n  ✓ Model meets accuracy target (F1 >= 0.70)")
                print("  ✓ Ready for deployment")
            elif test_f1 >= 0.5:
                print("\n  ~ Model has moderate accuracy")
                print("  ~ Suggestions: hyperparameter tuning, more data")
            else:
                print("\n  ✗ Model needs improvement (F1 < 0.5)")
                print(
                    "  ✗ Suggestions: adjust threshold, more data, feature engineering"
                )

    except Exception as e:
        print(f"\nError loading checkpoint: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
