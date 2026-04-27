#!/usr/bin/env python3
"""Extract metrics from model checkpoint without full dependencies."""

import sys
import os


def main():
    checkpoint_path = "models/heatwave_daily_xgboost_v6.pth"

    print("=" * 70)
    print("MODEL METRICS EXTRACTION")
    print("=" * 70)

    if not os.path.exists(checkpoint_path):
        print(f"\nError: Checkpoint not found at {checkpoint_path}")
        return

    try:
        import torch

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        print("\n[Checkpoint loaded successfully!]")

        if not isinstance(checkpoint, dict):
            print(f"Unexpected checkpoint type: {type(checkpoint)}")
            return

        print(f"Checkpoint keys: {list(checkpoint.keys())}")

        # Get metadata
        meta = checkpoint.get("metadata", {})
        if not meta:
            print("\nNo metadata found in checkpoint")
            return

        # Test metrics
        test_m = meta.get("test_metrics", {})
        val_m = meta.get("val_metrics", {})
        train_m = meta.get("train_metrics", {})

        if test_m:
            print("\n" + "=" * 70)
            print("=== TEST SET METRICS (UNSEEN DATA) ===")
            print("=" * 70)
            for k in ["accuracy", "precision", "recall", "f1", "pr_auc", "brier_score"]:
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
                status = "GOOD"
                color = "\033[92m"  # Green
            elif test_f1 >= 0.5:
                status = "MODERATE"
                color = "\033[93m"  # Yellow
            else:
                status = "NEEDS IMPROVEMENT"
                color = "\033[91m"  # Red

            reset = "\033[0m"
            print(f"  Status: {color}{status}{reset}")
            print(f"  Test F1 Score: {test_f1:.4f}")

            if test_f1 >= 0.7:
                print("\n  ✓ Model meets accuracy target (F1 >= 0.70)")
                print("  ✓ Ready for deployment")
            elif test_f1 >= 0.5:
                print("\n  ~ Model has moderate accuracy")
                print("  ~ Consider improvements:")
                print("    - Hyperparameter tuning")
                print("    - More training data")
                print("    - Feature engineering")
            else:
                print("\n  ✗ Model needs improvement (F1 < 0.5)")
                print("  ✗ Retraining recommended:")
                print("    - Adjust temperature threshold")
                print("    - Increase positive samples")
                print("    - Enable feature engineering")
                print("    - Try different algorithms")

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

        print(f"  F1 Score:    {test_f1:.4f}")
        print(f"  Precision:   {test_precision:.4f}")
        print(f"  Recall:      {test_recall:.4f}")
        print(f"  Accuracy:    {test_m.get('accuracy', 0):.4f}")

        if test_f1 >= 0.7:
            print("\n  Recommendation: KEEP - Model is ready for use")
        elif test_f1 >= 0.5:
            print("\n  Recommendation: IMPROVE - Consider optimizations")
        else:
            print("\n  Recommendation: RETRAIN - Significant improvements needed")

    except Exception as e:
        print(f"\nError loading checkpoint: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
