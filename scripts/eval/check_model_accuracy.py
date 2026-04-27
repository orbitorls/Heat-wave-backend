#!/usr/bin/env python3
"""Check model accuracy and decide if retraining is needed."""

import torch
import os
import glob


def check_model_metrics():
    print("=" * 70)
    print("MODEL ACCURACY EVALUATION")
    print("=" * 70)

    results = []

    # Check XGBoost models
    xgb_files = glob.glob("models/heatwave_daily_xgboost_v*.pth")
    for xgb_path in sorted(xgb_files):
        try:
            ckpt = torch.load(xgb_path, map_location="cpu", weights_only=False)
            meta = ckpt.get("metadata", {})
            test_m = meta.get("test_metrics", {})
            val_m = meta.get("val_metrics", {})
            train_m = meta.get("train_metrics", {})

            print(f"\n[{os.path.basename(xgb_path)}]")
            print("-" * 50)
            print(f"Model Type: {ckpt.get('model_type', 'unknown')}")
            print(f"Threshold: {meta.get('heatwave_temp_threshold', 'N/A')}C")
            print(
                f"Data Split: {meta.get('train_ratio', 0) * 100:.0f}%/{meta.get('val_ratio', 0) * 100:.0f}%/{meta.get('test_ratio', 0) * 100:.0f}% (train/val/test)"
            )

            # Positive rates (class balance)
            print(f"\nClass Balance:")
            print(f"  Train positive: {meta.get('train_positive_rate', 0) * 100:.1f}%")
            print(f"  Val positive: {meta.get('val_positive_rate', 0) * 100:.1f}%")
            print(f"  Test positive: {meta.get('test_positive_rate', 0) * 100:.1f}%")

            # Training metrics
            print(f"\nTraining Metrics:")
            for metric in ["accuracy", "precision", "recall", "f1"]:
                val = train_m.get(metric)
                if val is not None:
                    print(f"  {metric.capitalize():12s}: {val:.4f}")

            # Validation metrics
            print(f"\nValidation Metrics:")
            for metric in ["accuracy", "precision", "recall", "f1"]:
                val = val_m.get(metric)
                if val is not None:
                    print(f"  {metric.capitalize():12s}: {val:.4f}")

            # Test metrics (most important)
            print(f"\n===TEST SET METRICS (UNSEEN DATA) ===")
            for metric in ["accuracy", "precision", "recall", "f1"]:
                val = test_m.get(metric)
                if val is not None:
                    print(f"  {metric.capitalize():12s}: {val:.4f}")

            # Feature importance
            importance = meta.get("feature_importance", {})
            if importance is not None:
                print(f"\nTop 5 Feature Importance:")
                for feat, imp in sorted(importance.items(), key=lambda x: -x[1])[:5]:
                    print(f"  {feat}: {imp:.4f}")

            # Assessment
            test_f1 = test_m.get("f1", 0)
            if isinstance(test_f1, (int, float)):
                print(f"\n[ASSESSMENT]", end=" ")
                if test_f1 >= 0.7:
                    print(f"GOOD - F1 = {test_f1:.4f} >= 0.70")
                elif test_f1 >= 0.5:
                    print(f"MODERATE - F1 = {test_f1:.4f} (between 0.5 and 0.7)")
                else:
                    print(f"NEEDS IMPROVEMENT - F1 = {test_f1:.4f} < 0.5")

                results.append(
                    {
                        "model": os.path.basename(xgb_path),
                        "test_f1": test_f1,
                        "test_precision": test_m.get("precision", 0),
                        "test_recall": test_m.get("recall", 0),
                        "test_accuracy": test_m.get("accuracy", 0),
                        "assessment": "GOOD"
                        if test_f1 >= 0.7
                        else ("MODERATE" if test_f1 >= 0.5 else "NEEDS IMPROVEMENT"),
                    }
                )

            print(f"\nCreated: {meta.get('created_at', 'N/A')}")
            print(f"Training Time: {meta.get('training_time_seconds', 'N/A')}s")

        except Exception as e:
            print(f"\nError loading {xgb_path}: {e}")

    # Check ConvLSTM models
    convlstm_files = glob.glob("models/heatwave_model_checkpoint_v*.pth")
    for convlstm_path in sorted(convlstm_files):
        try:
            ckpt = torch.load(convlstm_path, map_location="cpu", weights_only=False)
            print(f"\n[{os.path.basename(convlstm_path)}]")
            print("-" * 50)
            print(f"Checkpoint keys: {list(ckpt.keys())[:10]}")
            meta = ckpt.get("metadata", {})
            if meta is not None:
                print(f"Metadata available: {len(meta)} keys")
                # Try to find metrics
                for key in [
                    "test_f1",
                    "test_accuracy",
                    "test_precision",
                    "test_recall",
                ]:
                    if key in meta:
                        print(f"  {key}: {meta[key]}")
        except Exception as e:
            print(f"\nError loading {convlstm_path}: {e}")

    # Summary
    if results is not None:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        for r in results:
            print(f"\n{r['model']}:")
            print(
                f"  F1: {r['test_f1']:.4f} | Precision: {r['test_precision']:.4f} | Recall: {r['test_recall']:.4f}"
            )
            print(f"  Accuracy: {r['test_accuracy']:.4f}")
            print(f"  Status: {r['assessment']}")

        # Recommendation
        best = max(results, key=lambda x: x["test_f1"])
        print(f"\nBest Model: {best['model']} (F1={best['test_f1']:.4f})")

        if best["test_f1"] < 0.7:
            print("\n[RECOMMENDATION] Model accuracy is below target (F1 < 0.70)")
            print("  Consider retraining with:")
            print("  - Different threshold")
            print("  - More data")
            print("  - Different features")
            print("  - Hyperparameter tuning")
        else:
            print("\n[RECOMMENDATION] Model meets accuracy target (F1 >= 0.70)")

    return results


if __name__ == "__main__":
    check_model_metrics()
