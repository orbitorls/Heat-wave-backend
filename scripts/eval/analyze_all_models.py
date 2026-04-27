#!/usr/bin/env python3
"""Comprehensive model analysis."""

import os
import glob

print("=" * 70)
print(" COMPREHENSIVE MODEL ANALYSIS")
print("=" * 70)

# List all models
models = sorted(glob.glob("models/heatwave_daily_xgboost_v*.pth"))
print(f"\n[Models found]: {len(models)}")

# Load and analyze each model
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


results = []

for model_path in models:
    version = model_path.split("_v")[-1].replace(".pth", "")
    print(f"\n{'=' * 70}")
    print(f" MODEL v{version}")
    print(f"{'=' * 70}")

    try:
        with zipfile.ZipFile(model_path, "r") as zf:
            pkl_path = (
                model_path.replace(".pth", "/data.pkl")
                .replace("models\\", "")
                .replace("models/", "")
            )

            with zf.open(pkl_path) as f:
                data_bytes = f.read()

            unpickler = CustomUnpickler(io.BytesIO(data_bytes))
            checkpoint = unpickler.load()

            meta = checkpoint.get("metadata", {})
            test_m = meta.get("test_metrics", {})
            val_m = meta.get("val_metrics", {})
            train_m = meta.get("train_metrics", {})

            threshold = meta.get("heatwave_temp_threshold", "N/A")
            train_pos = meta.get("train_positive_rate", 0)
            val_pos = meta.get("val_positive_rate", 0)
            test_pos = meta.get("test_positive_rate", 0)

            print(f"\n  Configuration:")
            print(f"    Threshold: {threshold}C")

            print(f"\n  Class Balance:")
            print(f"    Train: {train_pos * 100:.1f}%")
            print(f"    Val:   {val_pos * 100:.1f}%")
            print(f"    Test:  {test_pos * 100:.1f}%")

            print(f"\n  Performance:")
            print(
                f"    Train - Acc: {train_m.get('accuracy', 0):.4f}, P: {train_m.get('precision', 0):.4f}, R: {train_m.get('recall', 0):.4f}, F1: {train_m.get('f1', 0):.4f}"
            )
            print(
                f"    Val   - Acc: {val_m.get('accuracy', 0):.4f}, P: {val_m.get('precision', 0):.4f}, R: {val_m.get('recall', 0):.4f}, F1: {val_m.get('f1', 0):.4f}"
            )
            print(
                f"    Test  - Acc: {test_m.get('accuracy', 0):.4f}, P: {test_m.get('precision', 0):.4f}, R: {test_m.get('recall', 0):.4f}, F1: {test_m.get('f1', 0):.4f}"
            )

            # Overfitting analysis
            train_f1 = train_m.get("f1", 0)
            test_f1 = test_m.get("f1", 0)
            gap = train_f1 - test_f1

            print(f"\n  Overfitting Analysis:")
            print(f"    Train F1: {train_f1:.4f}")
            print(f"    Test F1:  {test_f1:.4f}")
            print(f"    Gap:      {abs(gap) * 100:.1f}%")

            if abs(gap) < 0.05:
                status = "GOOD"
            elif abs(gap) < 0.10:
                status = "MODERATE"
            else:
                status = "SEVERE"

            print(f"    Status:   {status}")

            # Data leakage warning
            if test_pos < 0.01:
                print(
                    f"\n  WARNING: Very low positive rate in test set ({test_pos * 100:.1f}%)"
                )
                print(f"  This may indicate insufficient positive samples!")

            results.append(
                {
                    "version": version,
                    "threshold": threshold,
                    "test_pos": test_pos,
                    "train_f1": train_f1,
                    "test_f1": test_f1,
                    "gap": gap,
                    "status": status,
                }
            )

    except Exception as e:
        print(f"  Error loading: {e}")

# Summary comparison
print("\n" + "=" * 70)
print(" SUMMARY COMPARISON")
print("=" * 70)

print(
    "\n{:<10} {:<12} {:<10} {:<10} {:<10} {:<12} {:<12}".format(
        "Version", "Threshold", "Test Pos%", "Train F1", "Test F1", "Gap%", "Status"
    )
)
print("-" * 76)

for r in results:
    print(
        "{:<10} {:<12} {:<10.1f} {:<10.4f} {:<10.4f} {:<12.1f} {:<12}".format(
            f"v{r['version']}",
            f"{r['threshold']}C",
            r["test_pos"] * 100,
            r["train_f1"],
            r["test_f1"],
            abs(r["gap"]) * 100,
            r["status"],
        )
    )

# Best model recommendation
print("\n" + "=" * 70)
print(" RECOMMENDATION FOR THAILAND")
print("=" * 70)

# Filter out models with very low test positive rate
valid_models = [r for r in results if r["test_pos"] > 0.01]

if valid_models:
    best = max(valid_models, key=lambda x: x["test_f1"])

    print(f"\n  Best Model: v{best['version']}")
    print(f"  Threshold: {best['threshold']}C")
    print(f"  Test F1:   {best['test_f1']:.4f}")
    print(f"  Overfitting: {best['status']}")

    if best["test_f1"] >= 0.7:
        print(f"\n  [OK] Model meets accuracy target (F1 >= 0.70)")
    else:
        print(f"\n  [ATTENTION] Model below target")

    # Use case recommendation
    print(f"\n  Recommended Use Cases:")
    if best["threshold"] <= 35:
        print(f"    - General heatwave warning (threshold {best['threshold']}C)")
        print(f"    - Moderate heat detection")
    elif best["threshold"] <= 38:
        print(f"    - Severe heat warning (threshold {best['threshold']}C)")
    else:
        print(f"    - Extreme heat emergency (threshold {best['threshold']}C)")
else:
    print("\n  [WARNING] All models have very low positive rates in test set!")
    print("  This indicates insufficient heatwave samples.")

print("\n" + "=" * 70)
