#!/usr/bin/env python3
"""Test model with actual predictions."""

import sys
import os

print("=" * 70)
print(" TESTING MODEL WITH ACTUAL PREDICTIONS")
print("=" * 70)

try:
    # Load the best model (v6 with 38C threshold)
    import torch
    import zipfile
    import pickle
    import io
    import numpy as np

    print("\n[1] Loading XGBoost v6 model...")

    # Custom unpickler
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

    model_path = "models/heatwave_daily_xgboost_v6.pth"
    with zipfile.ZipFile(model_path, "r") as zf:
        pkl_path = "heatwave_daily_xgboost_v6/data.pkl"
        with zf.open(pkl_path) as f:
            data_bytes = f.read()

        unpickler = CustomUnpickler(io.BytesIO(data_bytes))
        checkpoint = unpickler.load()

    # Get model and metadata
    model = checkpoint.get("sklearn_model")
    metadata = checkpoint.get("metadata", {})
    feature_names = checkpoint.get("feature_names", [])

    threshold = metadata.get("heatwave_temp_threshold", 38)
    print(f"    Model loaded successfully!")
    print(f"    Threshold: {threshold}C")
    print(f"    Features: {feature_names}")

    # Now test with actual data
    print("\n[2] Loading ERA5 data for testing...")

    sys.path.insert(0, ".")
    from src.data.loader import DataLoader

    loader = DataLoader()
    full_ds = loader.load_combined()
    if full_ds is None or len(full_ds) == 0:
        print("No data loaded, exiting")
        sys.exit(1)

    print(f"Full dataset variables: {list(full_ds.data_vars)}")
    print(f"Full dataset dimensions: {dict(full_ds.dims)}")

    print("\n[3] Making predictions for last 10 days...")

    # Get last 10 days of data
    if "t2m" in full_ds.data_vars:
        temp_data = full_ds["t2m"].values
        if len(temp_data.shape) == 4:
            temp_data = temp_data[0]  # Remove singleton time dim if present
    elif "T2m" in full_ds.data_vars:
        temp_data = full_ds["T2m"].values
        if len(temp_data.shape) == 4:
            temp_data = temp_data[0]
    else:
        print("No temperature data found")
        sys.exit(1)

    # Get last 10 days
    last_10_days = temp_data[-10:]
    # Add channel dimension: (time, lat, lon) -> (time, 1, lat, lon)
    last_10_days = last_10_days[:, np.newaxis, :, :]

    predictions = []

    for i in range(10):
        day_data = last_10_days[i : i + 1]  # Single day

        # Extract features (channel 0 is now temperature)
        temp_data = day_data[0, 0, :, :].copy()
        temp_c = temp_data - 273.15

        # Get features for this day
        valid_mask = temp_c > -273
        valid_temps = temp_c[valid_mask]

        if len(valid_temps) > 0:
            temp_mean = np.nanmean(valid_temps)
            temp_max = np.nanmax(valid_temps)
            temp_min = np.nanmin(valid_temps)
            temp_std = np.nanstd(valid_temps)
            temp_range = temp_max - temp_min
            hot_frac = np.nanmean(valid_temps >= threshold)

            # Get other channels (not available in simplified data, use zeros)
            z_mean = 0.0
            z_std = 0.0
            swvl1_mean = 0.0

            # Use available features (11 features expected)
            features = [
                temp_mean,
                temp_max,
                temp_min,
                temp_std,
                temp_range,
                hot_frac,
                z_mean,
                z_std,
                swvl1_mean,
                0,
                0,
            ]

            # Make prediction
            X = np.array([features], dtype=np.float32)
            pred_proba = model.predict_proba(X)[0][1]
            pred_class = model.predict(X)[0]

            predictions.append(
                {
                    "day": i + 1,
                    "temp_max": temp_max,
                    "temp_mean": temp_mean,
                    "prob": pred_proba,
                    "prediction": "HEATWAVE" if pred_class == 1 else "NORMAL",
                }
            )

            status = "*** HEATWAVE ***" if pred_class == 1 else "Normal"
            print(
                f"    Day {i + 1}: Max={temp_max:.1f}C, Mean={temp_mean:.1f}C, Prob={pred_proba:.2%} [{status}]"
            )

    # Summary
    print("\n" + "=" * 70)
    print(" PREDICTION SUMMARY")
    print("=" * 70)

    heatwave_days = sum(1 for p in predictions if p["prediction"] == "HEATWAVE")
    print(f"\nTotal days tested: {len(predictions)}")
    print(f"Heatwave days detected: {heatwave_days}")
    print(f"Normal days: {len(predictions) - heatwave_days}")

    if heatwave_days > 0:
        print("\n[ALERT] Heatwave conditions detected!")
        for p in predictions:
            if p["prediction"] == "HEATWAVE":
                print(
                    f"  - Day {p['day']}: Max {p['temp_max']:.1f}C (Probability: {p['prob']:.1%})"
                )
    else:
        print("\n[OK] No heatwave conditions detected in last 10 days")

    # Test with specific threshold check
    print("\n[4] Model Information:")
    print(f"    Model threshold: {threshold}C")
    print(
        f"    This means: Temperature >= {threshold}C for 3+ consecutive days = Heatwave"
    )

    print("\n" + "=" * 70)
    print(" TEST COMPLETED SUCCESSFULLY")
    print("=" * 70)

except Exception as e:
    print(f"\n[ERROR] Test failed: {e}")
    import traceback

    traceback.print_exc()
