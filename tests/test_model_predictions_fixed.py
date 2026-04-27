#!/usr/bin/env python3
"""Test model with actual predictions - FIXED VERSION."""

import sys
import os
import numpy as np

print("=" * 70)
print(" TESTING MODEL WITH ACTUAL PREDICTIONS")
print("=" * 70)

try:
    # Load the best model (v6 with 38C threshold)
    print("\n[1] Loading XGBoost v6 model...")

    model_path = "models/heatwave_daily_xgboost_v6.pth"

    # Use torch to load the checkpoint properly
    import torch

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    model = checkpoint.get("sklearn_model")
    metadata = checkpoint.get("metadata", {})
    feature_names = checkpoint.get("feature_names", [])

    threshold = metadata.get("heatwave_temp_threshold", 38)
    print(f"    Model loaded successfully!")
    print(f"    Threshold: {threshold}C")

    # Now test with actual data
    print("\n[2] Loading ERA5 data for testing...")

    sys.path.insert(0, ".")
    from src.data.loader import DataLoader, fill_nan_along_time

    loader = DataLoader()
    full_ds = loader.load_combined()

    # Correct way to get numpy array
    data_raw, stats = loader.prepare_training_data(full_ds, fill_nan=False)

    print(f"    Data shape: {data_raw.shape}")
    n_time = data_raw.shape[0]
    print(f"    Time steps: {n_time} days (~{n_time / 365:.1f} years)")

    # Get last 10 days for testing
    print("\n[3] Making predictions for last 10 days...")

    last_10_days = data_raw[-10:]  # Last 10 days

    # Fill NaN
    for ch in range(last_10_days.shape[1]):
        last_10_days[:, ch, :, :] = fill_nan_along_time(last_10_days[:, ch, :, :])

    predictions = []

    for i in range(10):
        day_data = last_10_days[i : i + 1]  # Single day

        # Extract features
        temp_channel = 1
        temp_data = day_data[0, temp_channel, :, :].copy()
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

            # Get other channels
            z_mean = np.nanmean(day_data[0, 0, :, :])
            z_std = np.nanstd(day_data[0, 0, :, :])
            swvl1_mean = np.nanmean(day_data[0, 2, :, :])

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

    print("\n[4] Model Information:")
    print(f"    Model threshold: {threshold}C")
    print(f"    Heatwave = Temperature >= {threshold}C for 3+ consecutive days")

    print("\n" + "=" * 70)
    print(" TEST COMPLETED SUCCESSFULLY")
    print("=" * 70)

except Exception as e:
    print(f"\n[ERROR] Test failed: {e}")
    import traceback

    traceback.print_exc()
