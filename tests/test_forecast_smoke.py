"""Smoke test for forecast training and prediction pipeline.

Trains a tiny XGBoost model on synthetic sinusoidal temperature data,
asserts that the model beats naive persistence, and verifies the
skill_score > 0 condition.

This test is self-contained: no real TMD data, no saved model files.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.ml.forecast.features import build_features
from app.ml.forecast.train import train


def _synthetic_df(n_hours: int = 60 * 24) -> pd.DataFrame:
    """Generate synthetic hourly temperature data with a daily sinusoidal cycle.

    temp = 30 + 5 * sin(2π * hour / 24)
    rh   = 70 + 10 * cos(2π * hour / 24)
    """
    hours = pd.date_range("2023-01-01", periods=n_hours, freq="h", tz="UTC")
    hour_of_day = np.arange(n_hours) % 24
    temp = 30 + 5 * np.sin(2 * np.pi * hour_of_day / 24)
    rh = 70 + 10 * np.cos(2 * np.pi * hour_of_day / 24)
    return pd.DataFrame({
        "ts_utc": hours,
        "station_id": "SYNTHETIC",
        "temp_c": temp,
        "rh": rh,
    })


def test_forecast_smoke_beats_persistence():
    """Train on 60 days of synthetic data, horizon=6h, assert skill_score > 0."""
    df = _synthetic_df(n_hours=60 * 24)  # 60 days
    horizon_h = 6

    X, y = build_features(df, horizon_h=horizon_h)
    assert len(X) >= 500, f"Expected >= 500 feature rows, got {len(X)}"

    # Use very few trials for speed in CI
    result = train(X, y, station_id="SYNTHETIC", horizon_h=horizon_h, n_trials=5)

    assert "all_boosters" in result
    assert "metrics" in result
    metrics = result["metrics"]

    assert metrics["mae"] < 3.0, f"MAE {metrics['mae']:.2f} too high for sinusoidal data"
    assert metrics["skill_score"] > 0, (
        f"skill_score={metrics['skill_score']:.3f} must be > 0 (model must beat baselines)"
    )


def test_feature_matrix_min_rows():
    """60 days of hourly data should produce > 1000 feature rows for horizon=6."""
    df = _synthetic_df(n_hours=60 * 24)
    X, y = build_features(df, horizon_h=6)
    assert len(X) > 1000, f"Expected > 1000 feature rows, got {len(X)}"


def test_train_returns_all_metric_keys():
    """train() result dict must contain all expected metric keys."""
    df = _synthetic_df(n_hours=30 * 24)  # 30 days (faster)
    X, y = build_features(df, horizon_h=6)
    result = train(X, y, station_id="SYNTHETIC", horizon_h=6, n_trials=3)
    expected_keys = {"mae", "rmse", "p90_abs_err", "mae_persistence", "mae_climatology", "skill_score"}
    assert expected_keys <= set(result["metrics"].keys()), (
        f"Missing metric keys: {expected_keys - set(result['metrics'].keys())}"
    )
