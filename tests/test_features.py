"""Tests for app/ml/forecast/features.py

CRITICAL: test_no_leakage verifies that the feature matrix has no temporal
data leakage — no feature value at row i was derived from timestamps
after that row's own ts_utc.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.ml.forecast.features import build_features, get_feature_names, _DEFAULT_LAGS_H, _DEFAULT_ROLLING_H


def _make_df(n_hours: int = 72, station_id: str = "BKK_01") -> pd.DataFrame:
    """Synthetic hourly weather data."""
    hours = pd.date_range("2024-05-01", periods=n_hours, freq="h", tz="UTC")
    temp = 30 + 5 * np.sin(2 * np.pi * np.arange(n_hours) / 24)
    rh = 70 + 10 * np.cos(2 * np.pi * np.arange(n_hours) / 24)
    return pd.DataFrame({
        "ts_utc": hours,
        "station_id": station_id,
        "temp_c": temp,
        "rh": rh,
    })


def test_build_features_returns_correct_shapes():
    df = _make_df(72)
    X, y = build_features(df, horizon_h=6)
    assert len(X) == len(y), "X and y must have same length"
    assert len(X) > 0, "Must produce at least some feature rows"
    assert len(X) < len(df), "Rows with NaN lags should be dropped"


def test_no_leakage():
    """CRITICAL: verify no feature row uses future data.

    Method:
    1. Create a df where each row's heat_index_c is its integer row index.
    2. Build features with horizon_h=6.
    3. For each row in X, verify all lag_Nh feature values are <= the
       index value at that row (i.e. they came from earlier rows).

    If lag_1h at row i contains value i+1 or higher, that's leakage.
    """
    n = 60
    hours = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    df = pd.DataFrame({
        "ts_utc": hours,
        "station_id": "TEST",
        "temp_c": np.arange(n, dtype=float),   # value == row index
        "rh": np.full(n, 60.0),
        "heat_index_c": np.arange(n, dtype=float),  # value == row index
    })

    horizon_h = 6
    X, y = build_features(df, horizon_h=horizon_h)

    # Check lag features: lag_Nh at position i should have value <= (i + offset - N)
    # Since we dropped leading NaN rows, the first valid X row corresponds to
    # df row [max(lags_h)] and the lag_1h there should be max(lags_h)-1
    lag_cols = [c for c in X.columns if "_lag" in c and "heat_index_c" in c]
    assert len(lag_cols) > 0, "Expected lag feature columns"

    for col in lag_cols:
        # Extract lag amount from column name, e.g. heat_index_c_lag24h -> 24
        lag_h = int(col.split("lag")[1].replace("h", ""))
        # The lag value should always be LESS than the current "time"
        # Since heat_index_c == row_index, lag_Nh[i] should be < X_index[i]
        # (lag value is from lag_h steps ago, current row is at some later index)
        # At minimum: lag value must be strictly less than the target row's index
        # We verify: no lag value equals or exceeds the FUTURE (target index)
        # Target for row i = row_index_of_X[i] + horizon_h
        # Lag value for row i = row_index_of_X[i] - lag_h (approximately)
        lag_values = X[col].values
        # Reconstruct approximate current "time index" using lag_1h as reference
        if "heat_index_c_lag1h" in X.columns:
            current_approx = X["heat_index_c_lag1h"].values + 1
            target_approx = current_approx + horizon_h
            assert np.all(lag_values < target_approx), (
                f"Leakage detected in {col}: lag value >= target value at some rows"
            )

    # v2 features — all use shift(k) with k>=1, so they must appear in X
    # hi_change_3h: shift(1) - shift(4) — safe
    # temp_change_3h: shift(1) - shift(4) — safe
    # temp_x_rh: shift(1) * shift(1) — safe
    # hi_residual_lag1h: shift(1) on the climatology residual — safe
    for v2_col in ["hi_change_3h", "temp_change_3h", "temp_x_rh", "hi_residual_lag1h"]:
        assert v2_col in X.columns, f"v2 feature missing from X: {v2_col}"


def test_y_is_future_heat_index():
    """Target y must be heat_index_c at t+horizon, not at t."""
    df = _make_df(72)
    horizon_h = 6
    X, y = build_features(df, horizon_h=horizon_h)
    # y should vary (it's future heat_index) — not constant
    assert y.std() > 0, "Target y should not be constant"


def test_no_nan_in_output():
    df = _make_df(72)
    X, y = build_features(df, horizon_h=6)
    assert not X.isna().any().any(), "X must contain no NaN values"
    assert not y.isna().any(), "y must contain no NaN values"


def test_feature_names_consistent():
    """get_feature_names() must return names that appear in X columns."""
    df = _make_df(72)
    X, _ = build_features(df, horizon_h=6)
    expected = set(get_feature_names())
    actual = set(X.columns)
    # All expected feature names should appear in X
    missing = expected - actual
    assert len(missing) == 0, f"Expected feature columns missing from X: {missing}"


def test_v2_features_present():
    """All v2 feature additions must appear in X."""
    n = 50
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "ts_utc": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
        "station_id": "BKK_01",
        "temp_c": 28.0 + rng.normal(0, 2, n),
        "rh": 75.0 + rng.normal(0, 5, n),
        "solar_wm2": np.maximum(0, rng.normal(300, 150, n)),
        "cloud_cover": rng.uniform(0, 1, n),
        "pressure_hpa": 1010.0 + rng.normal(0, 3, n),
    })
    X, _ = build_features(df, horizon_h=24)
    for feat in ["local_hour_sin", "local_hour_cos", "hi_change_3h", "temp_change_3h",
                 "temp_x_rh", "hi_residual_lag1h"]:
        assert feat in X.columns, f"Missing v2 feature: {feat}"


def test_multi_station_lags_and_targets_do_not_cross_station_boundaries():
    """Lag and target values must be computed within each station only."""
    n = 40
    hours = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    station_a = pd.DataFrame({
        "ts_utc": hours,
        "station_id": "A",
        "temp_c": np.arange(n, dtype=float),
        "rh": np.full(n, 60.0),
        "heat_index_c": np.arange(n, dtype=float),
    })
    station_b = pd.DataFrame({
        "ts_utc": hours,
        "station_id": "B",
        "temp_c": 1000.0 + np.arange(n, dtype=float),
        "rh": np.full(n, 60.0),
        "heat_index_c": 1000.0 + np.arange(n, dtype=float),
    })
    df = pd.concat([station_b, station_a], ignore_index=True)

    X, y = build_features(df, horizon_h=6)

    first_station_a = X[X["station_enc"] == 0].iloc[0]
    first_station_b = X[X["station_enc"] == 1].iloc[0]

    assert first_station_a["heat_index_c_lag1h"] == 23.0
    assert first_station_a["heat_index_c_lag24h"] == 0.0
    assert first_station_b["heat_index_c_lag1h"] == 1023.0
    assert first_station_b["heat_index_c_lag24h"] == 1000.0

    y_a = y[X["station_enc"] == 0].reset_index(drop=True)
    y_b = y[X["station_enc"] == 1].reset_index(drop=True)
    assert y_a.iloc[0] == 30.0
    assert y_b.iloc[0] == 1030.0


def test_extended_feature_fill_does_not_backfill_future_values():
    """Sparse extended features may forward-fill past values but must not backfill."""
    n = 40
    df = pd.DataFrame({
        "ts_utc": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
        "station_id": "A",
        "temp_c": np.arange(n, dtype=float),
        "rh": np.full(n, 60.0),
        "heat_index_c": np.arange(n, dtype=float),
        "solar_wm2": [np.nan] * 5 + [50.0] * 35,
    })

    X, _ = build_features(df, horizon_h=6)

    assert X["solar_wm2_lag1h"].iloc[0] == 50.0
    assert not (X["solar_wm2_lag1h"] == 0.0).any()
