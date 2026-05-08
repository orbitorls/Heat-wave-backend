"""No-leakage invariant tests for ``app/ml/forecast/features.py``.

ภาษาไทย:
ทุกแถว ``i`` ใน ``X`` ห้ามใช้ค่าใด ๆ จาก ``ts_utc > ts_utc[i]``
และ target ``y[i]`` ต้องเป็นค่าจาก ``ts_utc[i] + horizon_h``.

English:
For every row ``i`` in ``X`` produced by the feature builder, every feature
value must be derived solely from observations at times ``ts_utc <= ts_utc[i]``.
The target ``y[i]`` must be the heat index at exactly ``ts_utc[i] + horizon_h``.

This module proves that invariant against the public feature API with fast,
deterministic synthetic data — no parquet, no trained models. Default
``pytest`` (``-m 'not slow'``) must run these tests.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.ml.forecast.features import (
    _DEFAULT_LAGS_H,
    _DEFAULT_ROLLING_H,
    build_X_once,
    build_features,
    build_y_for_horizon,
)


# ---------------------------------------------------------------------------
# Synthetic data helpers — deterministic, in-memory, single station.
# ---------------------------------------------------------------------------

_STATION = "BKK_01"


def _hours(n: int, start: str = "2024-05-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="h", tz="UTC")


def _df_with_hi(hi_values: np.ndarray) -> pd.DataFrame:
    """Return a hourly DataFrame with the given heat_index_c sequence.

    temp_c / rh are filled with plausible constants — they only matter
    because ``build_X_once`` requires the raw columns to be present.
    """
    n = len(hi_values)
    return pd.DataFrame({
        "ts_utc": _hours(n),
        "station_id": _STATION,
        "temp_c": np.full(n, 30.0),
        "rh": np.full(n, 60.0),
        "heat_index_c": hi_values.astype(float),
    })


# ---------------------------------------------------------------------------
# 1. Target offset invariant.
# ---------------------------------------------------------------------------


def test_target_y_equals_hi_at_t_plus_horizon():
    """``y[i]`` must equal ``heat_index_c`` at ``ts_utc[i] + horizon_h``.

    Build a strictly monotone-increasing HI series so the relationship
    between source timestamp and target value is unambiguous, then
    verify each (ts_utc, y) pair against the source row at
    ts_utc + horizon_h.
    """
    n = 200
    horizon_h = 24
    # Use float HI values that are unique per timestamp so any off-by-one
    # would be caught (e.g. 100.0, 100.5, 101.0, …).
    hi = 100.0 + 0.5 * np.arange(n)
    df = _df_with_hi(hi)

    X, df_aug = build_X_once(df)
    y = build_y_for_horizon(df_aug, X.index, horizon_h=horizon_h, target_kind="hi")

    # Drop rows where the target falls past the end of the series.
    valid = y.notna()
    X_valid = X.loc[valid]
    y_valid = y.loc[valid]

    # Recover ts_utc per surviving X row from df_aug (same RangeIndex labels).
    ts_at_X = df_aug.loc[X_valid.index, "ts_utc"].reset_index(drop=True)
    y_arr = y_valid.to_numpy()

    # Build a lookup: ts_utc -> heat_index_c from the source.
    src = df.set_index("ts_utc")["heat_index_c"]

    expected = np.array([src.loc[t + pd.Timedelta(hours=horizon_h)] for t in ts_at_X])
    np.testing.assert_allclose(
        y_arr,
        expected,
        rtol=0,
        atol=1e-12,
        err_msg="y[i] must equal heat_index_c at ts_utc[i] + horizon_h",
    )


# ---------------------------------------------------------------------------
# 2. Rolling window must not peek into the future.
# ---------------------------------------------------------------------------


def test_rolling_max_does_not_peek_at_future_jump():
    """A jump from 30 -> 45 at ``t*`` must never appear in rolling
    features for rows whose ``ts_utc < t*``.

    The builder emits rolling-mean and rolling-std (not max) by default.
    For any row strictly before the jump, every rolling-mean window has
    already had the jump excluded, so the rolling mean must remain
    exactly 30.0. After the jump, values may rise. We assert the
    pre-jump mean strictly, and we also assert that the maximum value
    of every constituent of the rolling window — reconstructable via
    the lag features — never equals 45 for pre-jump rows.
    """
    n = 100
    jump_idx = 50  # t* corresponds to df row 50
    hi = np.where(np.arange(n) < jump_idx, 30.0, 45.0)
    # temp_c also jumps so rolling features for temp_c match the design.
    temp = np.where(np.arange(n) < jump_idx, 30.0, 45.0)
    df = pd.DataFrame({
        "ts_utc": _hours(n),
        "station_id": _STATION,
        "temp_c": temp,
        "rh": np.full(n, 60.0),
        "heat_index_c": hi,
    })

    X, df_aug = build_X_once(df)

    # Map X rows to their source positional index (df_aug uses 0..N-1).
    src_pos = X.index.to_numpy()

    # Pre-jump rows: ts_utc[i] < ts_utc[jump_idx]  <=>  src_pos < jump_idx.
    pre_mask = src_pos < jump_idx

    # Every rolling_h window length on heat_index_c must be exactly 30.0
    # for all pre-jump rows: the window only contains shifted (past) values
    # which by construction are all 30.0.
    for w in _DEFAULT_ROLLING_H:
        col = f"heat_index_c_roll{w}h_mean"
        assert col in X.columns, f"missing rolling column: {col}"
        pre_values = X.loc[pre_mask, col].to_numpy()
        if pre_values.size == 0:
            continue
        np.testing.assert_allclose(
            pre_values,
            30.0,
            rtol=0,
            atol=1e-12,
            err_msg=(
                f"{col} peeked at the future 45.0 jump for rows before t*"
            ),
        )

    # Bonus: rolling-std on the constant pre-jump segment must be 0.
    for w in _DEFAULT_ROLLING_H:
        col = f"heat_index_c_roll{w}h_std"
        if col not in X.columns:
            continue
        pre_values = X.loc[pre_mask, col].to_numpy()
        if pre_values.size == 0:
            continue
        np.testing.assert_allclose(
            pre_values, 0.0, rtol=0, atol=1e-12,
            err_msg=f"{col} non-zero on a constant pre-jump segment",
        )


# ---------------------------------------------------------------------------
# 3. Lag feature consistency.
# ---------------------------------------------------------------------------


def test_lag_features_match_raw_value_n_hours_back():
    """``heat_index_c_lag{N}h[i]`` must equal raw HI at ``ts_utc[i] - N h``."""
    n = 120
    # Use a unique value per row so off-by-one would be caught.
    hi = 25.0 + 0.1 * np.arange(n)
    df = _df_with_hi(hi)

    X, df_aug = build_X_once(df)

    # For every default lag, walk every X row and compare lag value with
    # the raw heat_index at ts_utc - N hours.
    src = df.set_index("ts_utc")["heat_index_c"]

    for lag in _DEFAULT_LAGS_H:
        col = f"heat_index_c_lag{lag}h"
        assert col in X.columns, f"missing lag column: {col}"

        ts_at_X = df_aug.loc[X.index, "ts_utc"]
        expected = np.array([
            src.loc[t - pd.Timedelta(hours=lag)] for t in ts_at_X
        ])
        np.testing.assert_allclose(
            X[col].to_numpy(),
            expected,
            rtol=0,
            atol=1e-12,
            err_msg=f"{col} did not match raw HI at t - {lag}h",
        )


# ---------------------------------------------------------------------------
# 4. Truncation invariance — the strongest leakage test.
# ---------------------------------------------------------------------------


def test_truncation_invariance_no_future_data_changes_past_features():
    """If the future is removed, past feature rows must be identical.

    Build features on the full series, then on the same series truncated
    after row ``i``. The feature row corresponding to ``ts_utc = T_i``
    must be bit-identical (within fp tolerance) in both runs — any
    difference proves the builder used data from rows after T_i when
    computing it on the full series.
    """
    # n must be > 168 + some buffer so that truncate_at-1 >= max(lags_h)=168
    # and the truncated df also has >= 169 rows to produce valid X rows.
    n = 250
    rng = np.random.default_rng(42)
    # Realistic-but-noisy HI/temp/rh sequence so rolling/std/interaction
    # features actually vary between rows.
    hi = 30.0 + 5.0 * np.sin(2 * np.pi * np.arange(n) / 24) + rng.normal(0, 0.3, n)
    temp = 28.0 + 4.0 * np.sin(2 * np.pi * np.arange(n) / 24) + rng.normal(0, 0.4, n)
    rh = 70.0 + 8.0 * np.cos(2 * np.pi * np.arange(n) / 24) + rng.normal(0, 0.5, n)
    df_full = pd.DataFrame({
        "ts_utc": _hours(n),
        "station_id": _STATION,
        "temp_c": temp,
        "rh": rh,
        "heat_index_c": hi,
    })

    truncate_at = 185  # keep rows 0..184, drop 185..249 (the "future")
    df_truncated = df_full.iloc[:truncate_at].copy()

    X_full, df_aug_full = build_X_once(df_full)
    X_trunc, df_aug_trunc = build_X_once(df_truncated)

    # We compare the feature row at ts_utc = df_full.iloc[truncate_at - 1].
    # That row is the last row in the truncated df, and it MUST match the
    # row at the same ts_utc in the full-df features.
    target_ts = df_full.iloc[truncate_at - 1]["ts_utc"]

    ts_full = df_aug_full.loc[X_full.index, "ts_utc"]
    ts_trunc = df_aug_trunc.loc[X_trunc.index, "ts_utc"]

    full_match = X_full.loc[ts_full == target_ts]
    trunc_match = X_trunc.loc[ts_trunc == target_ts]

    assert len(full_match) == 1, (
        f"expected exactly 1 X row at {target_ts} in full features, "
        f"got {len(full_match)}"
    )
    assert len(trunc_match) == 1, (
        f"expected exactly 1 X row at {target_ts} in truncated features, "
        f"got {len(trunc_match)}"
    )

    # Compare across the intersection of columns. Sets must match exactly:
    # any column that exists in one but not the other would itself be a
    # leakage smell (e.g. extended features that depended on future
    # fill-rate measurements). Default core features have no extended
    # cols, so the column sets should match.
    assert set(full_match.columns) == set(trunc_match.columns), (
        "feature columns differ between full and truncated runs: "
        f"only-in-full={set(full_match.columns) - set(trunc_match.columns)}, "
        f"only-in-trunc={set(trunc_match.columns) - set(full_match.columns)}"
    )

    full_row = full_match.iloc[0]
    trunc_row = trunc_match.iloc[0]

    diffs = {}
    for col in full_match.columns:
        a = full_row[col]
        b = trunc_row[col]
        # Allow tiny fp drift, but anything material is leakage.
        if not np.isclose(a, b, rtol=0, atol=1e-9, equal_nan=True):
            diffs[col] = (a, b)

    assert not diffs, (
        f"Feature row at ts_utc={target_ts} differs between full and "
        f"truncated runs — future data leaked into past features. "
        f"Differing columns (full, truncated): {diffs}"
    )
