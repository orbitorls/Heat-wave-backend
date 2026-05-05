"""Feature engineering for heat-index forecast model.

CRITICAL: All lag and rolling features must only use data available at
prediction time (t=0). The target y = heat_index_c at t+horizon_h.
No future values are ever used in X.
"""
from __future__ import annotations
from typing import Literal, Sequence
import numpy as np
import pandas as pd

from app.data.stations import STATIONS


_DEFAULT_LAGS_H = [1, 3, 6, 12, 24]
_DEFAULT_ROLLING_H = [3, 6, 24]


def add_heat_index_col(df: pd.DataFrame) -> pd.DataFrame:
    """Compute heat_index_c from temp_c + rh using the Rothfusz formula
    if the column is missing. Avoids importing the FastAPI app layer inside training."""
    if "heat_index_c" in df.columns:
        return df
    # Simplified Steadman for feature engineering (close enough for training features)
    T = df["temp_c"]
    R = df["rh"]
    hi = (-8.78469475556 +
           1.61139411 * T +
           2.33854883889 * R +
           -0.14611605 * T * R +
           -0.012308094 * T**2 +
           -0.0164248277778 * R**2 +
           0.002211732 * T**2 * R +
           0.00072546 * T * R**2 +
           -0.000003582 * T**2 * R**2)
    df = df.copy()
    df["heat_index_c"] = hi
    return df


def build_X_once(
    df: pd.DataFrame,
    lags_h: list[int] = _DEFAULT_LAGS_H,
    rolling_h: list[int] = _DEFAULT_ROLLING_H,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build feature matrix X (no target) and return (X, df_augmented).

    df_augmented retains all engineered columns plus ts_utc/station_id so
    build_y_for_horizon can compute the target without re-reading parquet.
    X drops rows where any core (non-extended) feature is NaN.

    Args:
        df: DataFrame with columns ts_utc (datetime, UTC), station_id,
            temp_c, rh, heat_index_c (optional — computed if missing).
            Must be sorted by ts_utc, 1-hour frequency.
        lags_h: Lag steps in hours to include as features.
        rolling_h: Rolling window sizes in hours for mean and std features.

    Returns:
        (X, df_augmented) where X is the feature matrix with valid rows only
        and df_augmented retains all rows with all engineered columns.
    """
    df = df.copy()
    df = add_heat_index_col(df)

    # Temporal features (derived from observation timestamp only)
    if not pd.api.types.is_datetime64_any_dtype(df["ts_utc"]):
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)

    if "station_id" not in df.columns:
        df["station_id"] = "default"

    df = df.sort_values(["station_id", "ts_utc"]).reset_index(drop=True)
    station_group = df.groupby("station_id", sort=False)
    df["hour"] = df["ts_utc"].dt.hour
    df["day_of_year"] = df["ts_utc"].dt.day_of_year
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    local_hour = (df["hour"] + 7) % 24
    df["local_hour_sin"] = np.sin(2 * np.pi * local_hour / 24)
    df["local_hour_cos"] = np.cos(2 * np.pi * local_hour / 24)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
    df["month"] = df["ts_utc"].dt.month

    # Station encoding (label encode)
    if "station_id" in df.columns:
        df["station_enc"] = df["station_id"].astype("category").cat.codes

    # Station geometry features — constant per-station lat/lon/elevation
    if "station_id" in df.columns:
        df["lat"] = df["station_id"].map(
            lambda sid: STATIONS[sid].lat if sid in STATIONS else 0.0
        )
        df["lon"] = df["station_id"].map(
            lambda sid: STATIONS[sid].lon if sid in STATIONS else 0.0
        )
        df["elevation_m"] = df["station_id"].map(
            lambda sid: STATIONS[sid].elevation_m if sid in STATIONS else 0.0
        )

    # Lag features — shift FORWARD in the index (shift(n) uses past values)
    for col in ["heat_index_c", "temp_c", "rh"]:
        for lag in lags_h:
            df[f"{col}_lag{lag}h"] = station_group[col].transform(lambda s, lag=lag: s.shift(lag))

    # Cooling-rate features — difference over a 3-hour window using past values only
    df["hi_change_3h"] = station_group["heat_index_c"].transform(lambda s: s.shift(1) - s.shift(4))
    df["temp_change_3h"] = station_group["temp_c"].transform(lambda s: s.shift(1) - s.shift(4))

    # Rolling features — on the already-shifted columns to avoid leakage
    # Rolling window of size w at position i uses rows [i-w+1 .. i] — past only
    for col in ["heat_index_c", "temp_c"]:
        for w in rolling_h:
            df[f"{col}_roll{w}h_mean"] = station_group[col].transform(
                lambda s, w=w: s.shift(1).rolling(w, min_periods=1).mean()
            )
            df[f"{col}_roll{w}h_std"] = station_group[col].transform(
                lambda s, w=w: s.shift(1).rolling(w, min_periods=1).std().fillna(0)
            )

    # Core interaction feature — uses shift(1) to avoid leakage
    df["temp_x_rh"] = (
        station_group["temp_c"].transform(lambda s: s.shift(1))
        * station_group["rh"].transform(lambda s: s.shift(1))
    )

    # Evening-peak interaction — captures high-temp risk during 16:00–20:00 local time
    local_hour_num = (df["ts_utc"].dt.hour + 7) % 24
    evening_mask = ((local_hour_num >= 16) & (local_hour_num <= 20)).astype(float)
    df["temp_x_evening"] = (
        station_group["temp_c"].transform(lambda s: s.shift(1))
        * evening_mask
    )

    # Extended atmospheric features — included when the column is present AND
    # at least 30% non-null. Missing values are forward-filled then backfilled
    # so lag creation doesn't explode with NaN.
    _EXTENDED_COLS = ["solar_wm2", "cloud_cover", "blh_m", "pressure_hpa", "lst_c"]
    _EXTENDED_LAGS = [1, 3, 6]  # shorter lag set for sparser extended variables
    for col in _EXTENDED_COLS:
        if col not in df.columns:
            continue
        fill_rate = df[col].notna().mean()
        if fill_rate < 0.30:
            continue  # not enough data to be useful
        # Forward-fill gaps within station only. Do not backfill: that would use
        # future observations to fill earlier feature rows.
        df[col] = station_group[col].transform(lambda s: s.ffill())
        for lag in _EXTENDED_LAGS:
            df[f"{col}_lag{lag}h"] = station_group[col].transform(lambda s, lag=lag: s.shift(lag))
        # Rolling mean for solar (captures day trend) + solar-temp interaction
        if col == "solar_wm2":
            df["solar_wm2_roll6h_mean"] = station_group[col].transform(
                lambda s: s.shift(1).rolling(6, min_periods=1).mean()
            )
            df["temp_x_solar"] = (
                station_group["temp_c"].transform(lambda s: s.shift(1))
                * station_group[col].transform(lambda s: s.shift(1))
            )

    # Wind and precipitation features — included when present AND ≥20% non-null.
    # Shorter lag set since these are typically sparser than core met variables.
    _WIND_PRECIP_COLS = ["wind_ms", "precip_mm"]
    _WIND_PRECIP_LAGS = [1, 3]
    for col in _WIND_PRECIP_COLS:
        if col not in df.columns:
            continue
        fill_rate = df[col].notna().mean()
        if fill_rate < 0.20:
            continue  # too sparse
        df[col] = station_group[col].transform(lambda s: s.ffill())
        for lag in _WIND_PRECIP_LAGS:
            df[f"{col}_lag{lag}h"] = station_group[col].transform(lambda s, lag=lag: s.shift(lag))

    # Always ensure lag columns exist for consistency with get_feature_names().
    # Filled with 0 when the source column was absent or too sparse.
    for col in _WIND_PRECIP_COLS:
        for lag in _WIND_PRECIP_LAGS:
            lag_col = f"{col}_lag{lag}h"
            if lag_col not in df.columns:
                df[lag_col] = 0.0

    # Climatology residual — causal expanding mean per (station, month, hour)
    # bucket. For row at time t with bucket key K, _hi_clim[t] is the mean of
    # heat_index_c at all rows <= t that share K. Using transform("mean") here
    # would leak future bucket members into past rows (caught by
    # test_truncation_invariance_no_future_data_changes_past_features).
    # Rows are already sorted by (station_id, ts_utc) above, so a cumulative
    # mean within each bucket is causal. For the first occurrence of a bucket
    # the expanding mean equals heat_index_c itself; that is still leak-free.
    bucket = df.groupby(["station_id", "month", "hour"], sort=False)["heat_index_c"]
    df["_hi_clim"] = bucket.transform(lambda s: s.expanding(min_periods=1).mean())
    df["hi_residual_lag1h"] = (
        station_group["heat_index_c"].transform(lambda s: s.shift(1))
        - station_group["_hi_clim"].transform(lambda s: s.shift(1))
    )

    # Columns excluded from X (identifiers, raw targets, or now-promoted wind/precip
    # that appear only through their lag derivatives)
    _NON_FEATURE = {"ts_utc", "station_id", "heat_index_c", "temp_c", "rh",
                    "hour", "day_of_year", "local_hour", "source"}
    feature_cols = [
        c for c in df.columns
        if c not in _NON_FEATURE and pd.api.types.is_numeric_dtype(df[c])
    ]
    X_all = df[feature_cols].copy()

    # Extended + wind/precip columns can still have NaN after causal forward-fill.
    # Training code imputes these after splitting using train-only medians.
    _ALL_EXTENDED = list(_EXTENDED_COLS) + _WIND_PRECIP_COLS
    ext_feat_cols = [c for c in X_all.columns if any(c.startswith(e) for e in _ALL_EXTENDED)]

    core_valid = X_all[
        [c for c in X_all.columns if c not in ext_feat_cols]
    ].notna().all(axis=1)

    output_meta = df.loc[core_valid, ["ts_utc", "station_id"]].copy()
    X = X_all[core_valid].copy()

    order = output_meta.sort_values(["ts_utc", "station_id"]).index
    # Keep original positional index (subset of 0..N-1) so build_y_for_horizon
    # can use .loc[X.index] to select the exact same rows from df_augmented.
    X = X.loc[order]
    output_meta = output_meta.loc[order]
    X.attrs["ts_utc"] = output_meta["ts_utc"]
    X.attrs["station_id"] = output_meta["station_id"]

    # df_augmented: all rows with all engineered columns. build_y_for_horizon
    # uses X.index (= order) to select the matching target rows via .loc.
    df_augmented = df.copy()

    return X, df_augmented


def build_y_for_horizon(
    df_augmented: pd.DataFrame,
    X_valid_index: pd.Index,
    horizon_h: int,
    target_kind: Literal["hi", "th"] = "hi",
) -> pd.Series | pd.DataFrame:
    """Compute target y for a specific horizon from the cached augmented df.

    Returns y aligned to X_valid_index rows, with additional NaN rows excluded.
    The caller is responsible for intersecting X and y on the final valid mask:
        valid = y.notna().all(axis=1) if isinstance(y, pd.DataFrame) else y.notna()
        X_final = X.loc[valid]
        y_final = y.loc[valid]

    Args:
        df_augmented: The df_augmented returned by build_X_once (all rows, all cols).
        X_valid_index: The index of X returned by build_X_once (core-valid rows).
        horizon_h: How many hours ahead to predict.
        target_kind: "hi" for heat_index_c, "th" for (temp_c, rh) pair.

    Returns:
        y aligned to X_valid_index.
    """
    station_group = df_augmented.groupby("station_id", sort=False)

    if target_kind == "th":
        y_full = pd.DataFrame({
            "temp_c": station_group["temp_c"].transform(lambda s: s.shift(-horizon_h)),
            "rh": station_group["rh"].transform(lambda s: s.shift(-horizon_h)),
        })
    else:
        y_full = station_group["heat_index_c"].transform(lambda s: s.shift(-horizon_h))

    # df_augmented has a 0..N-1 RangeIndex (from build_X_once's reset_index).
    # X_valid_index contains label values from that same range, so .loc selects
    # the matching rows and preserves those labels as the output index.
    # This keeps y's index aligned with X's index for downstream boolean filtering.
    y = y_full.loc[X_valid_index] if len(X_valid_index) > 0 else y_full.iloc[[]]
    return y


def build_features(
    df: pd.DataFrame,
    horizon_h: int,
    lags_h: list[int] = _DEFAULT_LAGS_H,
    rolling_h: list[int] = _DEFAULT_ROLLING_H,
    target_kind: Literal["hi", "th"] = "hi",
) -> tuple[pd.DataFrame, pd.DataFrame | pd.Series]:
    """Build lag + rolling features for XGBoost heat-index forecast.

    Backward-compatible wrapper around build_X_once + build_y_for_horizon.

    Args:
        df: DataFrame with columns ts_utc (datetime, UTC), station_id,
            temp_c, rh, heat_index_c (optional — computed if missing).
            Must be sorted by ts_utc, 1-hour frequency.
        horizon_h: How many hours ahead to predict.
        lags_h: Lag steps in hours to include as features.
        rolling_h: Rolling window sizes in hours for mean and std features.
        target_kind: "hi" for heat_index_c scalar target, "th" for (temp_c, rh).

    Returns:
        (X, y) where y = heat_index_c shifted by -horizon_h.

    CRITICAL invariant: for every row i in X, the feature values only use
    data from df rows with ts_utc <= df.ts_utc[i]. The target y[i] uses
    df.ts_utc[i + horizon_h]. No leakage.
    """
    X, df_aug = build_X_once(df, lags_h, rolling_h)
    y = build_y_for_horizon(df_aug, X.index, horizon_h, target_kind)
    valid = y.notna().all(axis=1) if isinstance(y, pd.DataFrame) else y.notna()
    return X[valid].reset_index(drop=True), y[valid].reset_index(drop=True)


def get_feature_names(
    lags_h: list[int] = _DEFAULT_LAGS_H,
    rolling_h: list[int] = _DEFAULT_ROLLING_H,
) -> list[str]:
    """Return the expected feature column names (for validation at predict time)."""
    names = [
        "hour_sin", "hour_cos", "doy_sin", "doy_cos", "month", "station_enc",
        # Station geometry — always present (filled with 0.0 for unknown stations)
        "lat", "lon", "elevation_m",
    ]
    for col in ["heat_index_c", "temp_c", "rh"]:
        for lag in lags_h:
            names.append(f"{col}_lag{lag}h")
    for col in ["heat_index_c", "temp_c"]:
        for w in rolling_h:
            names.append(f"{col}_roll{w}h_mean")
            names.append(f"{col}_roll{w}h_std")
    # v2 features — always present regardless of extended column availability
    names += [
        "local_hour_sin", "local_hour_cos",
        "hi_change_3h", "temp_change_3h",
        "temp_x_rh",
        "temp_x_evening",
    ]
    # Optional/extended features — present only when the source column has ≥20–30% fill
    # These are conditioned on column presence and fill rate at training time.
    names += [
        "wind_ms_lag1h", "wind_ms_lag3h",
        "precip_mm_lag1h", "precip_mm_lag3h",
    ]
    return names
