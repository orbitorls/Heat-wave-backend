"""Chronological split and train-only imputation helpers for forecast training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


@dataclass(frozen=True)
class ChronologicalSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    metadata: dict


@dataclass(frozen=True)
class XYSplit:
    X_train: pd.DataFrame
    y_train: pd.DataFrame | pd.Series
    X_val: pd.DataFrame
    y_val: pd.DataFrame | pd.Series
    X_test: pd.DataFrame
    y_test: pd.DataFrame | pd.Series
    metadata: dict


@dataclass(frozen=True)
class XYSplit4Way:
    X_train: pd.DataFrame
    y_train: pd.DataFrame | pd.Series
    X_val_es: pd.DataFrame
    y_val_es: pd.DataFrame | pd.Series
    X_val_cal: pd.DataFrame
    y_val_cal: pd.DataFrame | pd.Series
    X_test: pd.DataFrame
    y_test: pd.DataFrame | pd.Series
    metadata: dict


def _sort_for_time(df: pd.DataFrame) -> pd.DataFrame:
    if "ts_utc" in df.columns:
        keys = ["ts_utc"]
        if "station_id" in df.columns:
            keys.append("station_id")
        return df.sort_values(keys).reset_index(drop=True)
    return df.reset_index(drop=True)


def _range(series: pd.Series | None) -> dict | None:
    if series is None or len(series) == 0:
        return None
    return {"start": str(series.iloc[0]), "end": str(series.iloc[-1])}


def chronological_split(
    df: pd.DataFrame,
    *,
    horizon_h: int,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> ChronologicalSplit:
    """Split a frame into chronological train/val/test partitions."""
    ordered = _sort_for_time(df)
    n = len(ordered)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)

    train = ordered.iloc[:train_end].reset_index(drop=True)
    val = ordered.iloc[train_end:val_end].reset_index(drop=True)
    test = ordered.iloc[val_end:].reset_index(drop=True)

    metadata = {
        "horizon_h": horizon_h,
        "row_counts": {"train": len(train), "val": len(val), "test": len(test)},
        "date_ranges": {
            "train": _range(train["ts_utc"]) if "ts_utc" in train else None,
            "val": _range(val["ts_utc"]) if "ts_utc" in val else None,
            "test": _range(test["ts_utc"]) if "ts_utc" in test else None,
        },
        "stations": sorted(ordered["station_id"].dropna().astype(str).unique().tolist())
        if "station_id" in ordered
        else [],
    }
    return ChronologicalSplit(train=train, val=val, test=test, metadata=metadata)


def split_xy(
    X: pd.DataFrame,
    y: pd.DataFrame | pd.Series,
    *,
    horizon_h: int,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> XYSplit:
    """Split aligned X/y, using X attrs ts_utc/station_id for metadata if present."""
    meta = pd.DataFrame({"_pos": np.arange(len(X))})
    if "ts_utc" in X.attrs:
        meta["ts_utc"] = pd.Series(X.attrs["ts_utc"]).reset_index(drop=True)
    if "station_id" in X.attrs:
        meta["station_id"] = pd.Series(X.attrs["station_id"]).reset_index(drop=True)

    ordered_meta = _sort_for_time(meta)
    order = ordered_meta["_pos"].to_numpy()
    X_ordered = X.iloc[order].reset_index(drop=True)
    y_ordered = y.iloc[order].reset_index(drop=True)

    split = chronological_split(ordered_meta.drop(columns=["_pos"]), horizon_h=horizon_h,
                                train_frac=train_frac, val_frac=val_frac)
    n_train = split.metadata["row_counts"]["train"]
    n_val = split.metadata["row_counts"]["val"]
    val_end = n_train + n_val

    def _clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.reset_index(drop=True)
        out.attrs.clear()
        return out

    return XYSplit(
        X_train=_clean_frame(X_ordered.iloc[:n_train]),
        y_train=y_ordered.iloc[:n_train].reset_index(drop=True),
        X_val=_clean_frame(X_ordered.iloc[n_train:val_end]),
        y_val=y_ordered.iloc[n_train:val_end].reset_index(drop=True),
        X_test=_clean_frame(X_ordered.iloc[val_end:]),
        y_test=y_ordered.iloc[val_end:].reset_index(drop=True),
        metadata=split.metadata,
    )


def split_xy_4way(
    X: pd.DataFrame,
    y: pd.DataFrame | pd.Series,
    *,
    horizon_h: int,
    train_frac: float = 0.70,
    val_es_frac: float = 0.075,
    val_cal_frac: float = 0.075,
) -> XYSplit4Way:
    """Chronological 4-way split: train/val_es/val_cal/test (70/7.5/7.5/15).

    val_es  — early-stopping only (booster never calibrated on this)
    val_cal — MondrianCQR residuals + DangerGate threshold sweep
    test    — held-out metrics reported in bundle.json
    """
    meta = pd.DataFrame({"_pos": np.arange(len(X))})
    if "ts_utc" in X.attrs:
        meta["ts_utc"] = pd.Series(X.attrs["ts_utc"]).reset_index(drop=True)
    if "station_id" in X.attrs:
        meta["station_id"] = pd.Series(X.attrs["station_id"]).reset_index(drop=True)

    ordered_meta = _sort_for_time(meta)
    order = ordered_meta["_pos"].to_numpy()
    X_ordered = X.iloc[order].reset_index(drop=True)
    y_ordered = y.iloc[order].reset_index(drop=True)

    n = len(X_ordered)
    n_train = int(n * train_frac)
    n_val_es = int(n * val_es_frac)
    n_val_cal = int(n * val_cal_frac)
    es_end = n_train + n_val_es
    cal_end = es_end + n_val_cal

    def _cf(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.reset_index(drop=True)
        out.attrs.clear()
        return out

    def _cy(series: pd.Series) -> pd.Series:
        return series.reset_index(drop=True)

    ts_col = ordered_meta.get("ts_utc") if "ts_utc" in ordered_meta.columns else None

    def _rng(sl: slice) -> dict | None:
        if ts_col is None:
            return None
        chunk = ts_col.iloc[sl]
        return {"start": str(chunk.iloc[0]), "end": str(chunk.iloc[-1])} if len(chunk) else None

    metadata = {
        "horizon_h": horizon_h,
        "row_counts": {
            "train": n_train,
            "val_es": n_val_es,
            "val_cal": n_val_cal,
            "test": n - cal_end,
        },
        "date_ranges": {
            "train": _rng(slice(None, n_train)),
            "val_es": _rng(slice(n_train, es_end)),
            "val_cal": _rng(slice(es_end, cal_end)),
            "test": _rng(slice(cal_end, None)),
        },
        "stations": sorted(ordered_meta["station_id"].dropna().astype(str).unique().tolist())
        if "station_id" in ordered_meta.columns
        else [],
    }

    return XYSplit4Way(
        X_train=_cf(X_ordered.iloc[:n_train]),
        y_train=_cy(y_ordered.iloc[:n_train]),
        X_val_es=_cf(X_ordered.iloc[n_train:es_end]),
        y_val_es=_cy(y_ordered.iloc[n_train:es_end]),
        X_val_cal=_cf(X_ordered.iloc[es_end:cal_end]),
        y_val_cal=_cy(y_ordered.iloc[es_end:cal_end]),
        X_test=_cf(X_ordered.iloc[cal_end:]),
        y_test=_cy(y_ordered.iloc[cal_end:]),
        metadata=metadata,
    )


def time_series_cv_splits(
    X: pd.DataFrame,
    *,
    horizon_h: int,
    n_splits: int = 5,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield TimeSeriesSplit folds with a gap equal to the forecast horizon."""
    max_splits = max(2, min(n_splits, len(X) // max(horizon_h + 2, 2)))
    splitter = TimeSeriesSplit(n_splits=max_splits, gap=horizon_h)
    yield from splitter.split(X)


def fit_feature_medians(X_train: pd.DataFrame) -> dict[str, float]:
    """Fit numeric medians from the training split only."""
    medians: dict[str, float] = {}
    for col in X_train.columns:
        if pd.api.types.is_numeric_dtype(X_train[col]):
            value = X_train[col].median()
            medians[col] = 0.0 if pd.isna(value) else float(value)
    return medians


def apply_feature_medians(X: pd.DataFrame, medians: dict[str, float]) -> pd.DataFrame:
    """Apply train-fitted medians and replace any remaining numeric NaN with zero."""
    out = X.copy()
    out.attrs.clear()
    for col, value in medians.items():
        if col in out.columns:
            out[col] = out[col].fillna(value)
    numeric_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    out[numeric_cols] = out[numeric_cols].fillna(0.0)
    return out
