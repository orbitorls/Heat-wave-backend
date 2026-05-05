from __future__ import annotations

import numpy as np
import pandas as pd

from app.ml.forecast.splitting import chronological_split, time_series_cv_splits


def _frame(n: int = 30) -> pd.DataFrame:
    return pd.DataFrame({
        "ts_utc": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
        "station_id": ["A"] * (n // 2) + ["B"] * (n - n // 2),
        "x": np.arange(n),
    })


def test_chronological_split_preserves_order_and_counts():
    df = _frame(100)
    split = chronological_split(df, horizon_h=6)

    assert len(split.train) == 70
    assert len(split.val) == 15
    assert len(split.test) == 15
    assert split.train["ts_utc"].max() <= split.val["ts_utc"].min()
    assert split.val["ts_utc"].max() <= split.test["ts_utc"].min()
    assert split.metadata["horizon_h"] == 6
    assert split.metadata["row_counts"] == {"train": 70, "val": 15, "test": 15}


def test_time_series_cv_uses_horizon_gap():
    X = pd.DataFrame({"x": np.arange(60)})
    splits = list(time_series_cv_splits(X, horizon_h=6, n_splits=3))

    assert splits
    for train_idx, val_idx in splits:
        assert train_idx.max() + 6 < val_idx.min()

