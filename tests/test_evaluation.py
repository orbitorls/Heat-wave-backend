from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from app.ml.forecast.evaluation import evaluate_predictions


def _out_dir() -> Path:
    path = Path("logs") / "test_eval" / uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_evaluator_writes_metrics_and_summary():
    out_dir = _out_dir()
    y_true = np.array([31.0, 35.0, 41.0, 43.0, 45.0, 39.0])
    y_pred = np.array([30.5, 34.0, 40.0, 42.5, 41.0, 38.0])
    X = pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * np.arange(len(y_true)) / 24),
        "hour_cos": np.cos(2 * np.pi * np.arange(len(y_true)) / 24),
        "station_enc": [0, 0, 1, 1, 1, 0],
        "heat_index_c_lag1h": y_true - 0.5,
        "month": [1] * len(y_true),
    })

    metrics = evaluate_predictions(
        y_true,
        y_pred,
        X,
        out_dir=out_dir,
        horizon_h=6,
        station_labels={0: "A", 1: "B"},
        q05=y_pred - 1.0,
        q95=y_pred + 1.0,
        runtime={"train_seconds": 1.2, "eval_seconds": 0.3},
        split_metadata={"row_counts": {"train": 10, "val": 3, "test": 6}},
    )

    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "summary.md").exists()
    loaded = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert loaded["regression"]["mae"] == metrics["regression"]["mae"]
    assert "baselines" in loaded
    assert "risk" in loaded
    assert "safety" in loaded
    assert "prediction_interval" in loaded
    assert loaded["runtime"]["train_seconds"] == 1.2


def test_evaluator_skips_pi_metrics_when_quantiles_missing():
    out_dir = _out_dir()
    y_true = np.array([31.0, 35.0, 41.0])
    y_pred = np.array([31.5, 34.5, 39.0])
    X = pd.DataFrame({
        "hour_sin": [0.0, 0.0, 0.0],
        "hour_cos": [1.0, 1.0, 1.0],
        "station_enc": [0, 0, 0],
        "heat_index_c_lag1h": y_true,
    })

    metrics = evaluate_predictions(y_true, y_pred, X, out_dir=out_dir, horizon_h=6)

    assert metrics["prediction_interval"]["available"] is False
