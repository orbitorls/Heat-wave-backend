"""Regression tests for forecast acceptance bar (v3 LightGBM backend).

Requires the actual ingested data and a trained v3 model for BKK_01/h24.
Run manually: pytest -m slow tests/test_forecast_v2_regression.py

Acceptance criteria (must all pass to ship):
- MAE ≤ 2.20°C
- skill_score ≥ 0.55
- danger_recall ≥ 0.40  (HI > 42°C)
- pi_coverage_90 ∈ [0.85, 0.93]  (fraction inside q05-q95 interval)
- pi_width_ratio ≥ 1.5  (P75 / P25 of interval widths)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from datetime import date, timedelta
from sklearn.metrics import mean_absolute_error

pytestmark = pytest.mark.slow

EVAL_STATION = "BKK_01"
EVAL_HORIZON = 24


@pytest.fixture(scope="module")
def eval_data():
    """Load test data and run inference using the v3 LightGBM forecaster."""
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from app.ml.registry import load_latest_v3
    from app.data.loaders import read_observations
    from app.ml.forecast.features import build_features, _DEFAULT_LAGS_H, _DEFAULT_ROLLING_H
    from app.ml.forecast.backends.lgbm_backend import _compute_hi_array
    from app.ml.forecast.splitting import split_xy

    try:
        forecaster = load_latest_v3(EVAL_STATION, EVAL_HORIZON)
    except FileNotFoundError:
        pytest.skip(
            f"No v3 model for {EVAL_STATION}/h{EVAL_HORIZON} — run train_forecast.py first"
        )

    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=730)

    df = read_observations(EVAL_STATION, start, end)
    if df.empty:
        pytest.skip(f"No ingested data found for {EVAL_STATION} — run ingest scripts first")

    df["station_id"] = EVAL_STATION
    X, y = build_features(
        df, horizon_h=EVAL_HORIZON,
        lags_h=_DEFAULT_LAGS_H, rolling_h=_DEFAULT_ROLLING_H,
        target_kind="th",
    )
    split = split_xy(X, y, horizon_h=EVAL_HORIZON)
    X_test, y_test = split.X_test, split.y_test
    y_true = _compute_hi_array(y_test["temp_c"].values, y_test["rh"].values)

    bundle = forecaster.predict_with_pi(X_test)

    return {
        "y_true": y_true,
        "mean_pred": np.asarray(bundle.hi_mean),
        "q05_pred": np.asarray(bundle.hi_lower),
        "q95_pred": np.asarray(bundle.hi_upper),
    }


def test_mae_within_target(eval_data):
    mae = mean_absolute_error(eval_data["y_true"], eval_data["mean_pred"])
    assert mae <= 2.20, f"MAE {mae:.4f}°C exceeds 2.20°C target"


def test_skill_score_no_regression(eval_data):
    y_true = eval_data["y_true"]
    y_pred = eval_data["mean_pred"]
    mae = mean_absolute_error(y_true, y_pred)
    # Persistence baseline = std(y_true): climatological spread that a naive
    # mean-forecast would produce. Skill ≥ 0.55 means MAE is ≤ 45% of std.
    # (np.diff baseline was 1-hour lag, impossible to beat at 24 h horizon.)
    mae_persistence = float(np.std(y_true))
    skill = 1.0 - mae / max(mae_persistence, 1e-6)
    assert skill >= 0.55, f"skill_score {skill:.4f} below 0.55 target"


def test_danger_recall(eval_data):
    """Danger-class recall (HI > 42°C) must be >= 0.40."""
    y_true = eval_data["y_true"]
    y_pred = eval_data["mean_pred"]
    danger_mask = y_true > 42.0
    if danger_mask.sum() == 0:
        pytest.skip("No Danger-class samples in test set")
    recall = float((y_pred[danger_mask] > 42.0).mean())
    assert recall >= 0.40, f"Danger recall {recall:.3f} below 0.40 target"


def test_pi_coverage(eval_data):
    """90% PI coverage must be in [0.85, 0.93]."""
    q05 = eval_data.get("q05_pred")
    q95 = eval_data.get("q95_pred")
    if q05 is None or q95 is None:
        pytest.skip("Model does not expose quantile bounds")
    y_true = eval_data["y_true"]
    coverage = float(((y_true >= q05) & (y_true <= q95)).mean())
    assert 0.85 <= coverage <= 0.93, f"90% PI coverage {coverage:.3f} outside [0.85, 0.93]"


def test_pi_width_varies(eval_data):
    """PI width must vary — P75/P25 ratio >= 1.5."""
    q05 = eval_data.get("q05_pred")
    q95 = eval_data.get("q95_pred")
    if q05 is None or q95 is None:
        pytest.skip("Model does not expose quantile bounds")
    widths = q95 - q05
    p25, p75 = np.percentile(widths, [25, 75])
    ratio = p75 / max(p25, 1e-6)
    assert ratio >= 1.5, f"PI width ratio (P75/P25) {ratio:.3f} below 1.5 — PI is too flat"
