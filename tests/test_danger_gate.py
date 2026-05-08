from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.ml.forecast.danger_gate import DangerGate


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_danger_gate_brf_multiclass_predicts_three_tiers() -> None:
    pytest.importorskip("imblearn")
    rng = np.random.default_rng(42)
    n = 240
    X = pd.DataFrame({
        "f1": rng.normal(0.0, 1.0, size=n),
        "f2": rng.normal(0.0, 1.0, size=n),
        "f3": rng.normal(0.0, 1.0, size=n),
    })
    # Build synthetic HI with clear tier separation.
    y_hi = pd.Series(
        np.concatenate([
            rng.normal(35.0, 0.5, size=n // 3),
            rng.normal(39.5, 0.6, size=n // 3),
            rng.normal(43.5, 0.6, size=n - 2 * (n // 3)),
        ])
    )
    X = X.iloc[: len(y_hi)].reset_index(drop=True)

    gate = DangerGate(gate_backend="brf").fit(
        X,
        y_hi,
        val_X=X,
        val_y_hi=y_hi,
        n_estimators=80,
        random_state=42,
    )
    proba = gate.predict_tier_proba(X)
    tiers = gate.predict_tier(X)

    assert proba.shape == (len(X), 3)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)), atol=1e-6)
    assert set(np.unique(tiers)).issubset({0, 1, 2})
    assert 0.0 <= gate.warning_threshold <= 1.0
    assert 0.0 <= gate.danger_threshold <= 1.0
