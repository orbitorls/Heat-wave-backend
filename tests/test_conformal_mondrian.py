from __future__ import annotations

import numpy as np

from app.ml.forecast.conformal import MondianCQRCalibrator


def test_mondrian_cqr_supports_danger_tiers_and_roundtrip() -> None:
    n = 120
    rng = np.random.default_rng(7)
    y_true = rng.normal(35.0, 2.0, size=n)
    y_lower = y_true - rng.uniform(1.0, 3.0, size=n)
    y_upper = y_true + rng.uniform(1.0, 3.0, size=n)
    station_ids = np.array(["BKK_01"] * n)
    local_hours = rng.integers(0, 24, size=n)
    danger_tiers = rng.integers(0, 3, size=n)

    cal = MondianCQRCalibrator().fit(
        y_true,
        y_lower,
        y_upper,
        station_ids=station_ids,
        local_hours=local_hours,
        danger_tiers=danger_tiers,
        alpha=0.10,
    )
    lo, hi = cal.adjust(
        y_lower,
        y_upper,
        station_ids=station_ids,
        local_hours=local_hours,
        danger_tiers=danger_tiers,
    )
    assert lo.shape == y_lower.shape
    assert hi.shape == y_upper.shape
    assert np.all(hi >= lo)

    payload = cal.to_dict()
    restored = MondianCQRCalibrator.from_dict(payload)
    lo2, hi2 = restored.adjust(
        y_lower,
        y_upper,
        station_ids=station_ids,
        local_hours=local_hours,
        danger_tiers=danger_tiers,
    )
    np.testing.assert_allclose(lo, lo2)
    np.testing.assert_allclose(hi, hi2)
