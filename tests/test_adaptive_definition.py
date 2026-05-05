"""Tests for AdaptiveDefinitionEngine"""
import pytest
from datetime import date
from app.core.adaptive_definition import (
    AdaptiveDefinitionEngine,
    DailyObs,
)


def _make_historical(n=365, base=35.0, std=3.0):
    import random
    random.seed(42)
    return [base + random.gauss(0, std) for _ in range(n)]


def _make_obs(max_vals, min_vals=None):
    if min_vals is None:
        min_vals = [v - 8 for v in max_vals]
    return [
        DailyObs(date=date(2024, 5, i + 1), max_value=m, min_value=n)
        for i, (m, n) in enumerate(zip(max_vals, min_vals))
    ]


def test_heatwave_detected_when_consecutive_days_met():
    engine = AdaptiveDefinitionEngine(percentile=90, min_consecutive_days=2)
    historical = _make_historical()
    baseline = engine.build_baseline("BKK", historical)

    # ค่าสูงกว่า p90 มาก → ต้องเป็น heatwave
    extreme = [baseline.p90 + 5] * 3
    obs = _make_obs(extreme)
    event = engine.detect(obs, baseline)
    assert event.is_heatwave is True
    assert event.consecutive_days >= 2


def test_not_heatwave_when_insufficient_days():
    engine = AdaptiveDefinitionEngine(percentile=90, min_consecutive_days=3)
    historical = _make_historical()
    baseline = engine.build_baseline("BKK", historical)

    # เกินเพียง 2 วัน แต่ engine ต้องการ 3
    extreme = [baseline.p90 - 1, baseline.p90 + 5, baseline.p90 + 5]
    obs = _make_obs(extreme)
    event = engine.detect(obs, baseline)
    assert event.is_heatwave is False


def test_baseline_requires_30_obs():
    engine = AdaptiveDefinitionEngine()
    with pytest.raises(ValueError, match="30"):
        engine.build_baseline("X", [35.0] * 10)


def test_baseline_percentiles_ordered():
    engine = AdaptiveDefinitionEngine()
    historical = _make_historical(365)
    baseline = engine.build_baseline("BKK", historical)
    assert baseline.p90 <= baseline.p95
    assert baseline.mean < baseline.p90


def test_warm_nights_required():
    engine = AdaptiveDefinitionEngine(
        percentile=90,
        min_consecutive_days=2,
        require_warm_nights=True,
    )
    historical = _make_historical()
    baseline = engine.build_baseline("BKK", historical)

    # Days เกิน p90 แต่ nights ไม่เกิน → ไม่เป็น heatwave
    max_vals = [baseline.p90 + 5] * 3
    min_vals = [baseline.p90 - 10] * 3  # nights เย็น
    obs = _make_obs(max_vals, min_vals)
    event = engine.detect(obs, baseline)
    assert event.is_heatwave is False
