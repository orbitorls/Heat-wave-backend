"""Tests for Risk Scoring engine"""
import pytest
from app.core.heat_index import compute as compute_hi
from app.core.risk_scoring import (
    ActivityContext,
    ActivityIntensity,
    RiskClass,
    score,
)
from app.core.vulnerability import get_profile, ProfileID


def _ctx(
    intensity=ActivityIntensity.MODERATE,
    duration=60,
    shade=False,
    water=True,
    hour=13,
):
    return ActivityContext(
        intensity=intensity,
        duration_minutes=duration,
        shade_available=shade,
        water_access=water,
        time_of_day_hour=hour,
    )


def test_critical_for_child_in_heat():
    hi = compute_hi(40, 85)
    profile = get_profile(ProfileID.STUDENT_PRIMARY)
    activity = _ctx(intensity=ActivityIntensity.HIGH, duration=90, shade=False)
    result = score(hi, profile, activity)
    assert result.risk_class in (RiskClass.HIGH, RiskClass.CRITICAL)


def test_low_risk_for_adult_rest_cool():
    hi = compute_hi(28, 50)
    profile = get_profile(ProfileID.GENERAL_ADULT)
    activity = _ctx(intensity=ActivityIntensity.REST, duration=20, shade=True, hour=9)
    result = score(hi, profile, activity)
    assert result.risk_class == RiskClass.LOW


def test_shade_reduces_score():
    hi = compute_hi(38, 70)
    profile = get_profile(ProfileID.OUTDOOR_WORKER)
    no_shade = _ctx(shade=False)
    with_shade = _ctx(shade=True)
    r_no = score(hi, profile, no_shade)
    r_yes = score(hi, profile, with_shade)
    assert r_yes.score < r_no.score


def test_longer_duration_increases_score():
    hi = compute_hi(36, 65)
    profile = get_profile(ProfileID.OUTDOOR_WORKER)
    short = _ctx(duration=20)
    long_ = _ctx(duration=120)
    assert score(hi, profile, long_).score > score(hi, profile, short).score


def test_conservative_bump():
    hi = compute_hi(36, 65)
    profile = get_profile(ProfileID.GENERAL_ADULT)
    activity = _ctx()
    _order = [RiskClass.LOW, RiskClass.MODERATE, RiskClass.HIGH, RiskClass.CRITICAL]
    normal = score(hi, profile, activity, low_confidence=False)
    conservative = score(hi, profile, activity, low_confidence=True)
    # conservative risk class ต้องไม่ต่ำกว่าปกติ (เปรียบตาม order)
    assert _order.index(conservative.risk_class) >= _order.index(normal.risk_class)
    if normal.risk_class != RiskClass.CRITICAL:
        assert conservative.conservative_applied is True


def test_dominant_factors_not_empty():
    hi = compute_hi(39, 80)
    profile = get_profile(ProfileID.STUDENT_PRIMARY)
    result = score(hi, profile, _ctx(hour=13, duration=90))
    assert len(result.dominant_factors) > 0


def test_score_capped_at_100():
    hi = compute_hi(45, 95)
    profile = get_profile(ProfileID.ELDERLY)
    activity = _ctx(
        intensity=ActivityIntensity.VERY_HIGH,
        duration=180,
        shade=False,
        water=False,
    )
    result = score(hi, profile, activity)
    assert result.score <= 100
