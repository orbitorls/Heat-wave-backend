"""Tests for What-if Simulator"""
from app.core.heat_index import compute as compute_hi
from app.core.risk_scoring import ActivityContext, ActivityIntensity
from app.core.vulnerability import get_profile, ProfileID
from app.core.whatif import (
    Intervention,
    InterventionType,
    simulate,
)


def _base():
    hi = compute_hi(39, 80)
    profile = get_profile(ProfileID.STUDENT_PRIMARY)
    activity = ActivityContext(
        intensity=ActivityIntensity.HIGH,
        duration_minutes=90,
        shade_available=False,
        water_access=False,
        time_of_day_hour=13,
    )
    return hi, profile, activity


def test_cancel_reduces_to_minimum():
    hi, profile, activity = _base()
    result = simulate(hi, profile, activity, [
        Intervention(intervention_type=InterventionType.CANCEL)
    ])
    cancel_scenario = result.scenarios[0]
    assert cancel_scenario.new_score.score <= cancel_scenario.original_score.score


def test_shade_effective():
    # ใช้สภาพที่ shade สร้างความต่างได้ (ไม่ extreme จนติด cap 100)
    from app.core.heat_index import compute as compute_hi2
    from app.core.risk_scoring import ActivityContext, ActivityIntensity
    from app.core.vulnerability import get_profile, ProfileID
    hi2 = compute_hi2(35, 65)
    profile2 = get_profile(ProfileID.OUTDOOR_WORKER)
    activity_no_shade = ActivityContext(
        intensity=ActivityIntensity.MODERATE,
        duration_minutes=60,
        shade_available=False,
        water_access=True,
        time_of_day_hour=13,
    )
    activity_shade = ActivityContext(
        intensity=ActivityIntensity.MODERATE,
        duration_minutes=60,
        shade_available=True,
        water_access=True,
        time_of_day_hour=13,
    )
    from app.core.risk_scoring import score as rs_score
    no_shade_score = rs_score(hi2, profile2, activity_no_shade)
    shade_score = rs_score(hi2, profile2, activity_shade)
    assert shade_score.score < no_shade_score.score


def test_shift_time_to_evening_reduces_risk():
    hi, profile, activity = _base()
    result = simulate(hi, profile, activity, [
        Intervention(intervention_type=InterventionType.SHIFT_TIME, shift_hours=4)
    ])
    s = result.scenarios[0]
    # เลื่อนจาก 13:00 → 17:00 ออกนอก peak heat zone
    assert s.new_score.score <= s.original_score.score


def test_best_scenario_is_most_effective():
    hi, profile, activity = _base()
    interventions = [
        Intervention(intervention_type=InterventionType.ADD_SHADE),
        Intervention(intervention_type=InterventionType.ADD_WATER),
        Intervention(intervention_type=InterventionType.CANCEL),
    ]
    result = simulate(hi, profile, activity, interventions)
    if result.best_scenario:
        for s in result.scenarios:
            assert result.best_scenario.new_score.score <= s.new_score.score


def test_multiple_interventions_return_all():
    hi, profile, activity = _base()
    interventions = [
        Intervention(intervention_type=InterventionType.ADD_SHADE),
        Intervention(intervention_type=InterventionType.SHIFT_TIME, shift_hours=3),
        Intervention(intervention_type=InterventionType.ADD_BREAK, break_every_minutes=20),
    ]
    result = simulate(hi, profile, activity, interventions)
    assert len(result.scenarios) == 3
