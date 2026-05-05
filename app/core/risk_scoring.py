"""
Heat-Health Risk Scoring Engine

แปลง heat index + กิจกรรม + ระยะเวลา + vulnerability profile
ให้เป็น risk_score (0–100) และ risk_class (Low/Moderate/High/Critical)

ออกแบบให้ Explainable: บอก dominant_factors เสมอ
Conservative by design: uncertainty สูง → ส่ง class สูงขึ้นหนึ่งระดับ
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from app.core.heat_index import HeatIndexCategory, HeatIndexResult
from app.core.vulnerability import VulnerabilityProfile, effective_vulnerability


class ActivityIntensity(str, Enum):
    REST = "rest"           # นั่งเรียน/ประชุม
    LOW = "low"             # เดินเบาๆ
    MODERATE = "moderate"   # พลศึกษา, งานเบา
    HIGH = "high"           # วิ่ง, งานหนัก
    VERY_HIGH = "very_high" # กีฬาแข่งขัน, ก่อสร้างหนัก


class RiskClass(str, Enum):
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"
    CRITICAL = "Critical"


@dataclass(frozen=True)
class ActivityContext:
    intensity: ActivityIntensity
    duration_minutes: int      # ระยะเวลากิจกรรม
    shade_available: bool = False
    water_access: bool = True
    time_of_day_hour: int = 13  # ชั่วโมง (0-23) ที่เริ่มกิจกรรม


@dataclass(frozen=True)
class RiskScore:
    score: float               # 0–100
    risk_class: RiskClass
    dominant_factors: list[str]
    confidence: float          # 0–1
    heat_index_c: float
    profile_id: str
    intensity: str
    duration_minutes: int
    conservative_applied: bool = False


# ---------------------------------------------------------------------------
# Scoring weights
# ---------------------------------------------------------------------------

# Heat Index category → base score
_HI_BASE: dict[HeatIndexCategory, float] = {
    HeatIndexCategory.CAUTION: 25.0,
    HeatIndexCategory.EXTREME_CAUTION: 50.0,
    HeatIndexCategory.DANGER: 75.0,
    HeatIndexCategory.EXTREME_DANGER: 95.0,
}

# Activity intensity multiplier (คูณกับ base_score ก่อน vulnerability)
_INTENSITY_MULT: dict[ActivityIntensity, float] = {
    ActivityIntensity.REST: 0.50,
    ActivityIntensity.LOW: 0.70,
    ActivityIntensity.MODERATE: 1.00,
    ActivityIntensity.HIGH: 1.30,
    ActivityIntensity.VERY_HIGH: 1.60,
}

# Duration exposure factor — exposure > 60 นาทีเพิ่ม risk
def _duration_factor(minutes: int) -> float:
    if minutes <= 20:
        return 0.70
    if minutes <= 40:
        return 0.85
    if minutes <= 60:
        return 1.00
    if minutes <= 90:
        return 1.15
    return 1.30


# เวลาสูงสุดช่วงกลางวัน 11:00-15:00 → เพิ่ม 10%
def _time_of_day_factor(hour: int) -> float:
    if 11 <= hour <= 15:
        return 1.10
    if 10 <= hour <= 16:
        return 1.05
    return 1.00


# Shade ลด 10%, ไม่มีน้ำดื่ม +15%
def _env_adjustment(shade: bool, water: bool) -> float:
    adj = 1.0
    if shade:
        adj -= 0.10
    if not water:
        adj += 0.15
    return adj


def score(
    hi_result: HeatIndexResult,
    profile: VulnerabilityProfile,
    activity: ActivityContext,
    acclimatized: bool = False,
    low_confidence: bool = False,
) -> RiskScore:
    """
    คำนวณ Heat-Health Risk Score

    Args:
        hi_result: ผลจาก heat_index.compute()
        profile: VulnerabilityProfile ของกลุ่มเป้าหมาย
        activity: บริบทกิจกรรม (intensity, duration, shade, water, time)
        acclimatized: ร่างกายชินความร้อนหรือไม่
        low_confidence: ถ้า True → ใช้ conservative bump

    Returns:
        RiskScore พร้อม dominant_factors และ confidence
    """
    base = _HI_BASE[hi_result.category]
    vuln = effective_vulnerability(profile, acclimatized)
    intensity_mult = _INTENSITY_MULT[activity.intensity]
    dur_factor = _duration_factor(activity.duration_minutes)
    time_factor = _time_of_day_factor(activity.time_of_day_hour)
    env_factor = _env_adjustment(activity.shade_available, activity.water_access)

    raw = base * intensity_mult * dur_factor * time_factor * env_factor

    # ผสม vulnerability: weighted average ระหว่าง raw และ raw*vuln
    # vulnerability เป็น modifier ไม่ใช่ multiplier ตรงๆ เพื่อให้ score ไม่เกิน 100
    raw_with_vuln = raw * (0.5 + 0.5 * vuln)

    final_score = min(100.0, round(raw_with_vuln, 1))
    risk_class = _classify(final_score)

    # Conservative bump เมื่อ confidence ต่ำ
    conservative_applied = False
    if low_confidence and risk_class != RiskClass.CRITICAL:
        risk_class = _bump_class(risk_class)
        conservative_applied = True

    factors = _explain(hi_result, profile, activity, acclimatized, vuln)

    return RiskScore(
        score=final_score,
        risk_class=risk_class,
        dominant_factors=factors,
        confidence=0.6 if low_confidence else 0.85,
        heat_index_c=hi_result.heat_index,
        profile_id=profile.profile_id.value,
        intensity=activity.intensity.value,
        duration_minutes=activity.duration_minutes,
        conservative_applied=conservative_applied,
    )


def _classify(s: float) -> RiskClass:
    if s >= 80:
        return RiskClass.CRITICAL
    if s >= 60:
        return RiskClass.HIGH
    if s >= 35:
        return RiskClass.MODERATE
    return RiskClass.LOW


def _bump_class(rc: RiskClass) -> RiskClass:
    order = [RiskClass.LOW, RiskClass.MODERATE, RiskClass.HIGH, RiskClass.CRITICAL]
    idx = order.index(rc)
    return order[min(idx + 1, len(order) - 1)]


def _explain(
    hi: HeatIndexResult,
    profile: VulnerabilityProfile,
    activity: ActivityContext,
    acclimatized: bool,
    vuln: float,
) -> list[str]:
    factors = []
    factors.append(f"Heat index {hi.heat_index}°C ({hi.category.value})")

    if hi.humidity_rh >= 70:
        factors.append(f"ความชื้นสูง ({hi.humidity_rh:.0f}%)")

    if activity.intensity in (ActivityIntensity.HIGH, ActivityIntensity.VERY_HIGH):
        factors.append(f"กิจกรรมหนัก ({activity.intensity.value})")

    if activity.duration_minutes > 60:
        factors.append(f"ระยะเวลานาน ({activity.duration_minutes} นาที)")

    if 11 <= activity.time_of_day_hour <= 15:
        factors.append(f"ช่วงเวลาร้อนจัด ({activity.time_of_day_hour}:00 น.)")

    if vuln >= 0.75:
        factors.append(f"กลุ่มเปราะบางสูง ({profile.display_name_th})")

    if not activity.water_access:
        factors.append("ไม่มีจุดดื่มน้ำ")

    if not activity.shade_available and hi.heat_index >= 33:
        factors.append("ไม่มีร่มเงา")

    if acclimatized:
        factors.append("ร่างกายชินความร้อน (ลด risk)")

    return factors
