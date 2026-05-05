"""
What-if Simulator

จำลองผลของการปรับแผนกิจกรรม เช่น:
  - เลื่อนเวลาเริ่มต้น
  - เพิ่มจุดพักน้ำ / เพิ่ม shade
  - ลดระยะเวลา
  - ย้ายไปพื้นที่ที่มีร่มเงา

ทุก scenario recalculate ผ่าน risk_scoring engine เดิม
ผู้ใช้เห็นชัดว่า risk ลดจากอะไร (Explainable)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from app.core.heat_index import HeatIndexResult, compute as compute_heat_index
from app.core.risk_scoring import (
    ActivityContext,
    ActivityIntensity,
    RiskClass,
    RiskScore,
    score as compute_score,
)
from app.core.vulnerability import VulnerabilityProfile


class InterventionType(str, Enum):
    SHIFT_TIME = "shift_time"            # เลื่อนเวลาเริ่มกิจกรรม
    ADD_BREAK = "add_break"              # เพิ่มพักน้ำ/ลดระยะเวลาต่อเนื่อง
    ADD_SHADE = "add_shade"              # ย้ายไปพื้นที่มีร่มเงา
    ADD_WATER = "add_water"              # เพิ่มจุดดื่มน้ำ
    REDUCE_DURATION = "reduce_duration"  # ลดระยะเวลากิจกรรม
    REDUCE_INTENSITY = "reduce_intensity"  # ลด intensity กิจกรรม
    CANCEL = "cancel"                    # ยกเลิกกิจกรรม


@dataclass(frozen=True)
class Intervention:
    intervention_type: InterventionType
    # Params ตาม type
    shift_hours: float = 0.0            # สำหรับ SHIFT_TIME
    break_every_minutes: int = 20       # สำหรับ ADD_BREAK (แบ่งเป็น segments)
    add_shade: bool = False             # สำหรับ ADD_SHADE
    add_water: bool = False             # สำหรับ ADD_WATER
    reduce_duration_by: int = 0         # นาทีที่ลด สำหรับ REDUCE_DURATION
    new_intensity: ActivityIntensity | None = None  # สำหรับ REDUCE_INTENSITY


@dataclass(frozen=True)
class ScenarioResult:
    intervention: Intervention
    original_score: RiskScore
    new_score: RiskScore
    risk_class_change: str             # เช่น "Critical → High"
    score_reduction: float             # positive = ดีขึ้น
    summary_th: str                    # คำอธิบายภาษาไทย
    effective: bool                    # ลด risk class ได้หรือไม่


@dataclass(frozen=True)
class WhatIfResult:
    scenarios: list[ScenarioResult]
    best_scenario: ScenarioResult | None
    original_risk: RiskScore


def simulate(
    hi_result: HeatIndexResult,
    profile: VulnerabilityProfile,
    base_activity: ActivityContext,
    interventions: Sequence[Intervention],
    acclimatized: bool = False,
) -> WhatIfResult:
    """
    รัน what-if scenarios ทั้งหมดและเปรียบเทียบกับ baseline

    Args:
        hi_result: ผล heat index ปัจจุบัน (หรือคาดการณ์)
        profile: vulnerability profile ของกลุ่มเป้าหมาย
        base_activity: กิจกรรมต้นฉบับ
        interventions: รายการ intervention ที่ต้องการทดลอง
        acclimatized: ร่างกายชินความร้อนหรือไม่

    Returns:
        WhatIfResult พร้อม best_scenario
    """
    original = compute_score(hi_result, profile, base_activity, acclimatized)
    results: list[ScenarioResult] = []

    for interv in interventions:
        new_activity, new_hi = _apply_intervention(interv, base_activity, hi_result)
        new_score = compute_score(new_hi, profile, new_activity, acclimatized)
        results.append(_build_result(interv, original, new_score))

    effective = [r for r in results if r.effective]
    best = min(effective, key=lambda r: r.new_score.score) if effective else None

    return WhatIfResult(
        scenarios=results,
        best_scenario=best,
        original_risk=original,
    )


def _apply_intervention(
    interv: Intervention,
    activity: ActivityContext,
    hi: HeatIndexResult,
) -> tuple[ActivityContext, HeatIndexResult]:
    """แปลง intervention เป็น ActivityContext ใหม่"""
    new_hour = activity.time_of_day_hour
    new_duration = activity.duration_minutes
    new_shade = activity.shade_available
    new_water = activity.water_access
    new_intensity = activity.intensity

    if interv.intervention_type == InterventionType.SHIFT_TIME:
        new_hour = int(activity.time_of_day_hour + interv.shift_hours) % 24

    elif interv.intervention_type == InterventionType.ADD_BREAK:
        # พัก 5 นาทีทุก N นาที → ลด effective exposure duration
        break_interval = interv.break_every_minutes
        segments = max(1, activity.duration_minutes // break_interval)
        total_break = segments * 5
        new_duration = max(10, activity.duration_minutes - total_break)

    elif interv.intervention_type == InterventionType.ADD_SHADE:
        new_shade = True

    elif interv.intervention_type == InterventionType.ADD_WATER:
        new_water = True

    elif interv.intervention_type == InterventionType.REDUCE_DURATION:
        new_duration = max(0, activity.duration_minutes - interv.reduce_duration_by)

    elif interv.intervention_type == InterventionType.REDUCE_INTENSITY:
        if interv.new_intensity is not None:
            new_intensity = interv.new_intensity

    elif interv.intervention_type == InterventionType.CANCEL:
        new_duration = 0
        new_intensity = ActivityIntensity.REST

    new_activity = ActivityContext(
        intensity=new_intensity,
        duration_minutes=new_duration,
        shade_available=new_shade,
        water_access=new_water,
        time_of_day_hour=new_hour,
    )
    return new_activity, hi


def _build_result(
    interv: Intervention,
    original: RiskScore,
    new_score: RiskScore,
) -> ScenarioResult:
    _class_order = [RiskClass.LOW, RiskClass.MODERATE, RiskClass.HIGH, RiskClass.CRITICAL]
    orig_idx = _class_order.index(original.risk_class)
    new_idx = _class_order.index(new_score.risk_class)
    effective = new_idx < orig_idx

    change_str = f"{original.risk_class.value} → {new_score.risk_class.value}"
    reduction = round(original.score - new_score.score, 1)

    summary = _thai_summary(interv, original, new_score, reduction)

    return ScenarioResult(
        intervention=interv,
        original_score=original,
        new_score=new_score,
        risk_class_change=change_str,
        score_reduction=reduction,
        summary_th=summary,
        effective=effective,
    )


def _thai_summary(
    interv: Intervention,
    orig: RiskScore,
    new: RiskScore,
    reduction: float,
) -> str:
    action_map = {
        InterventionType.SHIFT_TIME: f"เลื่อนเวลาเป็น {new.heat_index_c:.1f}°C HI",
        InterventionType.ADD_BREAK: f"เพิ่มพักทุก {interv.break_every_minutes} นาที",
        InterventionType.ADD_SHADE: "ย้ายไปพื้นที่มีร่มเงา",
        InterventionType.ADD_WATER: "เพิ่มจุดดื่มน้ำ",
        InterventionType.REDUCE_DURATION: f"ลดระยะเวลา {interv.reduce_duration_by} นาที",
        InterventionType.REDUCE_INTENSITY: "ลดความหนักของกิจกรรม",
        InterventionType.CANCEL: "ยกเลิกกิจกรรม",
    }
    action = action_map.get(interv.intervention_type, str(interv.intervention_type))
    if reduction > 0:
        return f"{action}: ลด risk score {reduction:.1f} คะแนน ({orig.risk_class.value} → {new.risk_class.value})"
    return f"{action}: risk score ไม่เปลี่ยนแปลง ({orig.risk_class.value})"
