"""
Action Card Generator

สร้างคำแนะนำเชิงปฏิบัติภาษาไทยสำหรับครู/หัวหน้างาน/ผู้ปกครอง
ตาม risk_class + profile + activity context

ใช้ rule-based template ก่อน (ไม่พึ่ง LLM ใน MVP)
ออกแบบให้อ่านแล้วตัดสินใจได้เลย — ไม่ต้องตีความ
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from app.core.risk_scoring import ActivityContext, RiskClass, RiskScore
from app.core.vulnerability import AgeGroup, ProfileID, VulnerabilityProfile


class Audience(str, Enum):
    TEACHER = "teacher"          # ครู/ผู้ดูแลนักเรียน
    SUPERVISOR = "supervisor"    # หัวหน้างาน/ผู้จัดการ
    PARENT = "parent"            # ผู้ปกครอง
    WORKER = "worker"            # ตัวแรงงานเอง


@dataclass(frozen=True)
class ActionCard:
    risk_class: RiskClass
    audience: Audience
    headline_th: str             # หัวข้อหลักสั้นๆ
    immediate_actions: list[str] # สิ่งที่ต้องทำทันที
    monitoring_points: list[str] # สัญญาณที่ต้องติดตาม
    when_to_escalate: str        # เมื่อไหรควรโทรแจ้ง/ส่งโรงพยาบาล
    valid_until_hint: str        # "ใช้ได้ถึง..." เพื่อให้รู้ว่าต้องอัปเดตเมื่อไหร่
    confidence_note: str = ""    # ถ้า conservative → แจ้งให้รู้
    dominant_factors: list[str] = field(default_factory=list)


def generate(
    risk_score: RiskScore,
    profile: VulnerabilityProfile,
    activity: ActivityContext,
    audience: Audience = Audience.TEACHER,
    forecast_horizon_h: int = 0,
) -> ActionCard:
    """
    สร้าง Action Card จาก risk score + context

    Args:
        risk_score: ผลจาก risk_scoring.score()
        profile: vulnerability profile
        activity: บริบทกิจกรรม
        audience: กลุ่มผู้รับ card
        forecast_horizon_h: ล่วงหน้ากี่ชั่วโมง (0 = ปัจจุบัน)
    """
    rc = risk_score.risk_class
    template = _select_template(rc, profile, activity, audience)

    valid_hint = (
        f"พยากรณ์ล่วงหน้า {forecast_horizon_h} ชม. — ตรวจสอบอีกครั้งก่อนกิจกรรม"
        if forecast_horizon_h > 0
        else "ใช้ได้สำหรับช่วงเวลานี้ — อัปเดตทุก 1-3 ชั่วโมง"
    )

    conf_note = (
        "** ระบบใช้ค่า conservative เนื่องจากข้อมูลมีความไม่แน่นอนสูง **"
        if risk_score.conservative_applied
        else ""
    )

    return ActionCard(
        risk_class=rc,
        audience=audience,
        headline_th=template["headline"],
        immediate_actions=template["actions"],
        monitoring_points=template["monitor"],
        when_to_escalate=template["escalate"],
        valid_until_hint=valid_hint,
        confidence_note=conf_note,
        dominant_factors=risk_score.dominant_factors,
    )


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

def _select_template(
    rc: RiskClass,
    profile: VulnerabilityProfile,
    activity: ActivityContext,
    audience: Audience,
) -> dict:
    is_child = profile.age_group == AgeGroup.CHILD
    is_elderly = profile.age_group == AgeGroup.ELDERLY
    is_worker = profile.profile_id in (
        ProfileID.OUTDOOR_WORKER,
        ProfileID.OUTDOOR_WORKER_HEAVY,
    )

    if rc == RiskClass.LOW:
        return _template_low(audience, is_child)
    if rc == RiskClass.MODERATE:
        return _template_moderate(audience, is_child, is_elderly)
    if rc == RiskClass.HIGH:
        return _template_high(audience, is_child, is_elderly, is_worker)
    return _template_critical(audience, is_child, is_elderly, is_worker)


def _template_low(audience: Audience, is_child: bool) -> dict:
    return {
        "headline": "ความเสี่ยงต่ำ — ดำเนินกิจกรรมได้ตามปกติ",
        "actions": [
            "แจ้งผู้เข้าร่วมให้ดื่มน้ำก่อนและระหว่างกิจกรรม",
            "สังเกตอาการผิดปกติเบื้องต้น เช่น หน้าแดง เวียนหัว",
        ],
        "monitor": [
            "อุณหภูมิและความชื้นไม่เพิ่มขึ้นอย่างรวดเร็ว",
            "ไม่มีผู้แสดงอาการไม่สบาย",
        ],
        "escalate": "ถ้ามีผู้หมดสติหรือชักให้โทร 1669 ทันที",
    }


def _template_moderate(audience: Audience, is_child: bool, is_elderly: bool) -> dict:
    child_note = " โดยเฉพาะเด็กเล็กที่อาจไม่รายงานความไม่สบาย" if is_child else ""
    return {
        "headline": "ความเสี่ยงปานกลาง — ดำเนินกิจกรรมได้แต่ต้องระมัดระวัง",
        "actions": [
            f"เพิ่มพักน้ำทุก 20 นาที{child_note}",
            "ย้ายกิจกรรมไปพื้นที่ร่มเงาถ้าทำได้",
            "ลดความหนักของกิจกรรมลง 20–30%",
            "แจ้งผู้เข้าร่วมให้หยุดทันทีถ้ารู้สึกไม่ดี",
        ],
        "monitor": [
            "สีหน้าและเหงื่อออกของผู้เข้าร่วม",
            "ค่า heat index ว่าเพิ่มขึ้นเกิน 35°C หรือไม่",
        ],
        "escalate": "ถ้ามีผู้มีอาการตะคริว/หน้ามืด/หยุดเหงื่อ → ย้ายเข้าที่เย็นและโทร 1669",
    }


def _template_high(
    audience: Audience,
    is_child: bool,
    is_elderly: bool,
    is_worker: bool,
) -> dict:
    if is_worker and audience == Audience.SUPERVISOR:
        actions = [
            "ใช้ work-rest schedule 20 นาทีทำงาน / 10 นาทีพัก",
            "ตั้งจุดน้ำดื่มทุก 50 เมตร",
            "ห้ามทำงานคนเดียว — จับคู่สังเกตกัน",
            "ตรวจสอบ PPE ว่าระบายความร้อนได้",
        ]
    elif is_child:
        actions = [
            "เลื่อนกิจกรรมกลางแจ้งออกไปหลัง 16:00 น. หรือยกเลิก",
            "ถ้าจำเป็นต้องทำ: ลดเวลาเหลือไม่เกิน 20 นาที",
            "พักเข้าที่ร่มทุก 15 นาที ดื่มน้ำทุกพัก",
            "ครูนับหัวและสังเกตทุก 5 นาที",
        ]
    else:
        actions = [
            "พิจารณาเลื่อนกิจกรรมออกไปก่อน 10:00 หรือหลัง 16:00 น.",
            "ถ้ายังคงทำ: ลดระยะเวลาลง 50% และเพิ่มพักน้ำทุก 15 นาที",
            "จัดพื้นที่ร่มเงาหรือ cooling station",
            "มีผู้รับผิดชอบสังเกตอาการโดยเฉพาะ",
        ]

    return {
        "headline": "ความเสี่ยงสูง — ควรเลื่อนหรือปรับกิจกรรมอย่างจริงจัง",
        "actions": actions,
        "monitor": [
            "อาการตะคริว เวียนหัว คลื่นไส้ หน้าแดงร้อน",
            "ผู้ที่หยุดเหงื่อทั้งที่ยังร้อนมาก (อาการ heatstroke)",
            "ค่า heat index ทุก 30 นาที",
        ],
        "escalate": (
            "ถ้ามีผู้มีไข้สูง สับสน หมดสติ หรือหยุดเหงื่อ → "
            "นำเข้าที่เย็นทันที ประคบน้ำแข็ง แล้วโทร 1669"
        ),
    }


def _template_critical(
    audience: Audience,
    is_child: bool,
    is_elderly: bool,
    is_worker: bool,
) -> dict:
    return {
        "headline": "ความเสี่ยงวิกฤต — ยกเลิกหรืองดกิจกรรมกลางแจ้งทั้งหมด",
        "actions": [
            "ยกเลิกหรือย้ายกิจกรรมกลางแจ้งทั้งหมดเข้าในร่ม",
            "แจ้งผู้เกี่ยวข้องทุกฝ่ายทันที",
            "เปิดห้องแอร์หรือจุด cooling center สำหรับกลุ่มเสี่ยง",
            "เตรียมน้ำเย็น ผ้าชุบน้ำ และแผ่นประคบน้ำแข็งให้พร้อม",
            "มีบุคลากรปฐมพยาบาลประจำจุด",
        ],
        "monitor": [
            "สถานะผู้เปราะบางทุก 10 นาที (เด็กเล็ก ผู้สูงอายุ)",
            "อุณหภูมิภายนอกและ heat index แบบ real-time",
        ],
        "escalate": (
            "โทร 1669 ทันทีถ้ามีอาการ heatstroke: "
            "ไม่มีเหงื่อ + ไข้สูง + สับสน/หมดสติ"
        ),
    }
