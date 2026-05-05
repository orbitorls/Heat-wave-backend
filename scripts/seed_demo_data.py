"""
Demo data seeder — สร้างข้อมูลทดลองตาม scenario ใน proposal §11

Scenario: โรงเรียน A กรุงเทพฯ
พรุ่งนี้ช่วง 13:00-14:30 มีความเสี่ยงสูงสำหรับนักเรียนประถม
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.core.heat_index import compute as compute_hi
from app.core.risk_scoring import ActivityContext, ActivityIntensity, score
from app.core.vulnerability import get_profile, ProfileID
from app.core.whatif import Intervention, InterventionType, simulate
from app.core.action_card import Audience, generate
from app.core.adaptive_definition import AdaptiveDefinitionEngine, DailyObs
from datetime import date
import random, statistics

random.seed(2024)


def run_demo():
    print("=" * 60)
    print("HeatShield AI — Demo Scenario")
    print("พรุ่งนี้ 13:00-14:30 โรงเรียน A กรุงเทพฯ")
    print("=" * 60)

    # --- Step 1: Heat Index ---
    temp, hum = 37.5, 72
    hi = compute_hi(temp, hum)
    print(f"\n[1] Heat Index")
    print(f"    Temp={temp}°C  Humidity={hum}%")
    print(f"    Heat Index = {hi.heat_index}°C  ({hi.category.value})")

    # --- Step 2: Risk Timeline ---
    print(f"\n[2] Risk Timeline — นักเรียนประถม, พละ 90 นาที")
    for hour in [10, 11, 12, 13, 14, 15, 16]:
        hi_h = compute_hi(temp - (0.5 if hour < 12 or hour > 14 else 0), hum)
        profile = get_profile(ProfileID.STUDENT_PRIMARY)
        activity = ActivityContext(
            intensity=ActivityIntensity.HIGH,
            duration_minutes=90,
            shade_available=False,
            water_access=True,
            time_of_day_hour=hour,
        )
        rs = score(hi_h, profile, activity)
        bar = "█" * int(rs.score / 5)
        print(f"    {hour:02d}:00  {rs.risk_class.value:10s}  {rs.score:5.1f}  {bar}")

    # --- Step 3: Heatwave Event Detection ---
    print(f"\n[3] Heatwave Event Detection — สถานีกรุงเทพฯ")
    engine = AdaptiveDefinitionEngine(percentile=90, min_consecutive_days=2)
    hist = [35 + random.gauss(0, 2.5) for _ in range(365)]
    baseline = engine.build_baseline("BKK_01", hist)
    obs = [
        DailyObs(date=date(2024, 5, i + 1), max_value=baseline.p90 + 3, min_value=baseline.p90 - 2)
        for i in range(3)
    ]
    event = engine.detect(obs, baseline)
    print(f"    Local baseline p90 = {baseline.p90:.1f}°C  p95 = {baseline.p95:.1f}°C")
    print(f"    Is Heatwave: {event.is_heatwave}  (consecutive {event.consecutive_days} days)")
    for r in event.trigger_reason:
        print(f"    ✓ {r}")

    # --- Step 4: What-if Simulation ---
    print(f"\n[4] What-if Simulator — ถ้าเปลี่ยนอะไร risk จะลดได้แค่ไหน?")
    hi_now = compute_hi(37.5, 72)
    profile = get_profile(ProfileID.STUDENT_PRIMARY)
    base_activity = ActivityContext(
        intensity=ActivityIntensity.HIGH,
        duration_minutes=90,
        shade_available=False,
        water_access=False,
        time_of_day_hour=13,
    )
    interventions = [
        Intervention(intervention_type=InterventionType.SHIFT_TIME, shift_hours=3),
        Intervention(intervention_type=InterventionType.ADD_BREAK, break_every_minutes=20),
        Intervention(intervention_type=InterventionType.ADD_SHADE),
        Intervention(intervention_type=InterventionType.ADD_WATER),
        Intervention(intervention_type=InterventionType.REDUCE_DURATION, reduce_duration_by=30),
    ]
    result = simulate(hi_now, profile, base_activity, interventions)
    print(f"    Original: {result.original_risk.risk_class.value} ({result.original_risk.score:.1f})")
    for s in result.scenarios:
        arrow = "✓" if s.effective else " "
        print(f"    {arrow} {s.summary_th}")

    # --- Step 5: Action Card ---
    print(f"\n[5] Action Card — สำหรับครู")
    rs = score(hi_now, profile, base_activity)
    card = generate(rs, profile, base_activity, audience=Audience.TEACHER)
    print(f"    [{card.risk_class.value}] {card.headline_th}")
    print(f"    คำแนะนำ:")
    for a in card.immediate_actions:
        print(f"      • {a}")
    print(f"    เมื่อไหรควรโทร 1669: {card.when_to_escalate}")

    print(f"\n{'=' * 60}")
    print("Demo complete — ระบบทำงานครบทุก layer")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
