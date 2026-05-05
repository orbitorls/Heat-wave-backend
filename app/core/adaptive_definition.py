"""
Adaptive Heatwave Definition Engine

ใช้ local percentile (90th / 95th) ของข้อมูลย้อนหลังในพื้นที่นั้น
+ consecutive days/nights ตามแนวทาง WMO/WHO/UNDRR

ไม่ยึดตัวเลข absolute เดียวทั้งประเทศ เพราะเชียงรายกับสงขลา
มีภูมิอากาศและความทนต่อความร้อนต่างกันมาก
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Sequence


@dataclass(frozen=True)
class LocalBaseline:
    """สถิติพื้นฐานของพื้นที่ (คำนวณจากข้อมูลย้อนหลัง)"""
    station_id: str
    metric: str              # "heat_index" หรือ "temperature"
    mean: float
    p90: float               # 90th percentile
    p95: float               # 95th percentile
    sample_years: int        # จำนวนปีที่ใช้คำนวณ


@dataclass(frozen=True)
class DailyObs:
    """ข้อมูลรายวัน (max daytime + min nighttime)"""
    date: date
    max_value: float         # ค่าสูงสุดช่วงกลางวัน
    min_value: float         # ค่าต่ำสุดช่วงกลางคืน


@dataclass(frozen=True)
class HeatwaveEvent:
    is_heatwave: bool
    event_type: str                   # "local_percentile" | "none"
    percentile_threshold: float       # เช่น 90.0
    consecutive_days: int             # วันที่ max เกิน threshold
    consecutive_nights: int           # คืนที่ min เกิน threshold
    trigger_reason: list[str] = field(default_factory=list)
    local_baseline: LocalBaseline | None = None


class AdaptiveDefinitionEngine:
    """
    ตรวจจับ heatwave event ด้วย local percentile + consecutive days/nights

    Parameters
    ----------
    percentile : float
        percentile ที่ใช้เป็น threshold (default 90.0, strict=95.0)
    min_consecutive_days : int
        จำนวนวันต่อเนื่องที่ max ต้องเกิน p90/p95 (default 2)
    require_warm_nights : bool
        ถ้า True: ต้องการ min nighttime เกิน threshold ด้วย (WMO strict)
    """

    def __init__(
        self,
        percentile: float = 90.0,
        min_consecutive_days: int = 2,
        require_warm_nights: bool = False,
    ) -> None:
        if not (50 <= percentile <= 99):
            raise ValueError("percentile must be between 50 and 99")
        self.percentile = percentile
        self.min_consecutive_days = min_consecutive_days
        self.require_warm_nights = require_warm_nights

    def build_baseline(
        self,
        station_id: str,
        historical_values: Sequence[float],
        metric: str = "heat_index",
    ) -> LocalBaseline:
        """
        สร้าง local baseline จาก historical observations

        Args:
            station_id: รหัสสถานี
            historical_values: ลำดับค่า daily max ย้อนหลัง (หน่วยเดียวกับที่ใช้ detect)
            metric: "heat_index" หรือ "temperature"
        """
        if len(historical_values) < 30:
            raise ValueError(
                f"Need at least 30 observations for a reliable baseline, got {len(historical_values)}"
            )
        import statistics

        sorted_vals = sorted(historical_values)
        n = len(sorted_vals)

        def percentile_val(p: float) -> float:
            idx = (p / 100) * (n - 1)
            lo, hi = int(idx), min(int(idx) + 1, n - 1)
            return sorted_vals[lo] + (idx - lo) * (sorted_vals[hi] - sorted_vals[lo])

        return LocalBaseline(
            station_id=station_id,
            metric=metric,
            mean=round(statistics.mean(historical_values), 2),
            p90=round(percentile_val(90), 2),
            p95=round(percentile_val(95), 2),
            sample_years=max(1, n // 365),
        )

    def detect(
        self,
        observations: Sequence[DailyObs],
        baseline: LocalBaseline,
    ) -> HeatwaveEvent:
        """
        ตรวจจับว่าช่วง observations เป็น heatwave event หรือไม่

        Args:
            observations: ข้อมูลรายวัน เรียงตามวันที่ (ล่าสุดอยู่ท้าย)
            baseline: LocalBaseline ของพื้นที่นั้น

        Returns:
            HeatwaveEvent ที่บอกว่าเป็น heatwave + เหตุผล
        """
        threshold = baseline.p90 if self.percentile == 90 else baseline.p95

        day_streak = _count_trailing_streak(
            [obs.max_value > threshold for obs in observations]
        )
        night_streak = _count_trailing_streak(
            [obs.min_value > threshold for obs in observations]
        )

        reasons: list[str] = []
        days_met = day_streak >= self.min_consecutive_days
        nights_met = night_streak >= self.min_consecutive_days

        if days_met:
            reasons.append(
                f"max >{threshold:.1f}°C ต่อเนื่อง {day_streak} วัน "
                f"(p{self.percentile:.0f} local)"
            )
        if nights_met and self.require_warm_nights:
            reasons.append(
                f"min >{threshold:.1f}°C ต่อเนื่อง {night_streak} คืน"
            )

        if self.require_warm_nights:
            is_hw = days_met and nights_met
        else:
            is_hw = days_met

        return HeatwaveEvent(
            is_heatwave=is_hw,
            event_type="local_percentile" if is_hw else "none",
            percentile_threshold=self.percentile,
            consecutive_days=day_streak,
            consecutive_nights=night_streak,
            trigger_reason=reasons,
            local_baseline=baseline,
        )


def _count_trailing_streak(flags: Sequence[bool]) -> int:
    """นับจำนวน True ต่อเนื่องจากท้ายลำดับ"""
    count = 0
    for flag in reversed(flags):
        if flag:
            count += 1
        else:
            break
    return count
