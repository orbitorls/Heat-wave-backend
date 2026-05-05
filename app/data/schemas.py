"""Pydantic request/response schemas for all API endpoints"""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator

from app.core.risk_scoring import ActivityIntensity, RiskClass
from app.core.vulnerability import ProfileID
from app.core.action_card import Audience


# ---------------------------------------------------------------------------
# Heat Index
# ---------------------------------------------------------------------------

class HeatIndexRequest(BaseModel):
    temperature_c: float = Field(..., ge=-10, le=60, description="อุณหภูมิ dry-bulb (°C)")
    humidity_rh: float = Field(..., ge=0, le=100, description="ความชื้นสัมพัทธ์ (%)")


class HeatIndexResponse(BaseModel):
    heat_index_c: float
    category: str
    temperature_c: float
    humidity_rh: float


# ---------------------------------------------------------------------------
# Heatwave Event Detection
# ---------------------------------------------------------------------------

class DailyObsIn(BaseModel):
    date: str = Field(..., description="YYYY-MM-DD")
    max_value: float
    min_value: float


class HeatwaveDetectRequest(BaseModel):
    station_id: str
    observations: list[DailyObsIn] = Field(..., min_length=2)
    historical_values: list[float] = Field(
        ..., min_length=30, description="ค่าสูงสุดรายวันย้อนหลังสำหรับสร้าง baseline"
    )
    percentile: float = Field(90.0, ge=50, le=99)
    min_consecutive_days: int = Field(2, ge=1, le=7)
    require_warm_nights: bool = False
    metric: str = "heat_index"


class HeatwaveDetectResponse(BaseModel):
    is_heatwave: bool
    event_type: str
    percentile_threshold: float
    consecutive_days: int
    consecutive_nights: int
    trigger_reason: list[str]
    local_baseline: dict


# ---------------------------------------------------------------------------
# Risk Score
# ---------------------------------------------------------------------------

class RiskScoreRequest(BaseModel):
    temperature_c: float = Field(..., ge=-10, le=60)
    humidity_rh: float = Field(..., ge=0, le=100)
    profile_id: ProfileID
    activity_intensity: ActivityIntensity
    duration_minutes: int = Field(..., ge=0, le=720)
    shade_available: bool = False
    water_access: bool = True
    time_of_day_hour: int = Field(13, ge=0, le=23)
    acclimatized: bool = False
    low_confidence: bool = False


class RiskScoreResponse(BaseModel):
    score: float
    risk_class: RiskClass
    dominant_factors: list[str]
    confidence: float
    heat_index_c: float
    profile_id: str
    conservative_applied: bool


# ---------------------------------------------------------------------------
# What-if
# ---------------------------------------------------------------------------

class InterventionIn(BaseModel):
    intervention_type: str
    shift_hours: float = 0.0
    break_every_minutes: int = 20
    add_shade: bool = False
    add_water: bool = False
    reduce_duration_by: int = 0
    new_intensity: Optional[str] = None


class WhatIfRequest(BaseModel):
    temperature_c: float = Field(..., ge=-10, le=60)
    humidity_rh: float = Field(..., ge=0, le=100)
    profile_id: ProfileID
    activity_intensity: ActivityIntensity
    duration_minutes: int = Field(..., ge=1, le=720)
    shade_available: bool = False
    water_access: bool = True
    time_of_day_hour: int = Field(13, ge=0, le=23)
    acclimatized: bool = False
    interventions: list[InterventionIn] = Field(..., min_length=1)


class ScenarioOut(BaseModel):
    intervention_type: str
    original_score: float
    original_class: str
    new_score: float
    new_class: str
    score_reduction: float
    effective: bool
    summary_th: str


class WhatIfResponse(BaseModel):
    original_risk_score: float
    original_risk_class: str
    scenarios: list[ScenarioOut]
    best_intervention: Optional[str] = None
    best_score_reduction: Optional[float] = None


# ---------------------------------------------------------------------------
# Action Card
# ---------------------------------------------------------------------------

class ActionCardRequest(BaseModel):
    temperature_c: float = Field(..., ge=-10, le=60)
    humidity_rh: float = Field(..., ge=0, le=100)
    profile_id: ProfileID
    activity_intensity: ActivityIntensity
    duration_minutes: int = Field(..., ge=0, le=720)
    shade_available: bool = False
    water_access: bool = True
    time_of_day_hour: int = Field(13, ge=0, le=23)
    acclimatized: bool = False
    audience: Audience = Audience.TEACHER
    forecast_horizon_h: int = Field(0, ge=0, le=72)


class ActionCardResponse(BaseModel):
    risk_class: str
    audience: str
    headline_th: str
    immediate_actions: list[str]
    monitoring_points: list[str]
    when_to_escalate: str
    valid_until_hint: str
    confidence_note: str
    dominant_factors: list[str]


# ---------------------------------------------------------------------------
# Station Observation & Forecast (Data + ML layer)
# ---------------------------------------------------------------------------

class StationObservation(BaseModel):
    station_id: str
    ts_utc: datetime
    temp_c: float = Field(..., ge=-10, le=55)
    rh: float = Field(..., ge=0, le=100)
    wind_ms: float | None = None
    precip_mm: float | None = None
    source: Literal["tmd", "synthetic", "cache", "era5", "nasa_power", "modis"] = "tmd"
    # Extended reanalysis fields — available from ERA5 / NASA POWER
    solar_wm2: float | None = None      # Surface solar radiation downwards (W/m²)
    cloud_cover: float | None = None    # Total cloud cover (0–1 fraction)
    blh_m: float | None = None          # Boundary layer height (m)
    pressure_hpa: float | None = None   # Mean sea-level pressure (hPa)
    lst_c: float | None = None          # Land surface temperature from MODIS (°C)


class ForecastPoint(BaseModel):
    station_id: str
    ts_utc: datetime
    horizon_h: int
    heat_index_c: float
    temp_c: float
    rh: float
    pi_lower: float | None = None
    pi_upper: float | None = None
    model_version: str


class ForecastRequest(BaseModel):
    station_id: str
    horizons: list[int] = Field(default=[6, 24, 48], description="Forecast horizons in hours")
    recent_obs: list[StationObservation] | None = Field(
        None,
        description="Recent observations — if None, loads from local parquet store"
    )


class ForecastResponse(BaseModel):
    station_id: str
    generated_at: datetime
    forecasts: list[ForecastPoint]
    model_version: str
    low_confidence: bool = False
    confidence_reason: str | None = None
