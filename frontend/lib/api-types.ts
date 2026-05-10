// TypeScript types mirroring app/data/schemas.py + app/data/stations.py

// ---------------------------------------------------------------------------
// Enums (string unions matching Python str Enum values)
// ---------------------------------------------------------------------------

export type ProfileID =
  | "student_primary"
  | "student_secondary"
  | "student_university"
  | "outdoor_worker"
  | "outdoor_worker_heavy"
  | "athlete"
  | "elderly"
  | "general_adult";

export type ActivityIntensity = "rest" | "low" | "moderate" | "high" | "very_high";

export type RiskClass = "Low" | "Moderate" | "High" | "Critical";

export type Audience = "teacher" | "supervisor" | "parent" | "worker";

export type AgeGroup = "child" | "youth" | "adult" | "elderly";

// ---------------------------------------------------------------------------
// Station
// ---------------------------------------------------------------------------

export interface Station {
  station_id: string;
  name_th: string;
  lat: number;
  lon: number;
  elevation_m: number;
}

// ---------------------------------------------------------------------------
// Vulnerability Profile
// ---------------------------------------------------------------------------

export interface VulnerabilityProfile {
  profile_id: ProfileID;
  display_name_th: string;
  age_group: AgeGroup;
  base_vulnerability: number;
  acclimatization_factor: number;
  notes: string;
}

// ---------------------------------------------------------------------------
// Heat Index
// ---------------------------------------------------------------------------

export interface HeatIndexRequest {
  temperature_c: number;
  humidity_rh: number;
}

export interface HeatIndexResponse {
  heat_index_c: number;
  category: string;
  temperature_c: number;
  humidity_rh: number;
}

// ---------------------------------------------------------------------------
// Heatwave Events
// ---------------------------------------------------------------------------

export interface DailyObsIn {
  date: string; // YYYY-MM-DD
  max_value: number;
  min_value: number;
}

export interface HeatwaveDetectRequest {
  station_id: string;
  observations: DailyObsIn[];
  historical_values: number[];
  percentile?: number;
  min_consecutive_days?: number;
  require_warm_nights?: boolean;
  metric?: string;
}

export interface HeatwaveDetectResponse {
  is_heatwave: boolean;
  event_type: string;
  percentile_threshold: number;
  consecutive_days: number;
  consecutive_nights: number;
  trigger_reason: string[];
  local_baseline: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// Risk Score
// ---------------------------------------------------------------------------

export interface RiskScoreRequest {
  temperature_c: number;
  humidity_rh: number;
  profile_id: ProfileID;
  activity_intensity: ActivityIntensity;
  duration_minutes: number;
  shade_available?: boolean;
  water_access?: boolean;
  time_of_day_hour?: number;
  acclimatized?: boolean;
  low_confidence?: boolean;
}

export interface RiskScoreResponse {
  score: number;
  risk_class: RiskClass;
  dominant_factors: string[];
  confidence: number;
  heat_index_c: number;
  profile_id: string;
  conservative_applied: boolean;
}

// ---------------------------------------------------------------------------
// What-if
// ---------------------------------------------------------------------------

export interface InterventionIn {
  intervention_type: string;
  shift_hours?: number;
  break_every_minutes?: number;
  add_shade?: boolean;
  add_water?: boolean;
  reduce_duration_by?: number;
  new_intensity?: ActivityIntensity | null;
}

export interface WhatIfRequest {
  temperature_c: number;
  humidity_rh: number;
  profile_id: ProfileID;
  activity_intensity: ActivityIntensity;
  duration_minutes: number;
  shade_available?: boolean;
  water_access?: boolean;
  time_of_day_hour?: number;
  acclimatized?: boolean;
  interventions: InterventionIn[];
}

export interface ScenarioOut {
  intervention_type: string;
  original_score: number;
  original_class: string;
  new_score: number;
  new_class: string;
  score_reduction: number;
  effective: boolean;
  summary_th: string;
}

export interface WhatIfResponse {
  original_risk_score: number;
  original_risk_class: string;
  scenarios: ScenarioOut[];
  best_intervention: string | null;
  best_score_reduction: number | null;
}

// ---------------------------------------------------------------------------
// Action Card
// ---------------------------------------------------------------------------

export interface ActionCardRequest {
  temperature_c: number;
  humidity_rh: number;
  profile_id: ProfileID;
  activity_intensity: ActivityIntensity;
  duration_minutes: number;
  shade_available?: boolean;
  water_access?: boolean;
  time_of_day_hour?: number;
  acclimatized?: boolean;
  audience?: Audience;
  forecast_horizon_h?: number;
}

export interface ActionCardResponse {
  risk_class: string;
  audience: string;
  headline_th: string;
  immediate_actions: string[];
  monitoring_points: string[];
  when_to_escalate: string;
  valid_until_hint: string;
  confidence_note: string;
  dominant_factors: string[];
}

// ---------------------------------------------------------------------------
// Forecast
// ---------------------------------------------------------------------------

export interface ForecastPoint {
  station_id: string;
  ts_utc: string; // ISO datetime string
  horizon_h: number;
  heat_index_c: number;
  temp_c: number;
  rh: number;
  pi_lower: number | null;
  pi_upper: number | null;
  model_version: string;
}

export interface ForecastRequest {
  station_id: string;
  horizons?: number[];
}

export interface ForecastResponse {
  station_id: string;
  generated_at: string;
  forecasts: ForecastPoint[];
  model_version: string;
  low_confidence: boolean;
  confidence_reason: string | null;
}
