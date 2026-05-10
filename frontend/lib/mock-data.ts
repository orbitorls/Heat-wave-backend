// Mock data for standalone frontend development
import type {
  Station,
  ForecastResponse,
  RiskScoreResponse,
  WhatIfResponse,
  ActionCardResponse,
  HeatwaveDetectResponse,
  RiskClass,
} from "./api-types";

// Station IDs must match app/data/stations.py and lib/stations.ts
export const MOCK_STATIONS: Station[] = [
  { station_id: "BKK_01", name_th: "กรุงเทพมหานคร (Don Mueang)", lat: 13.9132, lon: 100.6067, elevation_m: 9.5 },
  { station_id: "CNX_01", name_th: "เชียงใหม่", lat: 18.7761, lon: 98.9769, elevation_m: 310.0 },
  { station_id: "KKN_01", name_th: "ขอนแก่น", lat: 16.4419, lon: 102.8359, elevation_m: 182.0 },
  { station_id: "HYI_01", name_th: "หาดใหญ่ (สงขลา)", lat: 6.9269, lon: 100.437, elevation_m: 8.0 },
  { station_id: "RYG_01", name_th: "ระยอง", lat: 12.6815, lon: 101.2816, elevation_m: 14.0 },
];

export function createMockForecast(stationId: string): ForecastResponse {
  const now = new Date();
  const forecasts = [];
  
  for (let i = 0; i < 24; i += 3) {
    const horizonH = i;
    const heatIndex = 32 + Math.random() * 8 + (horizonH * 0.1);
    forecasts.push({
      station_id: stationId,
      ts_utc: new Date(now.getTime() + horizonH * 3600000).toISOString(),
      horizon_h: horizonH,
      heat_index_c: Math.round(heatIndex * 10) / 10,
      temp_c: Math.round((heatIndex - 2) * 10) / 10,
      rh: Math.round((60 + Math.random() * 20) * 10) / 10,
      pi_lower: Math.round((heatIndex - 1.5) * 10) / 10,
      pi_upper: Math.round((heatIndex + 1.5) * 10) / 10,
      model_version: "lgbm-v1.0.0",
    });
  }

  return {
    station_id: stationId,
    generated_at: now.toISOString(),
    forecasts,
    model_version: "lgbm-v1.0.0",
    low_confidence: false,
    confidence_reason: null,
  };
}

export function createMockRiskScore(temperature_c: number, humidity_rh: number): RiskScoreResponse {
  const heatIndex = temperature_c + (humidity_rh - 50) * 0.05;
  let riskClass: RiskClass = "Low";
  
  if (heatIndex > 40) riskClass = "Critical";
  else if (heatIndex > 37) riskClass = "High";
  else if (heatIndex > 33) riskClass = "Moderate";

  return {
    score: Math.round(heatIndex * 2.5),
    risk_class: riskClass,
    dominant_factors: ["temperature", "humidity"],
    confidence: 0.92,
    heat_index_c: Math.round(heatIndex * 10) / 10,
    profile_id: "general_adult",
    conservative_applied: true,
  };
}

export function createMockWhatIf(baseScore: number): WhatIfResponse {
  return {
    original_risk_score: baseScore,
    original_risk_class: baseScore > 80 ? "Critical" : baseScore > 60 ? "High" : "Moderate",
    scenarios: [
      {
        intervention_type: "shift_time",
        original_score: baseScore,
        original_class: baseScore > 80 ? "Critical" : "High",
        new_score: Math.max(0, baseScore - 15),
        new_class: "Moderate",
        score_reduction: 15,
        effective: true,
        summary_th: "เลื่อนเวลาทำงานลดความเสี่ยงได้ดี",
      },
      {
        intervention_type: "add_shade",
        original_score: baseScore,
        original_class: baseScore > 80 ? "Critical" : "High",
        new_score: Math.max(0, baseScore - 10),
        new_class: "Moderate",
        score_reduction: 10,
        effective: true,
        summary_th: "เพิ่มร่มเงาลดความร้อนได้",
      },
    ],
    best_intervention: "shift_time",
    best_score_reduction: 15,
  };
}

export function createMockActionCard(riskClass: string): ActionCardResponse {
  return {
    risk_class: riskClass,
    audience: "teacher",
    headline_th: riskClass === "Critical" 
      ? "อันตราย: หยุดกิจกรรมกลางแจ้งทันที" 
      : "ระวัง: จำกัดเวลากิจกรรมกลางแจ้ง",
    immediate_actions: [
      "ให้นักเรียนอยู่ในร่มเงา",
      "จัดหาน้ำดื่มเพียงพอ",
      "หยุดกิจกรรมที่ต้องใช้แรงมาก",
    ],
    monitoring_points: [
      "สังเกตอาการอ่อนเพลียง",
      "ตรวจสอบอาการปวดศีรษะ",
    ],
    when_to_escalate: "หากมีนักเรียนมีอาการเวียนศีรษะหรือหน้ามืด",
    valid_until_hint: "2 ชั่วโมงถัดไป",
    confidence_note: "ข้อมูลมีความแม่นยำสูง",
    dominant_factors: ["temperature", "humidity", "no shade"],
  };
}

export function createMockHeatwaveDetect(stationId: string): HeatwaveDetectResponse {
  const isHeatwave = Math.random() > 0.5;
  
  return {
    is_heatwave: isHeatwave,
    event_type: isHeatwave ? "moderate" : "none",
    percentile_threshold: 90,
    consecutive_days: isHeatwave ? 3 : 0,
    consecutive_nights: isHeatwave ? 2 : 0,
    trigger_reason: isHeatwave 
      ? ["temperature_exceeds_threshold", "consecutive_hot_days"]
      : [],
    local_baseline: {
      p90: 38.5,
      p95: 40.2,
    },
  };
}
