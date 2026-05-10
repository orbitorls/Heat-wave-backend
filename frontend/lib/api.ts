import type {
  Station,
  VulnerabilityProfile,
  HeatIndexRequest,
  HeatIndexResponse,
  RiskScoreRequest,
  RiskScoreResponse,
  ForecastRequest,
  ForecastResponse,
  ActionCardRequest,
  ActionCardResponse,
  WhatIfRequest,
  WhatIfResponse,
  HeatwaveDetectRequest,
  HeatwaveDetectResponse,
} from "./api-types";
import {
  MOCK_STATIONS,
  createMockForecast,
  createMockRiskScore,
  createMockWhatIf,
  createMockActionCard,
  createMockHeatwaveDetect,
} from "./mock-data";

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const USE_MOCK = process.env.NEXT_PUBLIC_USE_MOCK === "true";

class ApiError extends Error {
  constructor(
    public status: number,
    public detail: unknown,
  ) {
    super(`API ${status}: ${JSON.stringify(detail)}`);
    this.name = "ApiError";
  }
}

async function post<T>(path: string, body: unknown): Promise<T> {
  if (USE_MOCK) {
    return mockHandler(path, body) as Promise<T>;
  }
  
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => res.statusText);
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

async function get<T>(path: string): Promise<T> {
  if (USE_MOCK) {
    return mockHandler(path) as Promise<T>;
  }
  
  const res = await fetch(`${BASE}${path}`, { method: "GET" });
  if (!res.ok) {
    const detail = await res.json().catch(() => res.statusText);
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

// Mock data handler
function mockHandler(path: string, body?: unknown): unknown {
  if (path === "/stations") {
    return MOCK_STATIONS;
  }
  
  if (path === "/profiles") {
    return [
      { profile_id: "general_adult", display_name_th: "ผู้ใหญ่ทั่วไป", age_group: "adult", base_vulnerability: 0.5, acclimatization_factor: 1.0, notes: "" },
      { profile_id: "student_primary", display_name_th: "นักเรียนประถม", age_group: "child", base_vulnerability: 0.8, acclimatization_factor: 0.9, notes: "" },
      { profile_id: "outdoor_worker", display_name_th: "แรงงานกลางแจ้ง", age_group: "adult", base_vulnerability: 0.7, acclimatization_factor: 1.0, notes: "" },
    ];
  }
  
  if (path === "/risk/score" && body) {
    const req = body as RiskScoreRequest;
    return createMockRiskScore(req.temperature_c, req.humidity_rh);
  }
  
  if (path === "/forecast/heat-index" && body) {
    const req = body as ForecastRequest;
    return createMockForecast(req.station_id);
  }
  
  if (path === "/whatif/simulate" && body) {
    const req = body as WhatIfRequest;
    const baseScore = createMockRiskScore(req.temperature_c, req.humidity_rh).score;
    return createMockWhatIf(baseScore);
  }
  
  if (path === "/action-card" && body) {
    const req = body as ActionCardRequest;
    const riskClass = createMockRiskScore(req.temperature_c, req.humidity_rh).risk_class;
    return createMockActionCard(riskClass);
  }
  
  if (path.startsWith("/events/auto-detect")) {
    const urlParams = new URLSearchParams(path.split("?")[1]);
    const stationId = urlParams.get("station_id") || "BKK001";
    return createMockHeatwaveDetect(stationId);
  }
  
  if (path === "/events/detect" && body) {
    const req = body as HeatwaveDetectRequest;
    return createMockHeatwaveDetect(req.station_id);
  }
  
  throw new Error(`Mock handler not implemented for: ${path}`);
}

// ---------------------------------------------------------------------------
// Public endpoints
// ---------------------------------------------------------------------------

export const api = {
  stations: (): Promise<Station[]> =>
    get("/stations"),

  profiles: (): Promise<VulnerabilityProfile[]> =>
    get("/profiles"),

  heatIndex: (req: HeatIndexRequest): Promise<HeatIndexResponse> =>
    post("/heat-index", req),

  riskScore: (req: RiskScoreRequest): Promise<RiskScoreResponse> =>
    post("/risk/score", req),

  forecast: (req: ForecastRequest): Promise<ForecastResponse> =>
    post("/forecast/heat-index", req),

  actionCard: (req: ActionCardRequest): Promise<ActionCardResponse> =>
    post("/action-card", req),

  whatIf: (req: WhatIfRequest): Promise<WhatIfResponse> =>
    post("/whatif/simulate", req),

  events: (req: HeatwaveDetectRequest): Promise<HeatwaveDetectResponse> =>
    post("/events/detect", req),

  autoDetect: (stationId: string, params?: { days_back?: number; baseline_days?: number; percentile?: number; require_warm_nights?: boolean }): Promise<HeatwaveDetectResponse> =>
    get(`/events/auto-detect?station_id=${stationId}${params?.days_back ? `&days_back=${params.days_back}` : ""}${params?.baseline_days ? `&baseline_days=${params.baseline_days}` : ""}${params?.percentile ? `&percentile=${params.percentile}` : ""}${params?.require_warm_nights ? `&require_warm_nights=true` : ""}`),
};

export { ApiError };
