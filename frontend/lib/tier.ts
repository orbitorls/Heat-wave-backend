import type { RiskClass } from "./api-types";

export interface TierConfig {
  label: string;
  labelEn: RiskClass;
  // OKLCH color tokens (same as CSS variables but as values for inline styles)
  color: string;         // border / icon
  bg: string;            // card background
  text: string;          // text on bg
  barColor: string;      // progress bar fill
}

export const TIER_CONFIG: Record<RiskClass, TierConfig> = {
  Low: {
    label: "ต่ำ",
    labelEn: "Low",
    color: "oklch(0.78 0.10 195)",
    bg: "oklch(0.95 0.025 195)",
    text: "oklch(0.35 0.10 195)",
    barColor: "oklch(0.78 0.10 195)",
  },
  Moderate: {
    label: "ปานกลาง",
    labelEn: "Moderate",
    color: "oklch(0.72 0.15 80)",
    bg: "oklch(0.96 0.030 80)",
    text: "oklch(0.38 0.15 80)",
    barColor: "oklch(0.72 0.15 80)",
  },
  High: {
    label: "สูง",
    labelEn: "High",
    color: "oklch(0.62 0.20 35)",
    bg: "oklch(0.95 0.030 35)",
    text: "oklch(0.35 0.18 35)",
    barColor: "oklch(0.62 0.20 35)",
  },
  Critical: {
    label: "วิกฤต",
    labelEn: "Critical",
    color: "oklch(0.52 0.22 22)",
    bg: "oklch(0.94 0.030 22)",
    text: "oklch(0.30 0.20 22)",
    barColor: "oklch(0.52 0.22 22)",
  },
};

export function getTier(riskClass: RiskClass): TierConfig {
  return TIER_CONFIG[riskClass] ?? TIER_CONFIG.Moderate;
}

// Derive risk class from heat index value (NOAA thresholds adapted for Thai climate)
export function hiToRiskClass(hiC: number): RiskClass {
  if (hiC >= 41) return "Critical";
  if (hiC >= 32) return "High";
  if (hiC >= 27) return "Moderate";
  return "Low";
}
