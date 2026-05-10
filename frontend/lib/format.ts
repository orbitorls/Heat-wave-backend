// Formatting utilities for HeatShield AI frontend

/**
 * Format a temperature in Celsius with 1 decimal place.
 */
export function formatTemp(c: number): string {
  return `${c.toFixed(1)} °C`;
}

/**
 * Format heat index as a rounded integer with unit.
 */
export function formatHI(c: number): string {
  return `${Math.round(c)}°C`;
}

/**
 * Format relative humidity.
 */
export function formatRH(rh: number): string {
  return `${Math.round(rh)}%`;
}

/**
 * Format a number as a risk score (0–100).
 */
export function formatScore(score: number): string {
  return Math.round(score).toString();
}

/**
 * Thai date/time string from an ISO UTC datetime string.
 */
export function toThaiDate(iso: string): string {
  try {
    const d = new Date(iso);
    return d.toLocaleDateString("th-TH", {
      day: "numeric",
      month: "short",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return iso;
  }
}

/**
 * Short time string (HH:MM) in local time.
 */
export function toTime(iso: string): string {
  try {
    const d = new Date(iso);
    return d.toLocaleTimeString("th-TH", { hour: "2-digit", minute: "2-digit" });
  } catch {
    return "";
  }
}

/**
 * Format a horizon in hours to a Thai label.
 */
export function formatHorizon(h: number): string {
  if (h < 24) return `${h} ชั่วโมง`;
  const days = h / 24;
  return days === 1 ? "1 วัน" : `${days} วัน`;
}

/**
 * Elevation with unit.
 */
export function formatElevation(m: number): string {
  return `${m.toFixed(0)} ม.`;
}
