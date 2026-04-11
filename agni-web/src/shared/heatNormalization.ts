export type HeatLevel = 'normal' | 'caution' | 'warning' | 'danger' | 'severe';

export interface RiskDescriptor {
  code: string;
  heatLevel: HeatLevel;
  labelTh: string;
  isUnknown: boolean;
}

export interface NormalizedForecastDay {
  day: number;
  date: string;
  dayName: string;
  temperature: number;
  risk: RiskDescriptor;
  probability: number | null;
}

export const HEAT_LABEL_TH: Record<HeatLevel, string> = {
  normal: 'ปกติ',
  caution: 'เฝ้าระวัง',
  warning: 'เตือน',
  danger: 'อันตราย',
  severe: 'วิกฤต',
};

export const HEAT_HEX: Record<HeatLevel, string> = {
  normal: '#b5923a',
  caution: '#d4a017',
  warning: '#c46000',
  danger: '#b02020',
  severe: '#7a1530',
};

export const HEAT_ADVICE_TEXT: Record<HeatLevel, string> = {
  normal: 'อุณหภูมิอยู่ในเกณฑ์ปกติ สามารถทำกิจกรรมกลางแจ้งได้ตามปกติ',
  caution:
    'ดื่มน้ำให้เพียงพอและสวมเสื้อผ้าที่ระบายอากาศได้ดี หลีกเลี่ยงการทำกิจกรรมกลางแจ้งเป็นเวลานาน',
  warning: 'หลีกเลี่ยงการสัมผัสแสงแดดโดยตรงเป็นเวลานาน ควรอยู่ในที่ร่มและดื่มน้ำมากๆ',
  danger: 'มีความเสี่ยงสูงต่อภาวะเพลียแดด ควรอยู่ในอาคารที่มีเครื่องปรับอากาศ',
  severe: 'อันตรายจากคลื่นความร้อนรุนแรง เสี่ยงต่อโรคลมแดด ควรงดออกนอกอาคารและดื่มน้ำอย่างสม่ำเสมอ',
};

export const ZONE_THAI: Record<string, string> = {
  Central: 'ภาคกลาง',
  North: 'ภาคเหนือ',
  Northeast: 'ภาคตะวันออกเฉียงเหนือ',
  South: 'ภาคใต้',
  East: 'ภาคตะวันออก',
  West: 'ภาคตะวันตก',
};

const ENGLISH_DAY_TO_THAI: Record<string, string> = {
  MON: 'วันจันทร์',
  MONDAY: 'วันจันทร์',
  TUE: 'วันอังคาร',
  TUESDAY: 'วันอังคาร',
  WED: 'วันพุธ',
  WEDNESDAY: 'วันพุธ',
  THU: 'วันพฤหัสบดี',
  THURSDAY: 'วันพฤหัสบดี',
  FRI: 'วันศุกร์',
  FRIDAY: 'วันศุกร์',
  SAT: 'วันเสาร์',
  SATURDAY: 'วันเสาร์',
  SUN: 'วันอาทิตย์',
  SUNDAY: 'วันอาทิตย์',
};

const API_RISK_TO_LEVEL: Record<string, HeatLevel> = {
  LOW: 'normal',
  NORMAL: 'normal',
  MEDIUM: 'caution',
  ELEVATED: 'caution',
  CAUTION: 'caution',
  HIGH: 'warning',
  WARNING: 'warning',
  'VERY HIGH': 'danger',
  DANGER: 'danger',
  CRITICAL: 'severe',
  SEVERE: 'severe',
};

const clamp = (value: number, min: number, max: number): number => {
  if (value < min) return min;
  if (value > max) return max;
  return value;
};

const readNumber = (value: unknown): number | null => {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return null;
};

export const deriveHeatLevelFromTemperature = (temperature: number): HeatLevel => {
  if (temperature >= 42) return 'severe';
  if (temperature >= 39) return 'danger';
  if (temperature >= 36) return 'warning';
  if (temperature >= 33) return 'caution';
  return 'normal';
};

export const extractTemperatureFromPayload = (payload: unknown): number | null => {
  if (!payload || typeof payload !== 'object') return null;
  const obj = payload as Record<string, unknown>;
  const weather =
    obj.weather && typeof obj.weather === 'object'
      ? (obj.weather as Record<string, unknown>)
      : null;

  return (
    readNumber(weather?.T2M_MAX) ??
    readNumber(obj.temperature) ??
    readNumber(obj.max_temperature) ??
    readNumber(obj.temp)
  );
};

export const normalizeProbability = (value: unknown): number | null => {
  const parsed = readNumber(value);
  if (parsed === null) return null;

  const normalized = parsed > 1 ? parsed / 100 : parsed;
  return clamp(normalized, 0, 1);
};

export const normalizeRisk = (rawRisk: unknown, _temperature?: number | null): RiskDescriptor => {
  const normalizedCode =
    typeof rawRisk === 'string' && rawRisk.trim().length > 0
      ? rawRisk.trim().toUpperCase()
      : 'UNKNOWN';

  const mappedLevel = API_RISK_TO_LEVEL[normalizedCode];
  if (mappedLevel) {
    return {
      code: normalizedCode,
      heatLevel: mappedLevel,
      labelTh: HEAT_LABEL_TH[mappedLevel],
      isUnknown: false,
    };
  }

  return {
    code: normalizedCode,
    heatLevel: 'caution',
    labelTh: 'ไม่ทราบ',
    isUnknown: true,
  };
};

export const normalizeForecastEntry = (entry: unknown, fallbackDay: number): NormalizedForecastDay | null => {
  if (!entry || typeof entry !== 'object') return null;
  const row = entry as Record<string, unknown>;
  const temperature = extractTemperatureFromPayload(row);
  if (temperature === null) return null;

  const dayNumber = readNumber(row.day);
  const date = typeof row.date === 'string' ? row.date : '';
  const dayName = typeof row.day_name === 'string' ? row.day_name : '';
  const risk = normalizeRisk(row.risk_level, temperature);
  const probability = normalizeProbability(row.probability);

  return {
    day: dayNumber === null ? fallbackDay : Math.max(1, Math.trunc(dayNumber)),
    date,
    dayName,
    temperature,
    risk,
    probability,
  };
};

export const formatThaiDate = (dateText: string): string => {
  if (!dateText) return '';
  const date = new Date(dateText);
  if (Number.isNaN(date.getTime())) return dateText;
  return new Intl.DateTimeFormat('th-TH', { day: 'numeric', month: 'short' }).format(date);
};

export const formatThaiDayName = (dayName: string, dateText: string): string => {
  const normalizedDayName = dayName.trim();
  if (normalizedDayName.startsWith('วัน')) return normalizedDayName;

  if (normalizedDayName) {
    const translated = ENGLISH_DAY_TO_THAI[normalizedDayName.toUpperCase()];
    if (translated) return translated;
    return `วัน${normalizedDayName}`;
  }

  if (dateText) {
    const date = new Date(dateText);
    if (!Number.isNaN(date.getTime())) {
      return new Intl.DateTimeFormat('th-TH', { weekday: 'long' }).format(date);
    }
  }

  return '';
};

export const formatTemperature = (value: number, fractionDigits = 1): string => `${value.toFixed(fractionDigits)}°C`;

export const formatProbabilityPercent = (probability: number | null): string => {
  if (probability === null) return '—';
  return `${Math.round(clamp(probability, 0, 1) * 100)}%`;
};
