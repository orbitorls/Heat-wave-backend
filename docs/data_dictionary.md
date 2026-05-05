# Data Dictionary

## ข้อมูลอุตุนิยมวิทยา (Meteorological)

| Field | Type | Unit | Source | Description |
|-------|------|------|--------|-------------|
| `temperature` | float | °C | TMD API | อุณหภูมิอากาศ (dry-bulb) |
| `humidity` | float | % | TMD API | ความชื้นสัมพัทธ์ |
| `heat_index` | float | °C | calculated | ดัชนีความร้อน (Rothfusz) |
| `observed_at` | datetime | UTC+7 | TMD | เวลาที่วัด |
| `station_id` | str | — | TMD | รหัสสถานีอุตุฯ |
| `lat` / `lon` | float | degrees | TMD/OSM | พิกัด |
| `forecast_hour` | int | hours | TMD | ชั่วโมงที่คาดการณ์ล่วงหน้า (0–72) |

## ข้อมูลกิจกรรม (Activity Context)

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `activity_type` | str | — | ประเภทกิจกรรม เช่น `assembly`, `pe_class`, `construction`, `farming` |
| `activity_intensity` | str | — | `low` / `moderate` / `high` / `very_high` |
| `start_time` | time | HH:MM | เวลาเริ่มกิจกรรม |
| `duration_minutes` | int | min | ระยะเวลากิจกรรม |
| `location_type` | str | — | `outdoor` / `semi-outdoor` / `indoor` |
| `shade_available` | bool | — | มีร่มเงาหรือไม่ |
| `water_access` | bool | — | มีจุดดื่มน้ำหรือไม่ |

## โปรไฟล์กลุ่มเสี่ยง (Vulnerability Profile)

| Field | Type | Description |
|-------|------|-------------|
| `profile_id` | str | รหัสโปรไฟล์ เช่น `student_primary`, `outdoor_worker` |
| `age_group` | str | `child` / `youth` / `adult` / `elderly` |
| `base_vulnerability` | float | 0.0–1.0 น้ำหนักความเปราะบางพื้นฐาน |
| `acclimatization` | bool | ร่างกายชินความร้อนแล้วหรือไม่ |
| `medical_condition` | list[str] | โรคประจำตัว เช่น `cardiovascular`, `diabetes` |

## Risk Score Output

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `risk_score` | float | 0–100 | คะแนนความเสี่ยงรวม |
| `risk_class` | str | — | `Low` / `Moderate` / `High` / `Critical` |
| `heat_index_cat` | str | — | หมวดดัชนีความร้อน (NOAA standard) |
| `dominant_factors` | list[str] | — | ปัจจัยหลักที่ทำให้ risk สูง (explainability) |
| `confidence` | float | 0–1 | ความมั่นใจของการประเมิน |

## Heatwave Event

| Field | Type | Description |
|-------|------|-------------|
| `is_heatwave` | bool | เป็น heatwave event หรือไม่ |
| `event_type` | str | `local_percentile` / `absolute` |
| `percentile_threshold` | float | percentile ที่ใช้ (เช่น 90.0) |
| `consecutive_days` | int | จำนวนวันต่อเนื่องที่เกิน threshold |
| `local_baseline` | dict | ค่าอ้างอิงพื้นที่นั้น (mean, p90, p95) |
