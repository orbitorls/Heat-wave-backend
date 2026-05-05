# System Architecture

## Pipeline Overview

```
[TMD API / OpenStreetMap / Activity Table]
          |
          v
    Data Layer (app/data/)
    - tmd_client.py       : ดึงอุณหภูมิ/ความชื้น/พยากรณ์
    - schemas.py          : Pydantic models
    - loaders.py          : data normalization
          |
          v
    Model Layer (app/models/)
    - forecast.py         : XGBoost/LightGBM/Prophet (24-72h heat index)
    - classifier.py       : ML risk classifier
          |
          v
    Core / Risk Layer (app/core/)
    - heat_index.py            : Rothfusz formula
    - adaptive_definition.py   : local percentile + consecutive days
    - vulnerability.py         : profile catalog
    - risk_scoring.py          : Heat-Health Risk Score
    - whatif.py                : What-if Simulator
    - action_card.py           : Action Card Generator
          |
          v
    API Layer (app/api/)       : FastAPI routes
          |
          v
    [Dashboard / Mobile App / B2B API consumers]
```

## Key Design Decisions

### 1. Adaptive Definition (ไม่ยึด heatwave นิยามเดียว)
- ใช้ **local percentile** (90th / 95th) ของข้อมูลย้อนหลัง 30 ปี (หรือที่มี) ต่อพื้นที่
- ต้องเกิน threshold **ต่อเนื่อง 2–3 วันกลางคืน** ตาม WMO/UNDRR
- แยก `event_definition` ออกจาก `health_risk_scoring` โดยเด็ดขาด

### 2. Profile-based Risk (ไม่ใช่อุณหภูมิเดียวกันสำหรับทุกคน)
- นักเรียนประถม vs. แรงงานกลางแจ้ง vs. ผู้สูงอายุ → vulnerability weight ต่างกัน
- Activity intensity × duration × shade → exposure factor

### 3. Conservative by Design (safety-critical)
- ถ้า confidence ต่ำ → ส่ง conservative alert
- ไม่ claim precision ที่ไม่มีข้อมูลรองรับ (hyperlocal)
- แสดง dominant_factors เสมอ (Explainable AI)

### 4. What-if ≠ simulation แบบ black box
- ทุก what-if scenario recalculate ผ่าน rule engine เดิม
- ผู้ใช้เห็นว่า risk ลดจากอะไร (เลื่อนเวลา vs. เพิ่มพักน้ำ vs. ย้ายสถานที่)

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI + Uvicorn |
| Validation | Pydantic v2 |
| ML | XGBoost, LightGBM, scikit-learn |
| Data | pandas, numpy |
| DB (future) | PostgreSQL + PostGIS |
| Frontend (future) | React / Flutter |
