# HeatShield AI — Backend

> Adaptive Heat-Health Risk Intelligence for School Safety and Outdoor Workers
>
> *"We do not forecast the weather. We convert heat into safer decisions."*

HeatShield AI เป็นระบบ AI ที่แปลง `อุณหภูมิ + ความชื้น + พื้นที่ + กิจกรรม` ให้กลายเป็น **คะแนนความเสี่ยงสุขภาพ** และ **คำแนะนำเชิงปฏิบัติ** สำหรับโรงเรียนและคนทำงานกลางแจ้ง — ไม่ยึดนิยาม heatwave แบบตัวเลขเดียว เพราะแต่ละพื้นที่มีภูมิอากาศและความเปราะบางไม่เหมือนกัน

ที่มาแนวคิด: ดูเอกสาร [`HeatShield_AI_Proposal_TH.pdf`](./HeatShield_AI_Proposal_TH.pdf)

---

## ระบบ 3 ชั้น

| ชั้น | คำถามที่ตอบ | วิธีคิด |
|------|-------------|---------|
| 1) Heatwave Event Detection | ช่วงนี้เป็นเหตุการณ์ร้อนผิดปกติเมื่อเทียบพื้นที่นั้นหรือไม่? | local percentile (90/95th) + consecutive days/nights |
| 2) Heat-Health Risk Scoring | วันนี้เสี่ยงต่อสุขภาพระดับไหนสำหรับกลุ่มนี้? | heat index + humidity + time of day + exposure duration + activity intensity + vulnerability |
| 3) Decision Support | ควรทำอะไรต่อ? | action card: เลื่อน/งด/พัก/แจ้งครู-หัวหน้างาน |

## สถาปัตยกรรม 5 Layers

```
Data Layer       → TMD API, OpenStreetMap, ตารางกิจกรรม
Model Layer      → XGBoost/LightGBM/Prophet/LSTM + rule-based safety threshold
Risk Layer       → Scoring engine + local percentile + vulnerability weight
Decision Layer   → Rule engine + templates + LLM summarization (text only)
Product Layer    → React/Flutter dashboard + FastAPI + PostgreSQL/PostGIS
```

## โครงสร้างโปรเจค

```
app/
├── core/                  ← domain logic ที่ไม่ขึ้นกับ ML
│   ├── heat_index.py      ← Rothfusz/Steadman formula
│   ├── adaptive_definition.py  ← local percentile + consecutive days
│   ├── vulnerability.py   ← profile catalog (student, worker, ...)
│   ├── risk_scoring.py    ← Heat-Health Risk Score
│   ├── whatif.py          ← What-if simulator
│   └── action_card.py     ← Action card generator
├── models/                ← ML models (forecasting, classification)
├── data/                  ← TMD client, schemas, loaders
└── api/                   ← FastAPI routes
docs/                      ← data dictionary, requirements, architecture
tests/                     ← pytest
scripts/                   ← demo data seeders
```

## การติดตั้ง (development)

```bash
python -m venv .venv
.venv\Scripts\activate          # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# รัน API
uvicorn app.main:app --reload

# รันเทส
pytest

# สร้างข้อมูล demo
python scripts/seed_demo_data.py
```

## Endpoints หลัก

| Method | Path | หน้าที่ |
|--------|------|---------|
| POST | `/heat-index` | คำนวณ heat index จาก temp/humidity |
| POST | `/events/detect` | ตรวจจับ heatwave event ด้วย local percentile |
| POST | `/risk/score` | คำนวณ Heat-Health Risk Score |
| POST | `/whatif/simulate` | จำลองผลของการเลื่อนเวลา/เพิ่มพักน้ำ/ย้ายสถานที่ |
| POST | `/action-card` | สร้างคำแนะนำเชิงปฏิบัติ |

## Impact Metrics (ตาม proposal §9)

- **Forecast accuracy** — MAE/RMSE เทียบ baseline
- **Alert lead time** — แจ้งล่วงหน้า 24–72 ชม.
- **High-risk exposure reduction** — ลด 30–60% ใน scenario
- **Decision improvement** — คะแนน decision quality
- **Action card usability** — ≥ 4/5 จากกลุ่มทดลอง
- **False reassurance control** — uncertainty สูง → conservative

## License & Reference

อ้างอิงตามเอกสารใน proposal §19: WHO, WMO, กรมควบคุมโรค, TMD, UNDRR
