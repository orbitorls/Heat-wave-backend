# ระบบพยากรณ์คลื่นความร้อนเชิงพื้นที่ (Thailand Heatwave Forecasting) 🌡️🇹🇭

ระบบนี้เป็น Backend สำหรับพยากรณ์อุณหภูมิและคลื่นความร้อนในประเทศไทย โดยใช้โมเดล Deep Learning ชนิด **ConvLSTM** ร่วมกับข้อมูลอุตุนิยมวิทยา **ERA5** 

## ✨ จุดเด่นของระบบ
- **Spatial Awareness**: โมเดลรับรู้ความแตกต่างของพื้นที่ (เช่น ภาคเหนือที่เป็นภูเขา vs ภาคกลางที่เป็นที่ราบ) ผ่านการป้อนข้อมูลพิกัด (Lat/Lon) และระดับความสูง (Elevation) เป็นช่องสัญญาณเสริม
- **Physics-Informed**: มีการคำนวณ Loss Function โดยอิงหลักฟิสิกส์บรรยากาศ (Adiabatic process) เพื่อให้ผลการพยากรณ์มีความสมจริงตามหลักอุตุนิยมวิทยา
- **Multi-Channel Data**: ใช้ข้อมูลจาก ERA5 ทั้ง Geopotential Height ($Z_{500}$), อุณหภูมิที่ 2 เมตร ($T_{2m}$), และความชื้นในดิน (Soil Moisture)

## 📁 โครงสร้างโปรเจกต์

```
├── src/                        # Package หลัก
│   ├── api/                    # FastAPI server (เวอร์ชันใหม่)
│   ├── cli/                    # Command-line interface
│   ├── core/                   # Config, logger, utils
│   ├── data/loader.py          # ✅ ERA5 data loading & preprocessing
│   └── models/
│       ├── convlstm.py         # ✅ HeatwaveConvLSTM + PhysicsInformedLoss
│       └── manager.py          # ✅ ModelManager (load/predict)
├── tests/                      # 🧪 Test suite (pytest)
│   ├── conftest.py             # Shared fixtures
│   ├── test_model.py           # Unit tests: ConvLSTM, PhysicsInformedLoss
│   ├── test_data_loader.py     # Unit tests: data utilities
│   └── test_model_manager.py   # Unit tests: ModelManager
├── config/config.yaml          # Configuration
├── era5_data/                  # ERA5 NetCDF inputs
├── models/                     # Model checkpoints (.pth)
├── output/                     # Generated artifacts
├── logs/                       # Log files
├── api_server.py               # Flask API entrypoint (Production)
├── Train_Ai.py                 # Training script
├── evaluate_model.py           # 🔍 Model evaluation CLI
├── heatwave_cli.py             # Interactive CLI
└── download_era5.py            # ERA5 downloader
```

## 🚀 วิธีการเริ่มต้นใช้งาน

### 1. ติดตั้ง Library ที่จำเป็น
```bash
pip install -r requirements.txt
```

### 2. เตรียมข้อมูล
ดาวน์โหลดข้อมูล ERA5 มาไว้ที่โฟลเดอร์ `era5_data/` หรือใช้สคริปต์:
```bash
python download_era5.py
```

### 3. การเทรนโมเดล (Training)
```bash
python Train_Ai.py
```
*ระบบจะสร้างไฟล์โมเดลไว้ใน `models/heatwave_model_checkpoint_v{N}.pth`*

### 4. รันเซิร์ฟเวอร์ API
```bash
python api_server.py
```

### 5. 🔍 ทดสอบ/ประเมินผลโมเดล (Model Evaluation)

**รัน unit tests ทั้งหมด:**
```bash
pytest
```

**ประเมินโมเดลบน test set พร้อมดู metrics:**
```bash
# ใช้ checkpoint ล่าสุดอัตโนมัติ
python evaluate_model.py

# ระบุ checkpoint เอง
python evaluate_model.py --checkpoint models/heatwave_convlstm_v3.pth

# บันทึกผลลัพธ์เป็น JSON
python evaluate_model.py --output-json output/eval_results.json

# ดู progress ระหว่าง evaluate
python evaluate_model.py --verbose
```

**Metrics ที่แสดง:**
| Metric | คำอธิบาย |
|--------|----------|
| MAE / RMSE (°C) | ความผิดพลาดของการพยากรณ์อุณหภูมิ |
| R² | ความแม่นยำโดยรวมของโมเดล |
| Heatwave Precision/Recall/F1 | ความสามารถตรวจจับคลื่นความร้อน (≥35°C) |
| MSE Loss / Physics Loss | ค่า Loss จาก training objective |
| Inference time | ความเร็วในการพยากรณ์ (ms/sample) |

## 📡 API Endpoints ที่สำคัญ
- `POST /api/predict`: พยากรณ์อุณหภูมิล่วงหน้าจากข้อมูลล่าสุด
- `GET /api/map`: ดึงข้อมูลพยากรณ์ในรูปแบบ GeoJSON สำหรับแสดงผลบนแผนที่
- `GET /api/health`: ตรวจสอบสถานะของระบบและเวอร์ชันโมเดลที่ใช้งาน

## 🛠️ รายละเอียดทางเทคนิค (สำหรับนักพัฒนา)
เพื่อให้โมเดลแยกแยะพื้นที่ได้ ระบบมีการทำ **Coordinate Encoding**:
- **Channel 0-2**: ข้อมูลอากาศ ($Z, T_{2m}, SWVL1$)
- **Channel 3**: ระดับความสูง (Elevation) เพื่อแยกแยะพื้นที่ภูเขา
- **Channel 4-5**: พิกัดรุ้งและแวง (Latitude, Longitude) เพื่อระบุตำแหน่งเชิงภูมิศาสตร์

---
พัฒนาโดยใช้ Python, PyTorch และ Xarray
