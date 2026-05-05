# HeatShield AI — Colab Notebooks

> **EN.** Training-only Colab notebooks for HeatShield AI. Inference (FastAPI) stays local; only ingest + training run in Colab so we can use longer sessions and free GPU when needed (TabPFN backend).
>
> **TH.** Notebook สำหรับเทรนโมเดล HeatShield AI บน Google Colab. ส่วน inference (FastAPI) ยังคงรันบนเครื่อง local — Colab ใช้เฉพาะตอน ingest ข้อมูลและเทรน เพื่อให้ใช้ session ยาวขึ้นและใช้ GPU ฟรีได้เมื่อจำเป็น (TabPFN backend).

## Notebooks

> Colab badge URLs use `orbitorls/HeatShield` as the GitHub coordinates.

| # | Notebook | EN purpose | TH หน้าที่ | Open in Colab |
|---|----------|------------|------------|----------------|
| 00 | `00_setup.ipynb` | Bootstrap: Drive mount, repo clone, pip install, sanity checks. | ตั้งค่าเริ่มต้น: mount Drive, clone repo, ติดตั้ง deps, ตรวจสอบความพร้อม. | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/orbitorls/HeatShield/blob/main/notebooks/colab/00_setup.ipynb) |
| 01 | `01_ingest.ipynb` | Ingest NASA POWER (5y) + ERA5 (3y) + optional TMD into the Drive parquet store. | ดึงข้อมูล NASA POWER (5 ปี) + ERA5 (3 ปี) + TMD (ถ้ามี key) ลง parquet บน Drive. | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/orbitorls/HeatShield/blob/main/notebooks/colab/01_ingest.ipynb) |

> Future notebooks (`02_train_lightgbm.ipynb`, `03_train_xgboost.ipynb`, `04_train_tabpfn.ipynb`, `05_evaluate.ipynb`) will follow the same Drive-backed layout.

## Quick start / เริ่มใช้งานอย่างเร็ว

**EN.**

1. In Colab, add three secrets via the key icon: `GH_PAT`, `CDSAPI_KEY`, `TMD_API_KEY` (only `GH_PAT` is mandatory; the others gate ERA5/TMD ingest).
2. Open `00_setup.ipynb` and run all cells. Verify every check prints `[OK]`.
3. Open `01_ingest.ipynb` and run all cells. The first ingest takes a while; subsequent runs are incremental.
4. Train with one of the `02_train_*.ipynb` notebooks. Artifacts land in `app/models/forecast_v3/{station}/h{H}/` on Drive.
5. Push the trained artifacts back into the local repo following the workflow in `docs/colab-training.md`.

**TH.**

1. ใน Colab ใส่ secrets ผ่านไอคอนรูปกุญแจ 3 ตัว: `GH_PAT`, `CDSAPI_KEY`, `TMD_API_KEY` (จำเป็นเฉพาะ `GH_PAT`; อีกสองตัวเอาไว้ใช้กับ ERA5/TMD).
2. เปิด `00_setup.ipynb` และรันทุก cell — ตรวจให้ทุกข้อขึ้น `[OK]`.
3. เปิด `01_ingest.ipynb` และรันทุก cell. ครั้งแรกใช้เวลานาน รอบถัดไปจะเป็น incremental.
4. เลือก `02_train_*.ipynb` เพื่อเทรน. Artifacts จะอยู่ที่ `app/models/forecast_v3/{station}/h{H}/` บน Drive.
5. push artifacts กลับเข้า repo ตามขั้นตอนใน `docs/colab-training.md`.

## File layout / โครงสร้างไฟล์

```text
/content/drive/MyDrive/heatshield/   # persistent across sessions
├── data/                            # symlinked into repo as data/
│   └── raw/station_id={id}/date={iso}/obs.parquet
├── models/forecast_v3/              # symlinked into repo as app/models/forecast_v3/
│   └── {station_id}/h{H}/...
└── logs/

/content/Heat-wave-backend/          # cloned repo (ephemeral)
├── data/  -> /content/drive/MyDrive/heatshield/data
├── app/models/forecast_v3/  -> /content/drive/MyDrive/heatshield/models/forecast_v3
└── ...
```

See `docs/colab-training.md` for full architecture, recovery procedure, and the Drive → repo push-back workflow.
