"""Helper to (re)generate the colab notebooks as valid nbformat-4 JSON.

ใช้ภายในตอนตั้งค่าครั้งแรกเท่านั้น — ไม่ได้เรียกใน runtime.
Internal one-shot generator; not used at runtime.

Run:
    python notebooks/colab/_build_nbs.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent


def code_cell(source: list[str]) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def md_cell(source: list[str]) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_nb(name: str, nb: dict) -> None:
    path = HERE / name
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path}")


# --- 00_setup.ipynb ---------------------------------------------------------

setup_nb = notebook(
    [
        md_cell(
            [
                "# 00 — Colab Setup / ตั้งค่า Colab\n",
                "\n",
                "**EN.** First-cell bootstrap that mounts Google Drive, clones the HeatShield repo using `GH_PAT` from Colab Secrets, symlinks `data/` and `app/models/forecast_v3/` to Drive, and installs `requirements-train.txt`. Idempotent — safe to re-run.\n",
                "\n",
                "**TH.** เซลล์ตั้งต้นของทุก notebook: mount Drive, clone repo (ใช้ `GH_PAT` จาก Colab Secrets), ทำ symlink ของ `data/` และ `app/models/forecast_v3/` ไปที่ Drive, และติดตั้ง `requirements-train.txt`. รันซ้ำได้โดยปลอดภัย.\n",
                "\n",
                "Required Colab userdata secrets / secrets ที่ต้องตั้ง:\n",
                "\n",
                "- `GH_PAT` — GitHub fine-grained personal access token (scope: read repo)\n",
                "- `CDSAPI_KEY` — สำหรับ ERA5 ingest (optional ใน notebook นี้)\n",
                "- `TMD_API_KEY` — สำหรับ TMD ingest (optional ใน notebook นี้)\n",
            ]
        ),
        code_cell(
            [
                "# --- bootstrap (clone + drive + pip) ---------------------------------------\n",
                "import os, urllib.request\n",
                "from google.colab import userdata\n",
                "\n",
                "# Read secrets (fail-soft so the cell still runs in dev mode).\n",
                "for key in (\"GH_PAT\", \"CDSAPI_KEY\", \"TMD_API_KEY\"):\n",
                "    try:\n",
                "        os.environ[key] = userdata.get(key)\n",
                "    except Exception as exc:\n",
                "        print(f\"[setup] secret {key} not set: {exc}\")\n",
                "\n",
                "# GitHub coordinates: orbitorls/HeatShield\n",
                "os.environ.setdefault(\"GH_OWNER\", \"orbitorls\")\n",
                "os.environ.setdefault(\"GH_REPO\", \"HeatShield\")\n",
                "os.environ.setdefault(\"GH_BRANCH\", \"main\")\n",
                "\n",
                "REPO_DIR = \"/content/Heat-wave-backend\"\n",
                "BOOTSTRAP = f\"{REPO_DIR}/scripts/colab_bootstrap.sh\"\n",
                "\n",
                "# Pull bootstrap script before the repo is cloned.\n",
                "if not os.path.exists(BOOTSTRAP):\n",
                "    raw_url = (\n",
                "        f\"https://raw.githubusercontent.com/{os.environ['GH_OWNER']}/\"\n",
                "        f\"{os.environ['GH_REPO']}/{os.environ['GH_BRANCH']}/scripts/colab_bootstrap.sh\"\n",
                "    )\n",
                "    tmp = \"/tmp/colab_bootstrap.sh\"\n",
                "    try:\n",
                "        urllib.request.urlretrieve(raw_url, tmp)\n",
                "        BOOTSTRAP = tmp\n",
                "    except Exception as exc:\n",
                "        print(f\"[setup] could not pre-fetch bootstrap ({exc}); will rely on the repo copy.\")\n",
                "\n",
                "!bash {BOOTSTRAP}\n",
            ]
        ),
        code_cell(
            [
                "# --- import test + GPU check ----------------------------------------------\n",
                "import sys, importlib\n",
                "\n",
                "for mod in (\"numpy\", \"pandas\", \"xgboost\", \"lightgbm\", \"sklearn\", \"optuna\"):\n",
                "    try:\n",
                "        m = importlib.import_module(mod)\n",
                "        print(f\"{mod:<10s} {getattr(m, '__version__', '?')}\")\n",
                "    except ImportError as exc:\n",
                "        print(f\"{mod:<10s} MISSING ({exc})\")\n",
                "\n",
                "# torch is optional — only relevant for TabPFN backend.\n",
                "try:\n",
                "    import torch\n",
                "    print(f\"torch      {torch.__version__} | cuda available: {torch.cuda.is_available()}\")\n",
                "    if torch.cuda.is_available():\n",
                "        print(f\"           device: {torch.cuda.get_device_name(0)}\")\n",
                "except Exception as exc:\n",
                "    print(f\"torch      not installed ({exc.__class__.__name__}); GPU check skipped.\")\n",
                "\n",
                "print(f\"python     {sys.version.split()[0]}\")\n",
            ]
        ),
        code_cell(
            [
                "# --- verify Drive mount + repo + station registry -------------------------\n",
                "import os, sys\n",
                "\n",
                "REPO_DIR = \"/content/Heat-wave-backend\"\n",
                "DRIVE_ROOT = \"/content/drive/MyDrive/heatshield\"\n",
                "\n",
                "checks = {\n",
                "    \"Drive mounted\":      os.path.isdir(\"/content/drive/MyDrive\"),\n",
                "    \"Drive heatshield/\":  os.path.isdir(DRIVE_ROOT),\n",
                "    \"Repo cloned\":        os.path.isdir(REPO_DIR),\n",
                "    \"data/ symlink\":      os.path.islink(f\"{REPO_DIR}/data\"),\n",
                "    \"models/forecast_v3 symlink\": os.path.islink(f\"{REPO_DIR}/app/models/forecast_v3\"),\n",
                "}\n",
                "for label, ok in checks.items():\n",
                "    print(f\"  [{'OK' if ok else '--'}] {label}\")\n",
                "\n",
                "# Make repo importable for subsequent cells.\n",
                "if REPO_DIR not in sys.path:\n",
                "    sys.path.insert(0, REPO_DIR)\n",
                "\n",
                "from app.data.stations import STATIONS\n",
                "print(f\"\\nStations registered: {list(STATIONS.keys())}\")\n",
                "assert set(STATIONS.keys()) == {\"BKK_01\", \"CNX_01\", \"KKN_01\", \"HYI_01\", \"RYG_01\"}, \\\n",
                "    \"Station registry mismatch — see app/data/stations.py\"\n",
                "print(\"Station registry OK.\")\n",
            ]
        ),
        md_cell(
            [
                "## Summary / สรุป\n",
                "\n",
                "**EN.** If every check above shows `[OK]`, the Colab environment is ready. Drive holds the persistent `data/` and `app/models/forecast_v3/` directories so subsequent notebooks can ingest, train, and push artifacts without re-downloading. Open `01_ingest.ipynb` next.\n",
                "\n",
                "**TH.** ถ้าตัวตรวจทุกข้อขึ้น `[OK]` แสดงว่าสภาพแวดล้อม Colab พร้อมใช้งาน. Drive จะเก็บ `data/` และ `app/models/forecast_v3/` แบบถาวรเพื่อให้ notebook ถัดไป ingest / train / push artifacts ได้โดยไม่ต้องโหลดข้อมูลใหม่. ขั้นถัดไปคือเปิด `01_ingest.ipynb`.\n",
                "\n",
                "**Troubleshooting / แก้ปัญหาเบื้องต้น**\n",
                "\n",
                "- `GH_PAT` missing → ใส่ใน Colab → key icon (Secrets).\n",
                "- Drive ขอ permission ใหม่ทุกครั้ง → กด `Authorize` ซ้ำ; bootstrap คือ idempotent.\n",
                "- Symlink ไม่ขึ้น → ลบ `data/`, `app/models/forecast_v3/` ในกรณี local เก่าค้าง แล้วรัน Cell 1 ซ้ำ.\n",
            ]
        ),
    ]
)
write_nb("00_setup.ipynb", setup_nb)


# --- 01_ingest.ipynb --------------------------------------------------------

ingest_nb = notebook(
    [
        md_cell(
            [
                "# 01 — Data Ingestion / ดึงข้อมูล\n",
                "\n",
                "**EN.** Wraps `scripts/ingest_all.py` in Colab. Pulls a multi-year window into the Drive-backed parquet store at `data/raw/station_id={id}/date={iso}/obs.parquet`. Idempotent — partitions that already exist are skipped unless `--force`.\n",
                "\n",
                "**TH.** ห่อ `scripts/ingest_all.py` สำหรับ Colab. ดึงข้อมูลย้อนหลังหลายปีลง parquet store บน Drive ที่ `data/raw/station_id={id}/date={iso}/obs.parquet`. รันซ้ำได้ — partition ที่มีอยู่จะถูกข้ามเว้นแต่ใช้ `--force`.\n",
                "\n",
                "**Recommended data window / ช่วงข้อมูลที่แนะนำ**\n",
                "\n",
                "- NASA POWER (no auth) — last **5 years** for solar + cloud + sfc temp.\n",
                "- ERA5 via CDS (`CDSAPI_KEY`) — last **3 years** at 0.25° resolution.\n",
                "- TMD (`TMD_API_KEY`) — current Thai station data, last **1–2 years** depending on availability.\n",
                "- Loader prefers ERA5 over NASA POWER for the same `(station_id, ts_utc)` (see `app/data/loaders.py`).\n",
            ]
        ),
        code_cell(
            [
                "# --- bootstrap (re-run if you opened this notebook before 00_setup) -------\n",
                "import os, sys\n",
                "REPO_DIR = \"/content/Heat-wave-backend\"\n",
                "if not os.path.exists(REPO_DIR):\n",
                "    !bash {REPO_DIR}/scripts/colab_bootstrap.sh || true\n",
                "if REPO_DIR not in sys.path:\n",
                "    sys.path.insert(0, REPO_DIR)\n",
                "os.chdir(REPO_DIR)\n",
                "print(os.getcwd())\n",
            ]
        ),
        code_cell(
            [
                "# --- write CDS credentials from Colab Secrets if present ------------------\n",
                "import os, pathlib\n",
                "cds_key = os.environ.get(\"CDSAPI_KEY\")\n",
                "if cds_key:\n",
                "    cdsapirc = pathlib.Path(\"/root/.cdsapirc\")\n",
                "    cdsapirc.write_text(\n",
                "        \"url: https://cds.climate.copernicus.eu/api\\n\"\n",
                "        f\"key: {cds_key}\\n\"\n",
                "    )\n",
                "    print(f\"wrote {cdsapirc}\")\n",
                "else:\n",
                "    print(\"CDSAPI_KEY not set — ERA5 ingest will be skipped.\")\n",
            ]
        ),
        code_cell(
            [
                "# --- run NASA POWER ingest (no auth, 5y window) ---------------------------\n",
                "from datetime import date, timedelta\n",
                "end = date.today() - timedelta(days=1)\n",
                "start_5y = end - timedelta(days=5 * 365)\n",
                "print(f\"NASA POWER window: {start_5y} -> {end}\")\n",
                "!python scripts/ingest_all.py --sources nasa_power --start {start_5y} --end {end}\n",
            ]
        ),
        code_cell(
            [
                "# --- run ERA5 ingest (3y window, requires CDSAPI_KEY) --------------------\n",
                "import os\n",
                "from datetime import date, timedelta\n",
                "if os.environ.get(\"CDSAPI_KEY\"):\n",
                "    end = date.today() - timedelta(days=1)\n",
                "    start_3y = end - timedelta(days=3 * 365)\n",
                "    print(f\"ERA5 window: {start_3y} -> {end}\")\n",
                "    !python scripts/ingest_all.py --sources era5 --start {start_3y} --end {end}\n",
                "else:\n",
                "    print(\"Skipping ERA5 — CDSAPI_KEY not set.\")\n",
            ]
        ),
        code_cell(
            [
                "# --- (optional) TMD ingest, recent window only ----------------------------\n",
                "import os\n",
                "from datetime import date, timedelta\n",
                "if os.environ.get(\"TMD_API_KEY\"):\n",
                "    end = date.today() - timedelta(days=1)\n",
                "    start_2y = end - timedelta(days=2 * 365)\n",
                "    print(f\"TMD window: {start_2y} -> {end}\")\n",
                "    !python scripts/ingest_all.py --sources tmd --start {start_2y} --end {end}\n",
                "else:\n",
                "    print(\"Skipping TMD — TMD_API_KEY not set.\")\n",
            ]
        ),
        code_cell(
            [
                "# --- coverage summary across all stations ---------------------------------\n",
                "from datetime import date, timedelta\n",
                "from app.data.stations import STATIONS\n",
                "from app.data.loaders import read_observations\n",
                "\n",
                "end = date.today() - timedelta(days=1)\n",
                "start = end - timedelta(days=5 * 365)\n",
                "\n",
                "for sid in STATIONS:\n",
                "    try:\n",
                "        df = read_observations(sid, start, end)\n",
                "        n = len(df)\n",
                "        srcs = df[\"source\"].value_counts().to_dict() if \"source\" in df.columns else {}\n",
                "        print(f\"  {sid}: {n:>7d} rows | sources={srcs}\")\n",
                "    except Exception as exc:\n",
                "        print(f\"  {sid}: ERROR {exc}\")\n",
            ]
        ),
        md_cell(
            [
                "## Next steps / ขั้นถัดไป\n",
                "\n",
                "**EN.** When coverage looks healthy, jump to `02_train_*.ipynb` (per backend) to train v3 forecasters. Artifacts will land in `app/models/forecast_v3/{station_id}/h{H}/` (Drive-backed) and can be pulled back to the local repo via the workflow described in `docs/colab-training.md`.\n",
                "\n",
                "**TH.** เมื่อ coverage ดูครบถ้วนแล้ว ไปที่ `02_train_*.ipynb` เพื่อเทรน v3 forecaster. ผลลัพธ์จะถูกเก็บที่ `app/models/forecast_v3/{station_id}/h{H}/` (อยู่บน Drive) และสามารถ push กลับเข้า repo ตามขั้นตอนใน `docs/colab-training.md`.\n",
            ]
        ),
    ]
)
write_nb("01_ingest.ipynb", ingest_nb)
