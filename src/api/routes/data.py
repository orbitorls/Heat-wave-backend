"""Data management routes"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
import os
import sys
from pathlib import Path
import threading
import uuid
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

router = APIRouter()

# Get project root for absolute paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# Track download jobs
download_jobs = {}


class DownloadRequest(BaseModel):
    year: int = 2023
    month: int = 1
    variables: list = None


@router.get("/files")
async def list_data_files():
    """List ERA5 data files"""
    era5_dir = PROJECT_ROOT / "era5_data"
    if not era5_dir.exists():
        return {"files": [], "count": 0}

    data_files = [f for f in os.listdir(era5_dir) if f.endswith('.nc')]
    return {
        "files": sorted(data_files),
        "count": len(data_files)
    }


@router.post("/audit")
async def audit_data():
    """Audit data integrity"""
    era5_dir = PROJECT_ROOT / "era5_data"
    issues = []
    files_checked = 0

    if era5_dir.exists():
        for f in os.listdir(era5_dir):
            if f.endswith('.nc'):
                files_checked += 1
                # Basic check - file not empty
                file_path = era5_dir / f
                if file_path.stat().st_size == 0:
                    issues.append(f"{f}: empty file")

    return {
        "status": "ok" if not issues else "issues_found",
        "files_checked": files_checked,
        "issues": issues
    }


def run_download_era5(job_id: str, year: int, month: int):
    """Download ERA5 data in background"""
    try:
        download_jobs[job_id]["status"] = "downloading"
        download_jobs[job_id]["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] Starting ERA5 download...")

        try:
            import cdsapi
            c = cdsapi.Client()

            output_file = str(PROJECT_ROOT / f"era5_data/era5_thailand_{year}_{month:02d}.nc")
            os.makedirs(PROJECT_ROOT / "era5_data", exist_ok=True)

            download_jobs[job_id]["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] Requesting data from CDS API...")

            # Thailand bounding box
            c.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "format": "netcdf",
                    "variable": [
                        "2m_temperature",
                        "geopotential",
                        "soil_temperature_level_1",
                        "total_precipitation",
                    ],
                    "year": str(year),
                    "month": str(month).zfill(2),
                    "day": [str(d).zfill(2) for d in range(1, 32)],
                    "time": [f"{h:02d}:00" for h in range(24)],
                    "area": [20.5, 97.5, 5.5, 106.0],  # Thailand bounding box
                },
                output_file
            )

            download_jobs[job_id]["status"] = "completed"
            download_jobs[job_id]["file"] = output_file
            download_jobs[job_id]["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] Download completed: {output_file}")

        except ImportError:
            download_jobs[job_id]["status"] = "failed"
            download_jobs[job_id]["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] Error: cdsapi not installed")
            download_jobs[job_id]["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] Run: pip install cdsapi")

        except Exception as e:
            download_jobs[job_id]["status"] = "failed"
            download_jobs[job_id]["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] Error: {str(e)}")

    except Exception as e:
        download_jobs[job_id]["status"] = "failed"
        download_jobs[job_id]["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] Error: {str(e)}")


@router.post("/download")
async def download_era5(request: DownloadRequest):
    """Start ERA5 data download from CDS API"""
    job_id = f"download_{uuid.uuid4().hex[:8]}"

    download_jobs[job_id] = {
        "id": job_id,
        "status": "starting",
        "year": request.year,
        "month": request.month,
        "logs": [],
        "created_at": datetime.now().isoformat()
    }

    # Run download in background thread
    thread = threading.Thread(
        target=run_download_era5,
        args=(job_id, request.year, request.month)
    )
    thread.daemon = True
    thread.start()

    return {"job_id": job_id, "status": "started"}


@router.get("/download/{job_id}")
async def get_download_status(job_id: str):
    """Get download job status"""
    if job_id not in download_jobs:
        raise HTTPException(status_code=404, detail="Download job not found")

    return download_jobs[job_id]
