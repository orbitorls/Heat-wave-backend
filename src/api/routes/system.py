"""System status routes"""

from fastapi import APIRouter
from pathlib import Path
import os
import glob

router = APIRouter()


@router.get("/gpu-status")
async def get_gpu_status():
    """Get GPU/CPU status"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count() if cuda_available else 0
        device_name = torch.cuda.get_device_name(0) if cuda_available and device_count > 0 else None

        return {
            "cuda_available": cuda_available,
            "device": "cuda" if cuda_available else "cpu",
            "device_count": device_count,
            "device_name": device_name
        }
    except ImportError:
        return {"cuda_available": False, "device": "cpu", "device_count": 0, "device_name": None}


# Get project root for absolute paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


@router.get("/logs")
async def get_logs(limit: int = 100):
    """Get recent system logs from log files"""
    logs = []

    # Look for log files in common locations with absolute paths
    log_dirs = [
        PROJECT_ROOT / "logs",
        PROJECT_ROOT / "output",
        PROJECT_ROOT,
    ]

    all_log_files = []
    for log_dir in log_dirs:
        if log_dir.exists():
            for log_file in log_dir.glob("*.log"):
                all_log_files.append(str(log_file))

    # Read recent lines from log files (max 3 files)
    for log_file in sorted(all_log_files, key=os.path.getmtime, reverse=True)[:3]:
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                # Get last N lines
                recent_lines = lines[-limit:] if len(lines) > limit else lines
                for line in recent_lines:
                    line = line.strip()
                    if line:
                        logs.append(f"[{Path(log_file).name}] {line}")
        except Exception as e:
            logs.append(f"Error reading {log_file}: {e}")

    # If no log files found, return system info
    if not logs:
        logs = ["No log files found. System running normally."]

    return {"logs": logs[-limit:]}
