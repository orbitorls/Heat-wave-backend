"""Training routes"""

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
import sys
from pathlib import Path
import threading
import uuid
from datetime import datetime

# Add paths for training scripts
scripts_dir = Path(__file__).parent.parent.parent.parent / "scripts" / "training"
sys.path.insert(0, str(scripts_dir))

router = APIRouter()

# Store training jobs in memory (consider using Redis for production)
training_jobs = {}


class TrainRequest(BaseModel):
    model_type: str = "xgboost"
    config: dict = {}


def run_training_job(job_id: str, model_type: str, config: dict):
    """Run training in background thread"""
    try:
        training_jobs[job_id]["status"] = "running"
        training_jobs[job_id]["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] Starting training...")

        if model_type == "xgboost":
            try:
                from train_daily_xgboost import train_xgboost_daily

                def on_progress(stage, progress, metadata):
                    training_jobs[job_id]["progress"] = progress
                    training_jobs[job_id]["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] {stage}: {progress*100:.1f}%")

                def on_log(level, message):
                    training_jobs[job_id]["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] [{level}] {message}")

                result = train_xgboost_daily(
                    config=config,
                    on_progress=on_progress,
                    on_log=on_log
                )

                if result:
                    training_jobs[job_id]["status"] = "completed"
                    training_jobs[job_id]["result"] = result
                    training_jobs[job_id]["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] Training completed successfully!")
                else:
                    training_jobs[job_id]["status"] = "failed"
                    training_jobs[job_id]["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] Training failed")

            except ImportError as e:
                training_jobs[job_id]["status"] = "failed"
                training_jobs[job_id]["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] Error: {e}")
                training_jobs[job_id]["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] Make sure xgboost and scikit-learn are installed")
        else:
            training_jobs[job_id]["status"] = "failed"
            training_jobs[job_id]["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] Model type '{model_type}' not yet supported")

    except Exception as e:
        training_jobs[job_id]["status"] = "failed"
        training_jobs[job_id]["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] Error: {str(e)}")


@router.post("/start")
async def start_training(request: TrainRequest, background_tasks: BackgroundTasks):
    """Start model training"""
    job_id = f"train_{uuid.uuid4().hex[:8]}"

    training_jobs[job_id] = {
        "id": job_id,
        "status": "starting",
        "progress": 0.0,
        "logs": [],
        "model_type": request.model_type,
        "created_at": datetime.now().isoformat()
    }

    # Run training in background
    thread = threading.Thread(
        target=run_training_job,
        args=(job_id, request.model_type, request.config)
    )
    thread.daemon = True
    thread.start()

    return {"job_id": job_id, "status": "started"}


@router.get("/{job_id}/status")
async def get_training_status(job_id: str):
    """Get training status"""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")

    job = training_jobs[job_id]
    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job.get("progress", 0),
        "model_type": job.get("model_type"),
        "created_at": job.get("created_at")
    }


@router.get("/{job_id}/logs")
async def get_training_logs(job_id: str):
    """Get training logs"""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")

    return {
        "job_id": job_id,
        "logs": training_jobs[job_id].get("logs", [])
    }
