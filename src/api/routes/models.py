"""Model management routes"""

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
import os
import sys
from pathlib import Path

# Add src to path
src_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_dir))

from src.models.manager import model_manager

router = APIRouter()


# Get project root for absolute paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


@router.get("/")
async def list_models():
    """List all available model checkpoints"""
    models_dir = PROJECT_ROOT / "models"
    if not models_dir.exists():
        return {"models": [], "count": 0}

    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    return {
        "models": sorted(model_files),
        "count": len(model_files)
    }


@router.post("/load")
async def load_model(model_name: str):
    """Load a specific model into memory"""
    try:
        model_path = PROJECT_ROOT / "models" / model_name
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

        success = model_manager.load_model(model_path)
        if success:
            return {
                "status": "loaded",
                "model": model_name,
                "model_type": model_manager.model_type,
                "metadata": {
                    "threshold": model_manager.metadata.get("heatwave_temp_threshold"),
                    "created_at": model_manager.metadata.get("created_at")
                }
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to load model")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")


@router.get("/current")
async def get_current_model():
    """Get currently loaded model info"""
    if model_manager.model is None:
        return {"loaded": False, "model": None}

    return {
        "loaded": True,
        "model": model_manager.current_path.name if model_manager.current_path else None,
        "model_type": model_manager.model_type,
        "metadata": model_manager.metadata
    }


@router.delete("/{model_name}")
async def delete_model(model_name: str):
    """Delete a model checkpoint"""
    try:
        model_path = PROJECT_ROOT / "models" / model_name
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

        # Don't allow deleting currently loaded model
        if model_manager.current_path and model_manager.current_path.name == model_name:
            raise HTTPException(status_code=400, detail="Cannot delete currently loaded model. Please load another model first.")

        os.remove(model_path)
        return {"status": "deleted", "model": model_name}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting model: {str(e)}")
