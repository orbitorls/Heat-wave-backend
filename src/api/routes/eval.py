"""Evaluation routes"""

from fastapi import APIRouter
import os
from pathlib import Path
import torch

router = APIRouter()


@router.get("/metrics")
async def get_metrics():
    """Get model evaluation metrics from actual model checkpoints"""
    try:
        # Get project root (Heat-wave-backend directory)
        project_root = Path(__file__).parent.parent.parent.parent
        models_dir = project_root / "models"

        # Find all XGBoost models
        xgb_files = sorted(models_dir.glob("heatwave_daily_xgboost_v*.pth"))

        results = []
        for xgb_path in xgb_files:
            try:
                ckpt = torch.load(xgb_path, map_location="cpu", weights_only=False)
                meta = ckpt.get("metadata", {})
                test_m = meta.get("test_metrics", {})

                test_f1 = test_m.get("f1", 0)
                if isinstance(test_f1, (int, float)) and test_f1 > 0:
                    # Determine assessment
                    if test_f1 >= 0.7:
                        assessment = "GOOD"
                    elif test_f1 >= 0.5:
                        assessment = "MODERATE"
                    else:
                        assessment = "NEEDS IMPROVEMENT"

                    results.append({
                        "model": xgb_path.name,
                        "test_f1": test_f1,
                        "test_precision": test_m.get("precision", 0),
                        "test_recall": test_m.get("recall", 0),
                        "test_accuracy": test_m.get("accuracy", 0),
                        "assessment": assessment
                    })
            except Exception as e:
                print(f"Error loading {xgb_path.name}: {e}")
                continue

        # Also check ConvLSTM models
        convlstm_files = sorted(models_dir.glob("heatwave_model_checkpoint_v*.pth"))
        for conv_path in convlstm_files:
            try:
                ckpt = torch.load(conv_path, map_location="cpu", weights_only=False)
                meta = ckpt.get("metadata", {})

                # ConvLSTM might have different metric structure
                test_f1 = meta.get("test_f1", 0)
                if test_f1 > 0:
                    if test_f1 >= 0.7:
                        assessment = "GOOD"
                    elif test_f1 >= 0.5:
                        assessment = "MODERATE"
                    else:
                        assessment = "NEEDS IMPROVEMENT"

                    results.append({
                        "model": conv_path.name,
                        "test_f1": test_f1,
                        "test_precision": meta.get("test_precision", 0),
                        "test_recall": meta.get("test_recall", 0),
                        "test_accuracy": meta.get("test_accuracy", 0),
                        "assessment": assessment
                    })
            except Exception as e:
                print(f"Error loading {conv_path.name}: {e}")
                continue

        # Convert to API format
        models = []
        for r in results:
            models.append({
                "name": r["model"],
                "test_f1": r["test_f1"],
                "test_precision": r["test_precision"],
                "test_recall": r["test_recall"],
                "test_accuracy": r["test_accuracy"],
                "assessment": r["assessment"]
            })

        return {"models": models}

    except Exception as e:
        print(f"Error fetching metrics: {e}")
        import traceback
        traceback.print_exc()
        return {"models": []}
