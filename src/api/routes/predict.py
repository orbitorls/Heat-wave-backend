"""Prediction routes"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import sys
from pathlib import Path
import os

# Add src to path for imports
src_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_dir))

from src.models.manager import model_manager
from src.data.loader import DataLoader
import numpy as np

router = APIRouter()

# Get project root for absolute paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


class PredictRequest(BaseModel):
    model_name: str
    date: str
    lat: float
    lon: float


@router.post("/")
async def predict(request: PredictRequest):
    """Run prediction with the loaded model"""
    try:
        # Ensure model is loaded
        if model_manager.model is None:
            # Try to load the requested model or latest
            model_path = None
            if request.model_name:
                model_path = PROJECT_ROOT / "models" / request.model_name
                if not model_path.exists():
                    model_path = None

            success = model_manager.load_model(model_path)
            if not success:
                raise HTTPException(status_code=400, detail="No model available. Please load a model first.")

        # Load data for the requested date
        loader = DataLoader()
        try:
            ds = loader.load_combined()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not load data: {e}")

        # Find date index
        dates = ds['date'].values if 'date' in ds.coords else ds['time'].values
        date_str = str(request.date)

        # Find closest date
        date_idx = None
        for i, d in enumerate(dates):
            d_str = str(d)[:10]  # YYYY-MM-DD format
            if d_str == date_str:
                date_idx = i
                break

        if date_idx is None:
            # Use most recent date as fallback
            date_idx = len(dates) - 1

        # Extract features at the specified location
        lat_idx = np.argmin(np.abs(ds.latitude.values - request.lat))
        lon_idx = np.argmin(np.abs(ds.longitude.values - request.lon))

        # Extract daily features
        temp = float(ds['t2m'].isel(time=date_idx, latitude=lat_idx, longitude=lon_idx).values)
        if temp > 200:  # Kelvin to Celsius
            temp = temp - 273.15

        # Get other features if available
        try:
            z500 = float(ds['z'].isel(time=date_idx, pressure_level=0, latitude=lat_idx, longitude=lon_idx).values)
        except:
            z500 = 50000  # Default

        try:
            soil = float(ds['swvl1'].isel(time=date_idx, latitude=lat_idx, longitude=lon_idx).values)
        except:
            soil = 0.3  # Default

        # Build feature vector (11 features matching XGBoost training)
        # Order: temp_mean, temp_max, temp_min, temp_std, temp_range,
        #        hot_fraction, z_mean, z_std, swvl1_mean, tp_mean, humidity_mean
        hot_fraction = 0.3  # Default: assume 30% of area is hot
        z_std = 1000.0      # Default geopotential std
        tp_mean = 0.0       # Default precipitation
        humidity_mean = 70.0  # Default humidity

        features = np.array([[
            temp,
            temp,  # temp_max approx
            temp,  # temp_min approx
            2.0,   # temp_std placeholder
            0.0,   # temp_range placeholder
            hot_fraction,
            z500,
            z_std,
            soil,
            tp_mean,
            humidity_mean
        ]])

        # Run prediction
        result = model_manager.predict_event(features)

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        prob = result["probabilities"][0] if result["probabilities"] else 0.5

        return {
            "temperature": round(temp, 1),
            "heatwave_probability": round(prob, 3),
            "is_heatwave": prob > 0.5,
            "model_used": str(model_manager.current_path.name) if model_manager.current_path else "unknown"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
