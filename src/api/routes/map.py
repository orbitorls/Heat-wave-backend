"""Map visualization routes"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import sys
from pathlib import Path
import os

# Add src to path
src_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_dir))

from src.models.manager import model_manager
from src.data.loader import DataLoader
import numpy as np

router = APIRouter()


@router.get("/forecast")
async def get_forecast(date: str = None, threshold: float = 38.0):
    """Get forecast data for map visualization"""
    try:
        # Ensure we have a loaded model
        if model_manager.model is None:
            # Try to auto-load latest model
            success = model_manager.load_model()
            if not success:
                raise HTTPException(status_code=400, detail="No model available. Please train or load a model first.")

        # Load data
        loader = DataLoader()
        try:
            ds = loader.load_combined()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not load ERA5 data: {e}")

        # Get grid dimensions
        lats = ds.latitude.values
        lons = ds.longitude.values

        # For map visualization, sample every Nth point to reduce data
        # Use 60 for dense Thailand coverage
        lat_step = max(1, len(lats) // 60)
        lon_step = max(1, len(lons) // 60)

        # Filter to Thailand bounds only: lat 5-21, lon 97-106
        thai_lat_mask = (lats >= 5) & (lats <= 21)
        thai_lon_mask = (lons >= 97) & (lons <= 106)
        
        thai_lats = lats[thai_lat_mask]
        thai_lons = lons[thai_lon_mask]
        
        sampled_lats = thai_lats[::lat_step]
        sampled_lons = thai_lons[::lon_step]

        # Get most recent time if no date specified
        if date is None:
            time_idx = -1
        else:
            dates = ds.time.values
            time_idx = -1
            for i, d in enumerate(dates):
                if str(d)[:10] == date:
                    time_idx = i
                    break

        # Extract temperature grid
        temp_grid = ds['t2m'].isel(time=time_idx).values
        if np.nanmean(temp_grid) > 200:
            temp_grid = temp_grid - 273.15  # Kelvin to Celsius

        # Calculate statistics
        max_temp = float(np.nanmax(temp_grid))
        avg_temp = float(np.nanmean(temp_grid))

        # Calculate heatwave area (percentage above threshold)
        above_threshold = np.sum(temp_grid > threshold)
        total_cells = np.sum(~np.isnan(temp_grid))
        heatwave_pct = (above_threshold / total_cells * 100) if total_cells > 0 else 0

        # Sample grid for response - use Thailand filtered indices
        # Find indices in original arrays
        lat_indices = np.where(thai_lat_mask)[0][::lat_step]
        lon_indices = np.where(thai_lon_mask)[0][::lon_step]
        
        sampled_grid = temp_grid[np.ix_(lat_indices, lon_indices)]

        # Convert to list for JSON serialization
        grid_data = []
        for i, lat in enumerate(sampled_lats):
            for j, lon in enumerate(sampled_lons):
                temp = sampled_grid[i, j]
                if not np.isnan(temp):
                    grid_data.append({
                        "lat": float(lat),
                        "lon": float(lon),
                        "temp": round(float(temp), 1),
                        "heatwave": bool(temp > threshold)
                    })

        return {
            "date": date or str(ds.time.values[time_idx])[:10],
            "threshold": threshold,
            "max_temp": round(max_temp, 1),
            "avg_temp": round(avg_temp, 1),
            "heatwave_area_pct": round(heatwave_pct, 1),
            "grid_points": len(grid_data),
            "grid": grid_data,
            "bounds": {
                "lat_min": float(sampled_lats.min()),
                "lat_max": float(sampled_lats.max()),
                "lon_min": float(sampled_lons.min()),
                "lon_max": float(sampled_lons.max())
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast error: {str(e)}")
