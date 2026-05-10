"""Auto heatwave detection using stored station data.

This endpoint fetches historical and recent data from parquet files,
calculates local baseline, and detects heatwave events automatically.
"""
from datetime import date, timedelta
from typing import List

from fastapi import APIRouter, HTTPException
import pandas as pd
import numpy as np

from app.data.schemas import HeatwaveDetectResponse, DailyObsInput
from app.data.loaders import read_observations
from app.core.adaptive_definition import AdaptiveDefinitionEngine, DailyObs, LocalBaseline
from app.core.heat_index import heat_index_rothfusz

router = APIRouter()


def _calculate_daily_observations(df: pd.DataFrame) -> List[DailyObs]:
    """Convert hourly observations to daily max/min.
    
    Uses heat_index if temperature and humidity available,
    otherwise uses temperature directly.
    """
    if df.empty:
        return []
    
    # Ensure we have required columns
    if "temp_c" not in df.columns:
        return []
    
    # Calculate heat index if humidity available
    if "humidity_pct" in df.columns:
        df = df.copy()
        df["heat_index"] = df.apply(
            lambda row: heat_index_rothfusz(row["temp_c"], row["humidity_pct"])
            if pd.notna(row.get("humidity_pct"))
            else row["temp_c"],
            axis=1,
        )
        value_col = "heat_index"
    else:
        value_col = "temp_c"
    
    # Group by date and get max/min
    df["date"] = pd.to_datetime(df["ts_utc"]).dt.date
    daily = df.groupby("date").agg(
        max_value=(value_col, "max"),
        min_value=(value_col, "min"),
    ).reset_index()
    
    return [
        DailyObs(
            date=row["date"],
            max_value=row["max_value"],
            min_value=row["min_value"],
        )
        for _, row in daily.iterrows()
    ]


def _build_baseline_from_data(
    station_id: str,
    historical_obs: List[DailyObs],
    metric: str = "heat_index",
) -> LocalBaseline:
    """Build baseline from historical daily observations."""
    if not historical_obs:
        # Fallback: create a dummy baseline with reasonable defaults for Thailand
        return LocalBaseline(
            station_id=station_id,
            metric=metric,
            mean=35.0,
            p90=38.0,
            p95=40.0,
            sample_years=0,
        )
    
    max_values = [obs.max_value for obs in historical_obs]
    
    return LocalBaseline(
        station_id=station_id,
        metric=metric,
        mean=float(np.mean(max_values)),
        p90=float(np.percentile(max_values, 90)),
        p95=float(np.percentile(max_values, 95)),
        sample_years=len(historical_obs) // 365,  # Rough estimate
    )


@router.get("/auto-detect", response_model=HeatwaveDetectResponse)
def auto_detect_heatwave(
    station_id: str,
    days_back: int = 14,
    baseline_days: int = 60,
    percentile: float = 90.0,
    min_consecutive_days: int = 2,
    require_warm_nights: bool = False,
) -> HeatwaveDetectResponse:
    """Auto-detect heatwave events using stored station data.
    
    Parameters
    ----------
    station_id : str
        Station ID to analyze (e.g., "48407" for Chiang Mai)
    days_back : int
        Number of recent days to analyze for heatwave detection
    baseline_days : int
        Number of historical days to use for baseline calculation
    percentile : float
        Percentile threshold (default 90.0)
    min_consecutive_days : int
        Minimum consecutive days above threshold
    require_warm_nights : bool
        Also require nighttime temperatures above threshold
    
    Returns
    -------
    HeatwaveDetectResponse
        Detection results with baseline statistics
    """
    today = date.today()
    
    # Date ranges
    recent_end = today
    recent_start = today - timedelta(days=days_back)
    baseline_end = recent_start - timedelta(days=1)
    baseline_start = baseline_end - timedelta(days=baseline_days)
    
    # Fetch data
    try:
        recent_df = read_observations(station_id, recent_start, recent_end)
        baseline_df = read_observations(station_id, baseline_start, baseline_end)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read station data: {str(e)}"
        )
    
    # Check if we have recent data
    if recent_df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No recent data found for station {station_id} in the last {days_back} days. "
                   "Data may need to be ingested first."
        )
    
    # Convert to daily observations
    recent_daily = _calculate_daily_observations(recent_df)
    baseline_daily = _calculate_daily_observations(baseline_df)
    
    if not recent_daily:
        raise HTTPException(
            status_code=422,
            detail="Unable to calculate daily observations from available data."
        )
    
    # Build baseline
    baseline = _build_baseline_from_data(station_id, baseline_daily, "heat_index")
    
    # Run detection
    engine = AdaptiveDefinitionEngine(
        percentile=percentile,
        min_consecutive_days=min_consecutive_days,
        require_warm_nights=require_warm_nights,
    )
    
    event = engine.detect(recent_daily, baseline)
    
    return HeatwaveDetectResponse(
        is_heatwave=event.is_heatwave,
        event_type=event.event_type,
        percentile_threshold=event.percentile_threshold,
        consecutive_days=event.consecutive_days,
        consecutive_nights=event.consecutive_nights,
        trigger_reason=event.trigger_reason,
        local_baseline={
            "station_id": baseline.station_id,
            "metric": baseline.metric,
            "mean": baseline.mean,
            "p90": baseline.p90,
            "p95": baseline.p95,
            "sample_years": baseline.sample_years,
        },
    )
