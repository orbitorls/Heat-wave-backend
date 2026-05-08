"""POST /forecast/heat-index — heat index forecast endpoint."""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException

from app.data.loaders import read_observations
from app.data.schemas import ForecastRequest, ForecastResponse, StationObservation
from app.data.stations import STATIONS
from app.ml.forecast.predict import predict

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/heat-index", response_model=ForecastResponse)
def forecast_heat_index(req: ForecastRequest) -> ForecastResponse:
    """Forecast heat index for a station at requested horizons.

    If `recent_obs` is not provided, loads the last 48 hours from the local parquet store.
    Requires a trained model (run scripts/train_forecast.py first).
    """
    if req.station_id not in STATIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown station_id '{req.station_id}'. Valid: {list(STATIONS.keys())}"
        )

    recent_obs: list[StationObservation] | None = req.recent_obs

    # Load from parquet if not provided
    if recent_obs is None:
        today = date.today()
        # Pull 3 days to ensure we have 48+ hours of recent observations
        # (predict() requires max(lags_h)+2 = 26 minimum)
        start = today - timedelta(days=3)
        try:
            import pandas as pd
            df = read_observations(req.station_id, start, today)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Failed to load observations: {exc}")

        if df.empty:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"No local observations for station '{req.station_id}'. "
                    "Provide recent_obs in the request body, or run scripts/ingest_tmd.py first."
                )
            )

        from app.data.schemas import StationObservation as SO
        import pandas as pd
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
        recent_obs = [
            SO(
                station_id=req.station_id,
                ts_utc=row.ts_utc,
                temp_c=float(row.temp_c),
                rh=float(row.rh),
                wind_ms=float(row.wind_ms) if row.wind_ms is not None and str(row.wind_ms) != "nan" else None,
                precip_mm=float(row.precip_mm) if row.precip_mm is not None and str(row.precip_mm) != "nan" else None,
                source="cache",
            )
            for row in df.itertuples(index=False)
        ]

    try:
        forecasts, low_confidence, confidence_reason = predict(
            station_id=req.station_id,
            recent_obs=recent_obs,
            horizons=req.horizons,
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="No trained forecast model found. Run scripts/train_forecast.py first."
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    model_version = forecasts[0].model_version if forecasts else "unknown"

    return ForecastResponse(
        station_id=req.station_id,
        generated_at=datetime.now(timezone.utc),
        forecasts=forecasts,
        model_version=model_version,
        low_confidence=low_confidence,
        confidence_reason=confidence_reason,
    )
