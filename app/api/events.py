from datetime import date as date_type
from fastapi import APIRouter
from app.data.schemas import HeatwaveDetectRequest, HeatwaveDetectResponse
from app.core.adaptive_definition import (
    AdaptiveDefinitionEngine,
    DailyObs,
)

router = APIRouter()


@router.post("/detect", response_model=HeatwaveDetectResponse)
def detect_heatwave(req: HeatwaveDetectRequest) -> HeatwaveDetectResponse:
    engine = AdaptiveDefinitionEngine(
        percentile=req.percentile,
        min_consecutive_days=req.min_consecutive_days,
        require_warm_nights=req.require_warm_nights,
    )
    baseline = engine.build_baseline(
        station_id=req.station_id,
        historical_values=req.historical_values,
        metric=req.metric,
    )
    observations = [
        DailyObs(
            date=date_type.fromisoformat(obs.date),
            max_value=obs.max_value,
            min_value=obs.min_value,
        )
        for obs in req.observations
    ]
    event = engine.detect(observations, baseline)
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
