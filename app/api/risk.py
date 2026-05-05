from fastapi import APIRouter
from app.data.schemas import RiskScoreRequest, RiskScoreResponse
from app.core import heat_index as hi_core
from app.core import risk_scoring
from app.core.vulnerability import get_profile

router = APIRouter()


@router.post("/score", response_model=RiskScoreResponse)
def compute_risk(req: RiskScoreRequest) -> RiskScoreResponse:
    hi = hi_core.compute(req.temperature_c, req.humidity_rh)
    profile = get_profile(req.profile_id)
    activity = risk_scoring.ActivityContext(
        intensity=req.activity_intensity,
        duration_minutes=req.duration_minutes,
        shade_available=req.shade_available,
        water_access=req.water_access,
        time_of_day_hour=req.time_of_day_hour,
    )
    result = risk_scoring.score(
        hi_result=hi,
        profile=profile,
        activity=activity,
        acclimatized=req.acclimatized,
        low_confidence=req.low_confidence,
    )
    return RiskScoreResponse(
        score=result.score,
        risk_class=result.risk_class,
        dominant_factors=result.dominant_factors,
        confidence=result.confidence,
        heat_index_c=result.heat_index_c,
        profile_id=result.profile_id,
        conservative_applied=result.conservative_applied,
    )
