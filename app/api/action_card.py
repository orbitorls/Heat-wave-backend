from fastapi import APIRouter
from app.data.schemas import ActionCardRequest, ActionCardResponse
from app.core import heat_index as hi_core
from app.core import risk_scoring
from app.core.vulnerability import get_profile
from app.core import action_card as ac_core

router = APIRouter()


@router.post("", response_model=ActionCardResponse)
def generate_action_card(req: ActionCardRequest) -> ActionCardResponse:
    hi = hi_core.compute(req.temperature_c, req.humidity_rh)
    profile = get_profile(req.profile_id)
    activity = risk_scoring.ActivityContext(
        intensity=req.activity_intensity,
        duration_minutes=req.duration_minutes,
        shade_available=req.shade_available,
        water_access=req.water_access,
        time_of_day_hour=req.time_of_day_hour,
    )
    rs = risk_scoring.score(hi, profile, activity, req.acclimatized)
    card = ac_core.generate(
        risk_score=rs,
        profile=profile,
        activity=activity,
        audience=req.audience,
        forecast_horizon_h=req.forecast_horizon_h,
    )
    return ActionCardResponse(
        risk_class=card.risk_class.value,
        audience=card.audience.value,
        headline_th=card.headline_th,
        immediate_actions=card.immediate_actions,
        monitoring_points=card.monitoring_points,
        when_to_escalate=card.when_to_escalate,
        valid_until_hint=card.valid_until_hint,
        confidence_note=card.confidence_note,
        dominant_factors=card.dominant_factors,
    )
