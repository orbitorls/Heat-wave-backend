from fastapi import APIRouter
from app.data.schemas import WhatIfRequest, WhatIfResponse, ScenarioOut
from app.core import heat_index as hi_core
from app.core import risk_scoring
from app.core.vulnerability import get_profile
from app.core.whatif import Intervention, InterventionType, simulate
from app.core.risk_scoring import ActivityIntensity

router = APIRouter()


@router.post("/simulate", response_model=WhatIfResponse)
def run_whatif(req: WhatIfRequest) -> WhatIfResponse:
    hi = hi_core.compute(req.temperature_c, req.humidity_rh)
    profile = get_profile(req.profile_id)
    base_activity = risk_scoring.ActivityContext(
        intensity=req.activity_intensity,
        duration_minutes=req.duration_minutes,
        shade_available=req.shade_available,
        water_access=req.water_access,
        time_of_day_hour=req.time_of_day_hour,
    )
    interventions = [
        Intervention(
            intervention_type=InterventionType(i.intervention_type),
            shift_hours=i.shift_hours,
            break_every_minutes=i.break_every_minutes,
            add_shade=i.add_shade,
            add_water=i.add_water,
            reduce_duration_by=i.reduce_duration_by,
            new_intensity=(
                ActivityIntensity(i.new_intensity) if i.new_intensity else None
            ),
        )
        for i in req.interventions
    ]
    result = simulate(hi, profile, base_activity, interventions, req.acclimatized)

    scenarios_out = [
        ScenarioOut(
            intervention_type=s.intervention.intervention_type.value,
            original_score=s.original_score.score,
            original_class=s.original_score.risk_class.value,
            new_score=s.new_score.score,
            new_class=s.new_score.risk_class.value,
            score_reduction=s.score_reduction,
            effective=s.effective,
            summary_th=s.summary_th,
        )
        for s in result.scenarios
    ]

    best = result.best_scenario
    return WhatIfResponse(
        original_risk_score=result.original_risk.score,
        original_risk_class=result.original_risk.risk_class.value,
        scenarios=scenarios_out,
        best_intervention=best.intervention.intervention_type.value if best else None,
        best_score_reduction=best.score_reduction if best else None,
    )
