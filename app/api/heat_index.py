from fastapi import APIRouter
from app.data.schemas import HeatIndexRequest, HeatIndexResponse
from app.core import heat_index as hi_core

router = APIRouter()


@router.post("", response_model=HeatIndexResponse)
def compute_heat_index(req: HeatIndexRequest) -> HeatIndexResponse:
    result = hi_core.compute(req.temperature_c, req.humidity_rh)
    return HeatIndexResponse(
        heat_index_c=result.heat_index,
        category=result.category.value,
        temperature_c=result.temp_c,
        humidity_rh=result.humidity_rh,
    )
