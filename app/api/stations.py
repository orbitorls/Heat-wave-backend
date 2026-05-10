from dataclasses import asdict

from fastapi import APIRouter

from app.data.stations import STATIONS

router = APIRouter()


@router.get("", summary="List all weather stations")
def list_stations() -> list[dict]:
    return [asdict(s) for s in STATIONS.values()]
