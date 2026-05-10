from dataclasses import asdict

from fastapi import APIRouter

from app.core.vulnerability import PROFILES

router = APIRouter()


@router.get("", summary="List all vulnerability profiles")
def list_profiles() -> list[dict]:
    return [asdict(p) for p in PROFILES.values()]
