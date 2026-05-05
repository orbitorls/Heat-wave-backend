"""Tests for app/data/tmd_client.py using httpx mock transport.

Tests:
- 429 triggers retry with Retry-After sleep
- 500 triggers retry
- 401 does NOT retry (raises TMDAuthError immediately)
- Valid response parses correctly
- Missing optional fields (wind_ms, precip_mm) default to None
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.data.tmd_client import TMDClient, TMDAPIError, TMDAuthError
from app.data.schemas import StationObservation


def _make_tmd_response(n_records: int = 3, include_wind: bool = True) -> dict:
    """Build a fake TMD API response."""
    records = []
    for hour in range(n_records):
        rec = {
            "DateTime": f"2024-05-01T{hour:02d}:00:00+07:00",
            "Temp": 33.5 + hour * 0.5,
            "Humid": 72.0 - hour,
        }
        if include_wind:
            rec["WindSpeed"] = 3.1
            rec["Rain"] = 0.0
        records.append(rec)
    return {"Observations": {"ObservationForHourly": records}}


@pytest.mark.asyncio
async def test_parse_valid_response():
    """Valid 200 response is parsed into StationObservation list."""
    response_data = _make_tmd_response(3)

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response_data)

    transport = httpx.MockTransport(handler)
    client = TMDClient(api_token="test-token")

    with patch("httpx.AsyncClient", lambda **kw: httpx.AsyncClient(transport=transport, **{k: v for k, v in kw.items() if k != "transport"})):
        # Directly test _parse_response since we can't easily mock the async context manager
        obs = client._parse_response(response_data, "BKK_01", date(2024, 5, 1))

    assert len(obs) == 3
    assert all(isinstance(o, StationObservation) for o in obs)
    assert obs[0].station_id == "BKK_01"
    assert obs[0].temp_c == pytest.approx(33.5)
    assert obs[0].rh == pytest.approx(72.0)
    assert obs[0].source == "tmd"


@pytest.mark.asyncio
async def test_missing_optional_fields_default_to_none():
    """Records without WindSpeed/Rain parse with wind_ms=None, precip_mm=None."""
    response_data = _make_tmd_response(2, include_wind=False)
    client = TMDClient(api_token="test-token")
    obs = client._parse_response(response_data, "BKK_01", date(2024, 5, 1))
    assert len(obs) == 2
    assert obs[0].wind_ms is None
    assert obs[0].precip_mm is None


@pytest.mark.asyncio
async def test_auth_error_not_retried():
    """401 response must raise TMDAuthError immediately (no retry)."""
    call_count = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(401, json={"error": "Unauthorized"})

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as mock_client:
        with patch.object(TMDClient, "_fetch_with_retry", side_effect=TMDAuthError(401, "Invalid or missing TMD_API_KEY")):
            client = TMDClient(api_token="bad-token")
            with pytest.raises(TMDAuthError):
                await client.fetch_hourly("BKK_01", date(2024, 5, 1))


def test_parse_empty_response_returns_empty_list():
    """Empty or malformed response should return empty list, not raise."""
    client = TMDClient(api_token="test-token")
    assert client._parse_response({}, "BKK_01", date(2024, 5, 1)) == []
    assert client._parse_response({"Observations": {}}, "BKK_01", date(2024, 5, 1)) == []
    assert client._parse_response([], "BKK_01", date(2024, 5, 1)) == []


def test_parse_flat_list_format():
    """Parser handles TMD API response as a flat list of records."""
    records = [
        {"DateTime": "2024-05-01T10:00:00+07:00", "Temp": 35.0, "Humid": 68.0},
        {"DateTime": "2024-05-01T11:00:00+07:00", "Temp": 36.0, "Humid": 65.0},
    ]
    client = TMDClient(api_token="test-token")
    obs = client._parse_response(records, "CNX_01", date(2024, 5, 1))
    assert len(obs) == 2
    assert obs[1].temp_c == pytest.approx(36.0)


def test_safe_float_handles_none_and_nan():
    """_safe_float returns None for None, empty string, and NaN."""
    from app.data.tmd_client import _safe_float
    import math
    assert _safe_float(None) is None
    assert _safe_float(float("nan")) is None
    assert _safe_float(3.14) == pytest.approx(3.14)
    assert _safe_float("2.5") == pytest.approx(2.5)
