"""Async TMD (Thai Meteorological Department) API client.

Authentication: Bearer token via TMD_API_KEY env var.
Rate limiting: Respects Retry-After header on 429.
Retry policy: 3 attempts, exponential backoff, fail-fast on 401.
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import date, datetime, timezone
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    RetryCallState,
)

from app.data.schemas import StationObservation

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://data.tmd.go.th/api/Weather/v1"


class TMDAPIError(Exception):
    """Raised when TMD API returns an unexpected error."""
    def __init__(self, status_code: int, message: str):
        super().__init__(f"TMD API error {status_code}: {message}")
        self.status_code = status_code


class TMDAuthError(TMDAPIError):
    """Raised on 401 — do not retry."""


def _should_retry(exc: BaseException) -> bool:
    """Retry on 5xx and 429, but NOT on 401 (auth failure)."""
    if isinstance(exc, TMDAuthError):
        return False
    if isinstance(exc, TMDAPIError) and exc.status_code == 401:
        return False
    return True


def _log_retry(retry_state: RetryCallState) -> None:
    if retry_state.outcome and retry_state.outcome.failed:
        exc = retry_state.outcome.exception()
        logger.warning(
            "TMD API retry attempt %d after error: %s",
            retry_state.attempt_number,
            exc,
        )


class TMDClient:
    """Async client for the TMD hourly weather observation API."""

    def __init__(
        self,
        api_token: str | None = None,
        base_url: str | None = None,
        timeout_s: float = 15.0,
    ) -> None:
        self._token = api_token or os.environ.get("TMD_API_KEY", "")
        self._base_url = (
            base_url
            or os.environ.get("TMD_BASE_URL", _DEFAULT_BASE_URL)
        ).rstrip("/")
        self._timeout = timeout_s

    async def fetch_hourly(
        self, station_id: str, day: date
    ) -> list[StationObservation]:
        """Fetch hourly observations for a station on a given day.

        Args:
            station_id: TMD station identifier (e.g. "BKK_01").
            day: The date to fetch (UTC).

        Returns:
            List of StationObservation (may be fewer than 24 if station had gaps).

        Raises:
            TMDAuthError: On 401 — check TMD_API_KEY.
            TMDAPIError: On other HTTP errors after retries.
        """
        return await self._fetch_with_retry(station_id, day)

    @retry(
        retry=retry_if_exception(_should_retry),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        after=_log_retry,
        reraise=True,
    )
    async def _fetch_with_retry(
        self, station_id: str, day: date
    ) -> list[StationObservation]:
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/json",
        }
        params = {
            "stationid": station_id,
            "date": day.isoformat(),
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(
                f"{self._base_url}/hourly",
                headers=headers,
                params=params,
            )

        if response.status_code == 401:
            raise TMDAuthError(401, "Invalid or missing TMD_API_KEY")

        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", "5"))
            logger.warning("TMD rate-limited. Waiting %ds (Retry-After).", retry_after)
            await asyncio.sleep(retry_after)
            raise TMDAPIError(429, "Rate limited")

        if response.status_code >= 500:
            raise TMDAPIError(response.status_code, response.text[:200])

        if response.status_code != 200:
            raise TMDAPIError(response.status_code, response.text[:200])

        return self._parse_response(response.json(), station_id, day)

    def _parse_response(
        self, data: dict[str, Any], station_id: str, day: date
    ) -> list[StationObservation]:
        """Parse TMD API JSON response into StationObservation list.

        TMD API response format (hourly):
        {
          "Observations": {
            "ObservationForHourly": [
              {
                "DateTime": "2024-05-01T13:00:00+07:00",
                "Temp": 35.2,
                "Humid": 72.0,
                "WindSpeed": 3.1,
                "Rain": 0.0,
                ...
              },
              ...
            ]
          }
        }

        If the actual TMD API returns a different format, this parser handles
        gracefully: missing optional fields default to None.
        """
        observations: list[StationObservation] = []

        # Navigate to the hourly records list — handle different TMD response shapes
        records: list[dict] = []
        if isinstance(data, list):
            records = data
        elif "Observations" in data:
            obs_block = data["Observations"]
            if isinstance(obs_block, dict):
                for key in ("ObservationForHourly", "Hourly", "data"):
                    if key in obs_block and isinstance(obs_block[key], list):
                        records = obs_block[key]
                        break
            elif isinstance(obs_block, list):
                records = obs_block
        elif "data" in data and isinstance(data["data"], list):
            records = data["data"]

        for record in records:
            try:
                # Try multiple possible field names (TMD API versions vary)
                dt_raw = (
                    record.get("DateTime")
                    or record.get("datetime")
                    or record.get("time")
                )
                temp_raw = (
                    record.get("Temp")
                    or record.get("temp")
                    or record.get("temperature")
                )
                rh_raw = (
                    record.get("Humid")
                    or record.get("humid")
                    or record.get("humidity")
                    or record.get("rh")
                )
                if dt_raw is None or temp_raw is None or rh_raw is None:
                    logger.debug("Skipping record with missing required fields: %s", record)
                    continue

                ts = datetime.fromisoformat(str(dt_raw)).astimezone(timezone.utc)

                obs = StationObservation(
                    station_id=station_id,
                    ts_utc=ts,
                    temp_c=float(temp_raw),
                    rh=float(rh_raw),
                    wind_ms=_safe_float(record.get("WindSpeed") or record.get("wind_speed")),
                    precip_mm=_safe_float(record.get("Rain") or record.get("rain") or record.get("precip")),
                    source="tmd",
                )
                observations.append(obs)
            except (ValueError, TypeError, KeyError) as exc:
                logger.debug("Failed to parse record %s: %s", record, exc)
                continue

        logger.info(
            "Fetched %d observations for station=%s date=%s",
            len(observations),
            station_id,
            day.isoformat(),
        )
        return observations


def _safe_float(val: Any) -> float | None:
    """Return float or None for missing/invalid values."""
    if val is None:
        return None
    try:
        f = float(val)
        return f if not (f != f) else None  # filter NaN
    except (ValueError, TypeError):
        return None
