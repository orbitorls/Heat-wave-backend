"""Hard-coded registry of TMD weather stations used by HeatShield AI."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WeatherStation:
    station_id: str
    name_th: str
    lat: float
    lon: float
    elevation_m: float


STATIONS: dict[str, WeatherStation] = {
    "BKK_01": WeatherStation(
        station_id="BKK_01",
        name_th="กรุงเทพมหานคร (Don Mueang)",
        lat=13.9132,
        lon=100.6067,
        elevation_m=9.5,
    ),
    "CNX_01": WeatherStation(
        station_id="CNX_01",
        name_th="เชียงใหม่",
        lat=18.7761,
        lon=98.9769,
        elevation_m=310.0,
    ),
    "KKN_01": WeatherStation(
        station_id="KKN_01",
        name_th="ขอนแก่น",
        lat=16.4419,
        lon=102.8359,
        elevation_m=182.0,
    ),
    "HYI_01": WeatherStation(
        station_id="HYI_01",
        name_th="หาดใหญ่ (สงขลา)",
        lat=6.9269,
        lon=100.4370,
        elevation_m=8.0,
    ),
    "RYG_01": WeatherStation(
        station_id="RYG_01",
        name_th="ระยอง",
        lat=12.6815,
        lon=101.2816,
        elevation_m=14.0,
    ),
}


def get_station(station_id: str) -> WeatherStation:
    """Return a WeatherStation by ID.

    Raises:
        KeyError: if station_id is not in the registry.
    """
    if station_id not in STATIONS:
        known = ", ".join(sorted(STATIONS))
        raise KeyError(
            f"Unknown station_id '{station_id}'. Known stations: {known}"
        )
    return STATIONS[station_id]
