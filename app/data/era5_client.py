"""ERA5 reanalysis data client — download hourly weather data for Thai stations.

Source: ECMWF ERA5 via Copernicus Climate Data Store (CDS).
Coverage: 1940-present, global, 0.25° × 0.25° grid, hourly.

Setup (one-time):
  pip install cdsapi xarray netCDF4
  Register at https://cds.climate.copernicus.eu/ → My Account → API key
  Create ~/.cdsapirc:
    url: https://cds.climate.copernicus.eu/api
    key: YOUR_PERSONAL_ACCESS_TOKEN
  OR set env vars: CDSAPI_URL, CDSAPI_KEY

Variable mapping:
  ERA5 long name                        → NetCDF short → Unit → stored as
  2m_temperature                        → t2m          → K    → temp_c (°C)
  2m_dewpoint_temperature               → d2m          → K    → rh (% via Magnus)
  10m_u_component_of_wind               → u10          → m/s  → wind_ms
  10m_v_component_of_wind               → v10          → m/s  → (combined with u10)
  total_precipitation                   → tp           → m    → precip_mm
  surface_solar_radiation_downwards     → ssrd         → J/m² → solar_wm2 (W/m²)
  total_cloud_cover                     → tcc          → 0–1  → cloud_cover
  boundary_layer_height                 → blh          → m    → blh_m
  mean_sea_level_pressure               → msl          → Pa   → pressure_hpa
"""
from __future__ import annotations

import logging
import math
import os
from datetime import date, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from app.data.schemas import StationObservation
from app.data.stations import STATIONS, WeatherStation

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).parents[2] / "data" / "era5_cache"

# Instantaneous variables (returned together by CDS in one request)
_ERA5_INSTANT_VARS = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_cloud_cover",                   # tcc: 0–1 fraction
    "boundary_layer_height",               # blh: metres
    "mean_sea_level_pressure",             # msl: Pa → ÷100 → hPa
]

# Accumulated flux variables — CDS v2 requires a separate request for these
_ERA5_ACCUM_VARS = [
    "total_precipitation",                 # tp: m per hour → ×1000 → mm
    "surface_solar_radiation_downwards",   # ssrd: J/m² per hour → ÷3600 → W/m²
]

# Keep for backward compatibility
_ERA5_VARIABLES = _ERA5_INSTANT_VARS + _ERA5_ACCUM_VARS

# [North, West, South, East] — bounding box covering Thailand + buffer
_THAILAND_AREA = [22.5, 97.0, 5.0, 106.5]


def _rh_from_t_td(t_c: float, td_c: float) -> float:
    """Relative humidity (%) from 2m temperature and dewpoint via Magnus formula."""
    a, b = 17.625, 243.04
    num = math.exp(a * td_c / (b + td_c))
    den = math.exp(a * t_c / (b + t_c))
    return min(100.0, max(0.0, 100.0 * num / den))


class ERA5Client:
    """Download and parse ERA5 hourly reanalysis for HeatShield AI stations.

    Downloads one day at a time as NetCDF, caches locally, then extracts
    the nearest 0.25° grid point to each station's lat/lon.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: str = "https://cds.climate.copernicus.eu/api",
        cache_dir: Optional[Path] = None,
    ):
        self.api_key = api_key or os.getenv("CDSAPI_KEY")
        self.api_url = api_url or os.getenv("CDSAPI_URL", "https://cds.climate.copernicus.eu/api")
        self.cache_dir = Path(cache_dir) if cache_dir else _CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_day(self, station_id: str, day: date) -> list[StationObservation]:
        """Download ERA5 for *day* and extract observations at *station_id*.

        Args:
            station_id: One of the 5 registered station IDs.
            day: UTC date to download.

        Returns:
            Up to 24 hourly StationObservation records (source="era5").

        Raises:
            KeyError: Unknown station_id.
            ImportError: cdsapi / xarray / netCDF4 not installed.
            Exception: CDS API auth failure or network error.
        """
        if station_id not in STATIONS:
            raise KeyError(f"Unknown station_id '{station_id}'")
        station = STATIONS[station_id]

        _require_libs()

        nc_path = self._download_nc(day)
        return self._extract_station(nc_path, station, day)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _download_nc(self, day: date) -> Path:
        """Download ERA5 NetCDF for the Thailand bounding box (cached by date).

        CDS API v2 may return a ZIP archive containing the NetCDF — this method
        detects and unpacks it automatically.
        """
        import cdsapi
        import zipfile

        nc_path = self.cache_dir / f"era5_{day.strftime('%Y%m%d')}.nc"
        if nc_path.exists():
            logger.debug("ERA5 cache hit: %s", nc_path)
            return nc_path

        client = cdsapi.Client(url=self.api_url, key=self.api_key, quiet=True)
        logger.info("Downloading ERA5 for %s from CDS (may take 1-5 min)…", day)

        def _fetch_vars(variables: list, suffix: str) -> Path:
            tmp = self.cache_dir / f"era5_{day.strftime('%Y%m%d')}_{suffix}.tmp"
            out = self.cache_dir / f"era5_{day.strftime('%Y%m%d')}_{suffix}.nc"
            if out.exists():
                return out
            client.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "variable": variables,
                    "year": str(day.year),
                    "month": f"{day.month:02d}",
                    "day": f"{day.day:02d}",
                    "time": [f"{h:02d}:00" for h in range(24)],
                    "area": _THAILAND_AREA,
                    "format": "netcdf",
                },
                str(tmp),
            )
            if zipfile.is_zipfile(tmp):
                with zipfile.ZipFile(tmp) as zf:
                    nc_names = [n for n in zf.namelist() if n.endswith(".nc")]
                    if not nc_names:
                        raise RuntimeError(f"No .nc in CDS ZIP: {zf.namelist()}")
                    with zf.open(nc_names[0]) as src, open(out, "wb") as dst:
                        dst.write(src.read())
                tmp.unlink()
            else:
                tmp.rename(out)
            return out

        # CDS v2 requires separate requests for instantaneous vs accumulated vars
        instant_path = _fetch_vars(_ERA5_INSTANT_VARS, "instant")
        logger.info("Merging accumulated variables (tp, ssrd)…")
        try:
            accum_path = _fetch_vars(_ERA5_ACCUM_VARS, "accum")
            import xarray as xr
            ds_instant = xr.open_dataset(str(instant_path))
            ds_accum = xr.open_dataset(str(accum_path))
            merged = xr.merge([ds_instant, ds_accum])
            merged.to_netcdf(str(nc_path))
            ds_instant.close()
            ds_accum.close()
            instant_path.unlink(missing_ok=True)
            accum_path.unlink(missing_ok=True)
        except Exception as exc:
            logger.warning("Accumulated vars fetch failed (%s) — proceeding with instantaneous only", exc)
            import shutil
            shutil.copy(str(instant_path), str(nc_path))
            instant_path.unlink(missing_ok=True)

        logger.info("Downloaded → %s (%.1f MB)", nc_path, nc_path.stat().st_size / 1e6)
        return nc_path

    def _extract_station(
        self,
        nc_path: Path,
        station: WeatherStation,
        day: date,
    ) -> list[StationObservation]:
        """Extract nearest ERA5 grid point for a station from a downloaded NetCDF."""
        import numpy as np
        import xarray as xr

        ds = xr.open_dataset(str(nc_path))
        try:
            lats = ds.latitude.values
            lons = ds.longitude.values
            lat_idx = int(np.argmin(np.abs(lats - station.lat)))
            lon_idx = int(np.argmin(np.abs(lons - station.lon)))

            time_dim = "valid_time" if "valid_time" in ds.dims else "time"
            n_times = len(ds[time_dim])

            obs_list: list[StationObservation] = []
            for t_idx in range(n_times):
                raw_ts = ds[time_dim].values[t_idx]
                ts = pd.Timestamp(raw_ts).to_pydatetime()
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)

                t2m_k = float(ds["t2m"].values[t_idx, lat_idx, lon_idx])
                t2m_c = t2m_k - 273.15

                d2m_k = float(ds["d2m"].values[t_idx, lat_idx, lon_idx])
                d2m_c = d2m_k - 273.15
                rh = _rh_from_t_td(t2m_c, d2m_c)

                u10 = float(ds["u10"].values[t_idx, lat_idx, lon_idx])
                v10 = float(ds["v10"].values[t_idx, lat_idx, lon_idx])
                wind_ms = math.sqrt(u10 ** 2 + v10 ** 2)

                # Extended variables — best-effort (may be missing in older cache files)
                def _safe_era5(var: str, scale: float = 1.0) -> float | None:
                    try:
                        v = float(ds[var].values[t_idx, lat_idx, lon_idx])
                        return round(v * scale, 2)
                    except (KeyError, IndexError):
                        return None

                # ERA5 tp is in metres; hourly slice → mm (optional — accumulated var)
                tp_raw = _safe_era5("tp")
                precip_mm = max(0.0, tp_raw * 1000.0) if tp_raw is not None else 0.0

                # ssrd is hourly accumulation (J/m²); divide by 3600 → W/m²
                solar = _safe_era5("ssrd", 1.0 / 3600.0)
                solar = max(0.0, solar) if solar is not None else None

                tcc = _safe_era5("tcc")
                tcc = min(1.0, max(0.0, tcc)) if tcc is not None else None

                blh = _safe_era5("blh")
                msl_pa = _safe_era5("msl")
                pressure_hpa = round(msl_pa / 100.0, 1) if msl_pa is not None else None

                obs_list.append(
                    StationObservation(
                        station_id=station.station_id,
                        ts_utc=ts,
                        temp_c=round(t2m_c, 2),
                        rh=round(rh, 1),
                        wind_ms=round(wind_ms, 2),
                        precip_mm=round(precip_mm, 2),
                        source="era5",
                        solar_wm2=solar,
                        cloud_cover=tcc,
                        blh_m=blh,
                        pressure_hpa=pressure_hpa,
                    )
                )
        finally:
            ds.close()

        logger.debug("Extracted %d observations for %s %s", len(obs_list), station.station_id, day)
        return obs_list


def _require_libs() -> None:
    missing = []
    for lib in ("cdsapi", "xarray", "netCDF4"):
        try:
            __import__(lib)
        except ImportError:
            missing.append(lib)
    if missing:
        raise ImportError(
            f"ERA5 ingestion requires: pip install {' '.join(missing)}\n"
            "Then register at https://cds.climate.copernicus.eu/ and create ~/.cdsapirc"
        )
