"""Download ERA5 reanalysis data for HeatShield AI stations.

ERA5 is ECMWF's global atmospheric reanalysis — free, high-quality historical
data from 1940 to present. No TMD API key required.

One-time setup:
  1. pip install cdsapi xarray netCDF4
  2. Register free account at https://cds.climate.copernicus.eu/
  3. Accept the ERA5 license on the dataset page
  4. Go to My Account → API Tokens → copy your Personal Access Token
  5. Create ~/.cdsapirc:
       url: https://cds.climate.copernicus.eu/api
       key: YOUR_PERSONAL_ACCESS_TOKEN
     OR export CDSAPI_KEY=YOUR_PERSONAL_ACCESS_TOKEN

Usage:
  # All stations, last 2 years (recommended for first run)
  python scripts/ingest_era5.py --start 2023-01-01 --end 2025-04-30

  # Single station, narrow range (quick test)
  python scripts/ingest_era5.py --station BKK_01 --start 2024-01-01 --end 2024-01-07

  # Skip dates already downloaded
  python scripts/ingest_era5.py --start 2023-01-01  # default: skip existing

Performance:
  - Each daily NetCDF (~1 MB) covers all Thai stations, so one CDS download per day.
  - CDS queue time varies: 30 s to 5 min per day. For 2 years = ~730 downloads.
  - Use --workers 2 to parallelize downloads (CDS allows limited concurrency).
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import os
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.data.era5_client import ERA5Client
from app.data.loaders import _partition_path, write_observations
from app.data.quality import filter_observations
from app.data.stations import STATIONS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def ingest_station_day(
    client: ERA5Client,
    station_id: str,
    day: date,
    force: bool = False,
) -> dict:
    """Ingest ERA5 data for one station-day. Returns summary dict.

    If the parquet already has nasa_power rows, ERA5 rows are merged in
    (both sources kept; read_observations deduplicates preferring ERA5).
    """
    parquet_path = _partition_path(station_id, day)
    if parquet_path.exists() and not force:
        existing = pd.read_parquet(parquet_path, engine="pyarrow")
        if "source" in existing.columns and "era5" in existing["source"].values:
            logger.debug("Skip (era5 exists): station=%s date=%s", station_id, day)
            return {"station_id": station_id, "date": str(day), "status": "skipped"}
        # Has nasa_power only — fetch ERA5 and merge below

    try:
        raw_obs = client.fetch_day(station_id, day)
    except Exception as exc:
        logger.error("ERA5 fetch failed: station=%s date=%s error=%s", station_id, day, exc)
        return {"station_id": station_id, "date": str(day), "status": "error", "error": str(exc)}

    kept, dropped = filter_observations(raw_obs)
    if dropped:
        logger.warning("Dropped %d/%d obs for %s %s", len(dropped), len(raw_obs), station_id, day)

    if kept:
        era5_df = pd.DataFrame([o.model_dump() for o in kept])
        if parquet_path.exists() and not force:
            existing = pd.read_parquet(parquet_path, engine="pyarrow")
            merged = pd.concat([existing, era5_df], ignore_index=True)
        else:
            merged = era5_df
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(parquet_path, index=False, engine="pyarrow")
        logger.info("Wrote %d ERA5 obs (merged): station=%s date=%s", len(kept), station_id, day)
    else:
        logger.warning("No valid observations: station=%s date=%s", station_id, day)

    return {"station_id": station_id, "date": str(day), "status": "ok", "observations": len(kept)}


def ingest_range(
    station_ids: list[str],
    start: date,
    end: date,
    force: bool = False,
    workers: int = 1,
) -> dict:
    """Download ERA5 for all station-days in date range.

    ERA5 downloads one NetCDF per calendar day covering the Thailand bounding box,
    so all stations share the same download — we download the file once and
    extract all stations from it.
    """
    client = ERA5Client()
    stats = {"ok": 0, "skipped": 0, "error": 0}

    days: list[date] = []
    current = start
    while current <= end:
        days.append(current)
        current += timedelta(days=1)

    total_days = len(days)
    total_tasks = len(station_ids) * total_days
    logger.info(
        "ERA5 backfill: %d stations × %d days = %d station-days",
        len(station_ids), total_days, total_tasks,
    )

    def process_day(day: date) -> list[dict]:
        results = []
        for sid in station_ids:
            results.append(ingest_station_day(client, sid, day, force=force))
        return results

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(process_day, d): d for d in days}
            for fut in as_completed(futures):
                for r in fut.result():
                    stats[r.get("status", "error")] = stats.get(r.get("status", "error"), 0) + 1
    else:
        for i, day in enumerate(days, 1):
            logger.info("Processing day %d/%d: %s", i, total_days, day)
            for r in process_day(day):
                status = r.get("status", "error")
                stats[status] = stats.get(status, 0) + 1

    logger.info("ERA5 backfill complete: %s", stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest ERA5 reanalysis data for Thai stations")
    parser.add_argument("--station", type=str, default=None,
                        help="Single station ID (default: all 5 stations)")
    parser.add_argument("--start", type=str, default=None,
                        help="Start date YYYY-MM-DD (default: 2 years ago)")
    parser.add_argument("--end", type=str, default=None,
                        help="End date YYYY-MM-DD (default: yesterday)")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if parquet already exists")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel CDS download workers (default: 1, max recommended: 2)")
    args = parser.parse_args()

    today = date.today()
    end_date = date.fromisoformat(args.end) if args.end else today - timedelta(days=1)
    start_date = date.fromisoformat(args.start) if args.start else end_date - timedelta(days=730)

    if args.station:
        if args.station not in STATIONS:
            logger.error("Unknown station '%s'. Valid: %s", args.station, list(STATIONS.keys()))
            sys.exit(1)
        station_ids = [args.station]
    else:
        station_ids = list(STATIONS.keys())

    if start_date > end_date:
        logger.error("--start must be before --end")
        sys.exit(1)

    logger.info(
        "Ingesting ERA5: stations=%s start=%s end=%s",
        station_ids, start_date, end_date,
    )

    try:
        ingest_range(station_ids, start_date, end_date, force=args.force, workers=args.workers)
    except ImportError as e:
        logger.error("%s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
