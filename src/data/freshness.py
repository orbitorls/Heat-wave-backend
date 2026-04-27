"""Data freshness tracking: record timestamps of downloaded data files and expose staleness checks."""
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

LOGGER = logging.getLogger(__name__)
METADATA_FILE = "era5_data/data_metadata.json"
STALE_THRESHOLD_DAYS = 30


def _load_metadata() -> dict:
    path = Path(METADATA_FILE)
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            LOGGER.warning("Could not read data metadata: %s", e)
    return {}


def _save_metadata(meta: dict) -> None:
    path = Path(METADATA_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)
    except Exception as e:
        LOGGER.warning("Could not save data metadata: %s", e)


def record_download(source: str, variables: Optional[list] = None, file_path: Optional[str] = None) -> None:
    """Record that data was downloaded from a source right now."""
    meta = _load_metadata()
    meta[source] = {
        "last_downloaded": datetime.now(timezone.utc).isoformat(),
        "variables": variables or [],
        "file": file_path or "",
    }
    _save_metadata(meta)
    LOGGER.info("Recorded download for source=%s", source)


def get_freshness_summary() -> dict:
    """Return a dict with staleness info for each data source."""
    meta = _load_metadata()
    now = datetime.now(timezone.utc)
    summary = {}
    for source, info in meta.items():
        try:
            last_dl = datetime.fromisoformat(info["last_downloaded"])
            age_days = (now - last_dl).days
            is_stale = age_days > STALE_THRESHOLD_DAYS
        except Exception:
            age_days = -1
            is_stale = True
        summary[source] = {
            "last_downloaded": info.get("last_downloaded"),
            "age_days": age_days,
            "is_stale": is_stale,
            "variables": info.get("variables", []),
        }

    # Also scan era5_data/ for NetCDF files and infer age from mtime if no metadata
    era5_dir = Path("era5_data")
    if era5_dir.exists():
        nc_files = list(era5_dir.glob("*.nc")) + list(era5_dir.glob("**/*.nc"))
        if nc_files:
            newest = max(nc_files, key=lambda p: p.stat().st_mtime)
            mtime = datetime.fromtimestamp(newest.stat().st_mtime, tz=timezone.utc)
            age_days = (now - mtime).days
            summary["_files_newest_nc"] = {
                "file": newest.name,
                "last_modified": mtime.isoformat(),
                "age_days": age_days,
                "is_stale": age_days > STALE_THRESHOLD_DAYS,
            }

    return summary


def is_data_stale() -> bool:
    """Return True if ANY data source is stale or no metadata exists."""
    summary = get_freshness_summary()
    if not summary:
        return True
    return any(v.get("is_stale", True) for v in summary.values())
