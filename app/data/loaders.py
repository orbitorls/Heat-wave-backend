"""Parquet I/O helpers for weather observations."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from app.data.schemas import StationObservation

_RAW_ROOT = Path(__file__).parents[2] / "data" / "raw"


def _partition_path(station_id: str, day: date) -> Path:
    return _RAW_ROOT / f"station_id={station_id}" / f"date={day.isoformat()}" / "obs.parquet"


def write_observations(obs: list[StationObservation], station_id: str, day: date) -> Path:
    """Write observations to partitioned parquet. Idempotent (overwrites same partition)."""
    if not obs:
        return _partition_path(station_id, day)

    records = [o.model_dump() for o in obs]
    df = pd.DataFrame(records)
    path = _partition_path(station_id, day)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow")
    return path


def read_observations(
    station_id: str,
    start: date,
    end: date,
) -> pd.DataFrame:
    """Read all parquet partitions for a station between start and end (inclusive).

    Returns DataFrame with columns matching StationObservation fields.
    Returns empty DataFrame if no data found.
    """
    frames: list[pd.DataFrame] = []
    current = start
    while current <= end:
        path = _partition_path(station_id, current)
        if path.exists():
            frames.append(pd.read_parquet(path, engine="pyarrow"))
        current += timedelta(days=1)

    if not frames:
        return pd.DataFrame()

    frames = [f.dropna(axis=1, how="all") for f in frames]
    df = pd.concat(frames, ignore_index=True)
    if "ts_utc" in df.columns:
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)

    if "source" in df.columns and df["source"].nunique() > 1:
        # TMD > ERA5 > NASA POWER: highest fidelity (ground truth) wins
        source_order = pd.CategoricalDtype(["nasa_power", "era5", "tmd"], ordered=True)
        df["source"] = df["source"].astype(source_order)
        df = (df.sort_values(["ts_utc", "source"])
                .drop_duplicates(subset=["station_id", "ts_utc"], keep="last")
                .reset_index(drop=True))
    else:
        df = df.sort_values("ts_utc").reset_index(drop=True)

    return df
