"""
Model drift detection for heatwave forecasting.
Tracks rolling prediction statistics and alerts when output drifts
more than a configured threshold from the historical baseline.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np

LOGGER = logging.getLogger(__name__)

DRIFT_STORE_PATH = os.getenv("DRIFT_STORE_PATH", "logs/drift_history.json")
DRIFT_MAE_THRESHOLD = float(os.getenv("DRIFT_MAE_THRESHOLD", "5.0"))   # °C
DRIFT_WINDOW = int(os.getenv("DRIFT_WINDOW", "30"))                     # predictions to keep


def _load_store() -> dict:
    path = Path(DRIFT_STORE_PATH)
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            LOGGER.warning("Could not read drift store: %s", e)
    return {"predictions": [], "baseline_mean": None, "baseline_std": None, "alerts": []}


def _save_store(store: dict) -> None:
    path = Path(DRIFT_STORE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(store, f, indent=2, default=str)
    except Exception as e:
        LOGGER.warning("Could not save drift store: %s", e)


def record_prediction(predicted_temp: float, region: str = "thailand") -> Optional[Dict]:
    """
    Record a new prediction. Returns a drift alert dict if drift is detected, else None.
    """
    store = _load_store()
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "region": region,
        "temp": float(predicted_temp),
    }
    store["predictions"].append(entry)
    # keep only recent window
    store["predictions"] = store["predictions"][-DRIFT_WINDOW * 10:]

    recent = [p["temp"] for p in store["predictions"][-DRIFT_WINDOW:]]
    if len(recent) < 5:
        _save_store(store)
        return None

    current_mean = float(np.mean(recent))
    current_std = float(np.std(recent))

    # First time: establish baseline
    if store["baseline_mean"] is None:
        store["baseline_mean"] = current_mean
        store["baseline_std"] = current_std
        LOGGER.info("Drift baseline established: mean=%.2f std=%.2f", current_mean, current_std)
        _save_store(store)
        return None

    # Check drift
    mae_from_baseline = abs(current_mean - store["baseline_mean"])
    alert = None
    if mae_from_baseline > DRIFT_MAE_THRESHOLD:
        alert = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "type": "mean_drift",
            "baseline_mean": store["baseline_mean"],
            "current_mean": current_mean,
            "drift_magnitude": round(mae_from_baseline, 3),
            "threshold": DRIFT_MAE_THRESHOLD,
            "message": (
                f"Model output mean drifted {mae_from_baseline:.1f}°C from baseline "
                f"({store['baseline_mean']:.1f}°C → {current_mean:.1f}°C)"
            ),
        }
        store["alerts"].append(alert)
        store["alerts"] = store["alerts"][-100:]  # keep last 100 alerts
        LOGGER.warning("DRIFT ALERT: %s", alert["message"])

    _save_store(store)
    return alert


def get_drift_status() -> Dict:
    """Return current drift status summary."""
    store = _load_store()
    recent = [p["temp"] for p in store["predictions"][-DRIFT_WINDOW:]]
    return {
        "n_predictions_tracked": len(store["predictions"]),
        "baseline_mean": store["baseline_mean"],
        "recent_mean": float(np.mean(recent)) if recent else None,
        "recent_std": float(np.std(recent)) if recent else None,
        "drift_threshold": DRIFT_MAE_THRESHOLD,
        "recent_alerts": store["alerts"][-5:],
        "drift_detected": len(store["alerts"]) > 0 and (
            store["alerts"][-1]["ts"] == store["alerts"][-1]["ts"]  # always True, just surface latest
        ),
    }


def reset_baseline() -> None:
    """Reset baseline to current prediction distribution (e.g. after retraining)."""
    store = _load_store()
    recent = [p["temp"] for p in store["predictions"][-DRIFT_WINDOW:]]
    if recent:
        store["baseline_mean"] = float(np.mean(recent))
        store["baseline_std"] = float(np.std(recent))
        store["alerts"] = []
        LOGGER.info("Drift baseline reset: mean=%.2f", store["baseline_mean"])
    _save_store(store)
