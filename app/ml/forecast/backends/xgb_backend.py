"""XGBoost backend adapter for HeatShield AI forecast v3 dispatcher.

Wraps existing v2 XGBoost booster artifacts so they satisfy BaseForecaster
Protocol and can be served through the new model-agnostic predict.py path.
No retraining — this is purely an adapter layer.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from app.ml.forecast.base import BaseForecaster, PredictionBundle


class XGBForecaster:
    """Adapter that wraps existing v2 XGBoost booster dicts as a BaseForecaster."""

    backend_name: str = "xgboost"
    target_kind: Literal["hi", "th"] = "hi"

    def __init__(self) -> None:
        self.feature_list: list[str] = []
        self._loaded: dict | None = None   # raw dict from registry.load_latest()
        self._metadata: dict = {}
        self._station_id: str = ""
        self._horizon_h: int = 24

    # ------------------------------------------------------------------
    # Factory — build from existing registry output
    # ------------------------------------------------------------------

    @classmethod
    def from_loaded(
        cls,
        loaded: dict,
        station_id: str,
        horizon_h: int,
    ) -> "XGBForecaster":
        """Construct adapter from the dict that registry.load_latest() returns."""
        obj = cls()
        obj._loaded = loaded
        obj._metadata = loaded.get("metadata", {})
        obj.feature_list = obj._metadata.get("feature_list", [])
        obj._station_id = station_id
        obj._horizon_h = horizon_h
        return obj

    # ------------------------------------------------------------------
    # BaseForecaster Protocol implementation
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame | pd.Series,
        *,
        station_id: str,
        horizon_h: int,
    ) -> None:
        raise NotImplementedError("XGBForecaster is a read-only adapter; use train_forecast.py to retrain")

    def predict_with_pi(
        self,
        X: pd.DataFrame,
        alpha: float = 0.10,
    ) -> PredictionBundle:
        """Run inference using the wrapped v2 booster and return PredictionBundle."""
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError("pip install xgboost") from exc

        if self._loaded is None:
            raise RuntimeError("XGBForecaster not initialised — use from_loaded()")

        dmat = xgb.DMatrix(X)
        loaded = self._loaded
        meta = self._metadata

        def _ensemble_predict(boosters: list, dmat: xgb.DMatrix) -> np.ndarray:
            preds = np.stack([b.predict(dmat) for b in boosters], axis=0)
            return preds.mean(axis=0)

        # Multi-horizon v2 layout: loaded["horizons"][h]["mean"|"q05"|"q95"]
        h = self._horizon_h
        if "horizons" in loaded and h in loaded.get("horizons", {}):
            horizon_data = loaded["horizons"][h]
            hi_mean = _ensemble_predict(horizon_data["mean"], dmat)
            hi_lower = _ensemble_predict(horizon_data["q05"], dmat)
            hi_upper = _ensemble_predict(horizon_data["q95"], dmat)
        elif "mean" in loaded:
            # Single-horizon v2
            hi_mean = _ensemble_predict(loaded["mean"], dmat)
            hi_lower = _ensemble_predict(loaded["q05"], dmat)
            hi_upper = _ensemble_predict(loaded["q95"], dmat)
        else:
            # v1 fallback: single booster, flat PI
            booster = loaded.get("booster")
            hi_mean = booster.predict(dmat)
            p90 = float(meta.get("p90_abs_err", 3.0))
            hi_lower = hi_mean - p90
            hi_upper = hi_mean + p90

        return PredictionBundle(
            hi_mean=hi_mean,
            hi_lower=hi_lower,
            hi_upper=hi_upper,
        )

    def save(self, dir_path: Path) -> None:
        raise NotImplementedError("XGBForecaster adapter does not save; artifacts managed by v2 registry path")

    @classmethod
    def load(cls, dir_path: Path) -> "XGBForecaster":
        raise NotImplementedError("Use XGBForecaster.from_loaded() with registry.load_latest() output")
