"""Model-agnostic forecaster interface for HeatShield AI v3.

All forecast backends (XGBoost, LightGBM, TabPFN) must satisfy BaseForecaster
so that predict.py and registry.py can dispatch without caring about the impl.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

import numpy as np
import pandas as pd


@dataclass
class PredictionBundle:
    """Unified output from any BaseForecaster backend."""
    hi_mean: np.ndarray            # point forecast, heat-index °C, shape (N,)
    hi_lower: np.ndarray           # EnbPI-calibrated lower bound
    hi_upper: np.ndarray           # EnbPI-calibrated upper bound
    temp_mean: np.ndarray | None = None   # only set when target_kind="th"
    rh_mean: np.ndarray | None = None    # only set when target_kind="th"
    danger_proba: np.ndarray | None = None  # gate classifier output, shape (N,)


@runtime_checkable
class BaseForecaster(Protocol):
    """Protocol every v3 backend must satisfy.

    backend_name identifies which implementation to instantiate on load.
    target_kind="hi" predicts heat_index_c directly.
    target_kind="th" predicts (temp_c, rh) then composes via Rothfusz.
    """
    backend_name: str
    feature_list: list[str]
    target_kind: Literal["hi", "th"]

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame | pd.Series,
        *,
        station_id: str,
        horizon_h: int,
    ) -> None:
        """Train on aligned (X, y) pair. y is DataFrame[temp_c,rh] when target_kind="th"."""
        ...

    def predict_with_pi(
        self,
        X: pd.DataFrame,
        alpha: float = 0.10,
    ) -> PredictionBundle:
        """Return point forecast + calibrated PI for rows in X."""
        ...

    def save(self, dir_path: Path) -> None:
        """Persist all artifacts into dir_path (created if absent)."""
        ...

    @classmethod
    def load(cls, dir_path: Path) -> "BaseForecaster":
        """Reconstruct from a previously saved dir_path."""
        ...
