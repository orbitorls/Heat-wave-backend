"""CatBoost quantile forecaster for HeatShield AI v3.

Directly predicts heat_index_c with quantile loss.
Uses same interface as LGBMDirectHIForecaster: val_es for early-stop,
refit on train+val_es at best_iter, CQR on val_cal.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from app.ml.forecast.base import BaseForecaster, PredictionBundle
from app.ml.forecast.conformal import MondianCQRCalibrator
from app.ml.forecast.splitting import (
    apply_feature_medians,
    fit_feature_medians,
    split_xy_4way,
)

logger = logging.getLogger(__name__)

_QUANTILES = {"q05": 0.05, "q50": 0.50, "q95": 0.95, "q97": 0.97}
_SEEDS = [42, 123, 7]


def _detect_catboost_device() -> str:
    """Return 'GPU' if CatBoost can use GPU, else 'CPU'."""
    import os
    if os.getenv("HEATSHIELD_FORCE_CPU") == "1":
        return "CPU"
    try:
        from catboost import CatBoostRegressor
        import numpy as np
        m = CatBoostRegressor(
            iterations=1, task_type="GPU", verbose=0, allow_writing_files=False,
        )
        m.fit(np.zeros((4, 2)), np.zeros(4), verbose=False)
        return "GPU"
    except Exception:
        return "CPU"


def _make_sample_weight(y: np.ndarray) -> np.ndarray:
    """Tail-class sample weights: 5x for HI>38°C, 9x for HI>42°C."""
    return 1.0 + 4.0 * (y > 38).astype(float) + 8.0 * (y > 42).astype(float)


class CatBoostForecaster:
    """CatBoost quantile forecaster satisfying BaseForecaster Protocol."""

    backend_name: str = "catboost_quantile"
    target_kind: str = "hi"

    def __init__(
        self,
        n_trials: int = 50,
        random_state: int = 42,
    ) -> None:
        self.n_trials = n_trials
        self.random_state = random_state
        self.feature_list: list[str] = []
        self._station_id: str = ""
        self._horizon_h: int = 24
        self._boosters: dict[str, list] = {q: [] for q in _QUANTILES}
        self._feature_medians: dict = {}
        self._calibrator: MondianCQRCalibrator | None = None
        self._metadata: dict = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | pd.DataFrame,
        *,
        station_id: str,
        horizon_h: int,
    ) -> None:
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:
            raise ImportError("pip install catboost") from exc

        started = time.perf_counter()
        if isinstance(y, pd.DataFrame):
            if "heat_index_c" not in y.columns:
                raise ValueError("CatBoostForecaster requires scalar y or y['heat_index_c']")
            y = y["heat_index_c"]

        self._station_id = station_id
        self._horizon_h = horizon_h
        self.feature_list = list(X.columns)

        valid = y.notna()
        X, y = X[valid].copy(), y[valid].copy()

        split = split_xy_4way(X, y, horizon_h=horizon_h)
        X_train, y_train = split.X_train, split.y_train
        X_val_es, y_val_es = split.X_val_es, split.y_val_es
        X_val_cal, y_val_cal = split.X_val_cal, split.y_val_cal
        X_test, y_test = split.X_test, split.y_test

        self._feature_medians = fit_feature_medians(X_train)
        X_train = apply_feature_medians(X_train, self._feature_medians)
        X_val_es = apply_feature_medians(X_val_es, self._feature_medians)
        X_val_cal = apply_feature_medians(X_val_cal, self._feature_medians)
        X_test = apply_feature_medians(X_test, self._feature_medians)

        device = _detect_catboost_device()
        logger.info(
            "CatBoostForecaster fit: station=%s h=%d device=%s rows=%d",
            station_id, horizon_h, device, len(X),
        )

        weights_train = _make_sample_weight(y_train.to_numpy(dtype=float))

        common_params = dict(
            iterations=2000,
            learning_rate=0.03,
            depth=8,
            l2_leaf_reg=3,
            od_type="Iter",
            od_wait=50,
            task_type=device,
            verbose=0,
            allow_writing_files=False,
            random_seed=self.random_state,
        )

        # Phase 1: pilot booster (q50) with early-stop on val_es -> best_iter
        pilot = CatBoostRegressor(loss_function="Quantile:alpha=0.5", **common_params)
        pilot.fit(
            X_train, y_train.to_numpy(dtype=float),
            sample_weight=weights_train,
            eval_set=(X_val_es, y_val_es.to_numpy(dtype=float)),
            use_best_model=True,
            verbose=False,
        )
        best_iter = max(pilot.get_best_iteration() or 1, 1)
        logger.info("CatBoost pilot early-stop: best_iter=%d", best_iter)
        del pilot

        # Phase 2: refit all boosters on train+val_es at best_iter (no early-stop)
        X_tv = pd.concat([X_train, X_val_es], ignore_index=True)
        y_tv = pd.concat([y_train, y_val_es], ignore_index=True)
        weights_tv = _make_sample_weight(y_tv.to_numpy(dtype=float))

        refit_params = {**common_params, "iterations": best_iter, "od_type": "Iter", "od_wait": best_iter + 1}
        for role, alpha in _QUANTILES.items():
            seed_boosters = []
            for seed in _SEEDS:
                p = {**refit_params, "random_seed": seed, "loss_function": f"Quantile:alpha={alpha}"}
                bst = CatBoostRegressor(**p)
                bst.fit(
                    X_tv, y_tv.to_numpy(dtype=float),
                    sample_weight=weights_tv,
                    verbose=False,
                )
                seed_boosters.append(bst)
            self._boosters[role] = seed_boosters
            logger.debug("CatBoost refit: role=%s alpha=%.2f seeds=%d", role, alpha, len(_SEEDS))

        # Mondrian CQR on val_cal
        bundle_val_cal = self.predict_with_pi(X_val_cal)
        station_ids_cal = np.full(len(X_val_cal), station_id)
        angles_cal = np.arctan2(X_val_cal["local_hour_sin"].values, X_val_cal["local_hour_cos"].values)
        local_hours_cal = np.round(angles_cal * 24 / (2 * np.pi)).astype(int) % 24
        danger_tier_cal = np.zeros(len(X_val_cal), dtype=int)

        self._calibrator = MondianCQRCalibrator()
        self._calibrator.fit(
            y_val_cal.to_numpy(dtype=float),
            bundle_val_cal.hi_lower,
            bundle_val_cal.hi_upper,
            station_ids=station_ids_cal,
            local_hours=local_hours_cal,
            danger_tiers=danger_tier_cal,
            alpha=0.10,
        )

        lo_cal, hi_cal = self._calibrate(X_val_cal, bundle_val_cal.hi_lower, bundle_val_cal.hi_upper)
        self._metadata["median_pi_width"] = float(np.median(hi_cal - lo_cal))

        if len(X_test) > 0:
            bundle_test = self._predict_bundle_calibrated(X_test)
            y_true = y_test.to_numpy(dtype=float)
            mae = float(np.mean(np.abs(bundle_test.hi_mean - y_true)))
            coverage = float(((y_true >= bundle_test.hi_lower) & (y_true <= bundle_test.hi_upper)).mean())
            self._metadata.update({
                "mae": round(mae, 3),
                "pi_coverage_90": round(coverage, 3),
                "n_test": len(X_test),
                "n_train": len(X_train),
                "split_metadata": split.metadata,
            })
            logger.info("CatBoost test: MAE=%.3f coverage_90=%.3f", mae, coverage)

        self._metadata["train_seconds"] = float(time.perf_counter() - started)

    def _align(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.reindex(columns=self.feature_list)
        for col, med in self._feature_medians.items():
            if col in out.columns:
                out[col] = out[col].fillna(med)
        return out.fillna(0.0)

    def _calibrate(self, X: pd.DataFrame, lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._calibrator is None:
            return lo, hi
        station_ids = np.full(len(X), self._station_id)
        angles = np.arctan2(X["local_hour_sin"].values, X["local_hour_cos"].values)
        local_hours = np.round(angles * 24 / (2 * np.pi)).astype(int) % 24
        danger_tier = np.zeros(len(X), dtype=int)
        return self._calibrator.adjust(lo, hi, station_ids=station_ids, local_hours=local_hours, danger_tiers=danger_tier)

    def _predict_raw(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        aligned = self._align(X)
        result = {}
        for role in _QUANTILES:
            preds = np.stack([b.predict(aligned) for b in self._boosters[role]], axis=0).mean(axis=0)
            result[role] = preds
        return result

    def predict_with_pi(self, X: pd.DataFrame) -> PredictionBundle:
        raw = self._predict_raw(X)
        hi_mean = raw["q50"]
        hi_lower = raw["q05"]
        hi_upper = raw["q95"]
        lo_cal, hi_cal = self._calibrate(X, hi_lower, hi_upper)
        return PredictionBundle(
            hi_mean=hi_mean,
            hi_lower=lo_cal,
            hi_upper=hi_cal,
            danger_proba=None,
        )

    def _predict_bundle_calibrated(self, X: pd.DataFrame) -> PredictionBundle:
        return self.predict_with_pi(X)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for role in _QUANTILES:
            for i, bst in enumerate(self._boosters[role]):
                bst.save_model(str(path / f"{role}_s{_SEEDS[i]}.cbm"))
        if self._calibrator:
            self._calibrator.save(path / "calibrator.json")
        meta = {
            "backend_name": self.backend_name,
            "feature_list": self.feature_list,
            "feature_medians": self._feature_medians,
            "station_id": self._station_id,
            "horizon_h": self._horizon_h,
            **self._metadata,
        }
        (path / "bundle.json").write_text(json.dumps(meta, indent=2, default=str))

    @classmethod
    def load(cls, path: Path) -> "CatBoostForecaster":
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:
            raise ImportError("pip install catboost") from exc
        path = Path(path)
        bundle = json.loads((path / "bundle.json").read_text())
        obj = cls()
        obj.feature_list = bundle["feature_list"]
        obj._feature_medians = bundle.get("feature_medians", {})
        obj._station_id = bundle.get("station_id", "")
        obj._horizon_h = int(bundle.get("horizon_h", 24))
        obj._metadata = {k: v for k, v in bundle.items() if k not in ("feature_list", "feature_medians", "station_id", "horizon_h", "backend_name")}
        for role in _QUANTILES:
            seed_boosters = []
            for seed in _SEEDS:
                p = path / f"{role}_s{seed}.cbm"
                if p.exists():
                    bst = CatBoostRegressor()
                    bst.load_model(str(p))
                    seed_boosters.append(bst)
            obj._boosters[role] = seed_boosters
        cal_path = path / "calibrator.json"
        if cal_path.exists():
            obj._calibrator = MondianCQRCalibrator.load(cal_path)
        return obj
