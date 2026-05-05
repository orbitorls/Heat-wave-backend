"""LightGBM two-head quantile forecaster for HeatShield AI v3.

Predicts (temp_c, rh) jointly via 6 quantile boosters, composes to heat_index_c
via the Rothfusz formula, then applies EnbPI conformal calibration for valid PIs.
An optional DangerGate classifier routes samples ≥ 40°C through a tail regressor.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from app.ml.forecast.base import BaseForecaster, PredictionBundle
from app.ml.forecast.conformal import EnbPICalibrator, MondianCQRCalibrator
from app.ml.forecast.danger_gate import DangerGate
from app.ml.forecast.splitting import apply_feature_medians, fit_feature_medians, split_xy

logger = logging.getLogger(__name__)

_QUANTILES = [0.05, 0.50, 0.95, 0.97]
_TARGETS = ["temp_c", "rh"]
_SEEDS = [42, 123, 7]

# Detect GPU availability at import time (OpenCL backend, standard pip wheel)
# Set LGBM_DEVICE=cpu in environment to force CPU mode (use when OpenCL causes crashes)
def _detect_device() -> str:
    import os
    if os.environ.get("LGBM_DEVICE", "").lower() == "cpu":
        logger.info("LightGBM forced to CPU (LGBM_DEVICE=cpu)")
        return "cpu"
    try:
        import lightgbm as lgb
        import numpy as np
        d = lgb.Dataset(np.zeros((10, 2)), np.zeros(10))
        lgb.train({"objective": "regression", "device_type": "gpu", "verbose": -1,
                   "num_leaves": 4}, d, num_boost_round=1)
        logger.info("LightGBM GPU (OpenCL) detected — training will use GPU")
        return "gpu"
    except Exception:
        return "cpu"

_DEVICE_TYPE = _detect_device()

# Default LightGBM params (overridden by Optuna)
_DEFAULT_PARAMS = {
    "objective": "quantile",
    "metric": "quantile",
    "verbose": -1,
    "device_type": _DEVICE_TYPE,
    **({"n_jobs": -1} if _DEVICE_TYPE == "cpu" else {}),
}

# Change 6: persistent Optuna study path
_OPTUNA_STORAGE_PATH = Path(__file__).parents[4] / "app" / "models" / "forecast_v3" / "optuna_studies.db"


def _compute_hi_array(temp_c: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """Fully vectorized heat-index computation replicating app.core.heat_index.compute().

    Implements:
    - Steadman simple formula for temp_c < 27°C or rh < 40%
    - Rothfusz regression (via °F) for the main regime
    - Low-humidity adjustment (rh < 13% and 80 <= t_f <= 112)
    - High-humidity adjustment (rh > 85% and 80 <= t_f <= 87)
    All in NumPy, no Python loop.
    """
    # Change 2: vectorized, no scalar loop, no import from app.core.heat_index
    T = np.asarray(temp_c, dtype=np.float64)
    R = np.clip(np.asarray(rh, dtype=np.float64), 0.0, 100.0)

    # °C → °F
    T_f = T * 9.0 / 5.0 + 32.0

    # Steadman simple formula (used when temp_c < 27 or rh < 40)
    simple_hi_f = 0.5 * (T_f + 61.0 + (T_f - 68.0) * 1.2 + R * 0.094)
    simple_hi_c = (simple_hi_f - 32.0) * 5.0 / 9.0

    # Rothfusz regression (operates in °F)
    rothfusz_hi_f = (
        -42.379
        + 2.04901523 * T_f
        + 10.14333127 * R
        + -0.22475541 * T_f * R
        + -0.00683783 * T_f**2
        + -0.05481717 * R**2
        + 0.00122874 * T_f**2 * R
        + 0.00085282 * T_f * R**2
        + -0.00000199 * T_f**2 * R**2
    )

    # Low-humidity adjustment: rh < 13 and 80 <= t_f <= 112
    low_rh_mask = (R < 13.0) & (T_f >= 80.0) & (T_f <= 112.0)
    low_adj = ((13.0 - R) / 4.0) * np.sqrt(np.maximum(0.0, (17.0 - np.abs(T_f - 95.0)) / 17.0))
    rothfusz_hi_f_adjusted = np.where(low_rh_mask, rothfusz_hi_f - low_adj, rothfusz_hi_f)

    # High-humidity adjustment: rh > 85 and 80 <= t_f <= 87
    high_rh_mask = (R > 85.0) & (T_f >= 80.0) & (T_f <= 87.0) & ~low_rh_mask
    high_adj = ((R - 85.0) / 10.0) * ((87.0 - T_f) / 5.0)
    rothfusz_hi_f_adjusted = np.where(high_rh_mask, rothfusz_hi_f_adjusted + high_adj, rothfusz_hi_f_adjusted)

    rothfusz_hi_c = (rothfusz_hi_f_adjusted - 32.0) * 5.0 / 9.0

    # Select regime: Steadman when temp_c < 27 or rh < 40, else Rothfusz
    use_simple = (T < 27.0) | (R < 40.0)
    return np.where(use_simple, simple_hi_c, rothfusz_hi_c)


class LGBMForecaster:
    """Two-head LightGBM quantile forecaster satisfying BaseForecaster Protocol."""

    backend_name: str = "lightgbm_quantile"
    target_kind: Literal["hi", "th"] = "th"

    def __init__(self, n_trials: int = 50, random_state: int = 42) -> None:
        self.n_trials = n_trials
        self.random_state = random_state
        self.feature_list: list[str] = []
        self._station_id: str = ""
        self._horizon_h: int = 24
        # Change 3b: auto-size to _QUANTILES instead of hardcoded 3 slots
        self._boosters: dict[str, list[list]] = {t: [[] for _ in _QUANTILES] for t in _TARGETS}
        self._best_params: dict = {}
        self._calibrator: EnbPICalibrator | None = None
        self._gate: DangerGate | None = None
        self._feature_medians: dict = {}
        self._metadata: dict = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        *,
        station_id: str,
        horizon_h: int,
    ) -> None:
        """Train all quantile boosters + calibrator + danger gate.

        y must be a DataFrame with columns ["temp_c", "rh"].
        """
        try:
            import lightgbm as lgb
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError as exc:
            raise ImportError("pip install lightgbm optuna") from exc

        started = time.perf_counter()
        self._station_id = station_id
        self._horizon_h = horizon_h
        self.feature_list = list(X.columns)

        # Align and drop rows with missing labels. Feature imputation is fitted
        # from the training split only after chronological splitting.
        valid = y.notna().all(axis=1)
        X, y = X[valid].copy(), y[valid].copy()

        split = split_xy(X, y, horizon_h=horizon_h)
        X_train, y_train = split.X_train, split.y_train
        X_val, y_val = split.X_val, split.y_val
        X_test, y_test = split.X_test, split.y_test
        n = len(X)
        n_train = len(X_train)
        n_val = len(X_val)   # Change 1: fix NameError — n_val was logged but never defined
        self._feature_medians = fit_feature_medians(X_train)
        X_train = apply_feature_medians(X_train, self._feature_medians)
        X_val = apply_feature_medians(X_val, self._feature_medians)
        X_test = apply_feature_medians(X_test, self._feature_medians)

        logger.info("LGBMForecaster fit: station=%s h=%d rows=%d (train=%d val=%d test=%d)",
                    station_id, horizon_h, n, n_train, n_val, len(X_test))

        # Sample weights (on training HI, derived from T+RH)
        from app.ml.forecast.train import _make_sample_weight
        y_hi_train = _compute_hi_array(y_train["temp_c"].values, y_train["rh"].values)
        weights = _make_sample_weight(pd.Series(y_hi_train))

        # Change 5: tail sample weight boost — _make_sample_weight gives 5x for >38°C and
        # 9x for >42°C, but nothing special at exactly >=40°C; ensure minimum 2.0 there.
        tail_mask = y_hi_train >= 40.0
        weights = weights.astype(float).copy()
        weights[tail_mask] = np.maximum(weights[tail_mask], 2.0)

        # Optuna hyperparameter search (use q50 of temp_c as proxy objective)
        self._best_params = self._tune(X_train, y_train["temp_c"], weights, X_val, y_val["temp_c"])

        # Train boosters: len(_QUANTILES) quantiles × 2 targets, multi-seed averaged
        for t_idx, target in enumerate(_TARGETS):
            for q_idx, alpha in enumerate(_QUANTILES):
                seed_boosters = []
                for seed in _SEEDS:
                    params = {
                        **_DEFAULT_PARAMS,
                        **self._best_params,
                        "alpha": alpha,
                        "seed": seed,
                    }
                    dtrain = lgb.Dataset(X_train, label=y_train[target], weight=weights)
                    dval = lgb.Dataset(X_val, label=y_val[target], reference=dtrain)
                    booster = lgb.train(
                        params,
                        dtrain,
                        num_boost_round=2000,
                        valid_sets=[dval],
                        callbacks=[
                            lgb.early_stopping(50, verbose=False),
                            lgb.log_evaluation(-1),
                        ],
                    )
                    seed_boosters.append(booster)
                self._boosters[target][q_idx] = seed_boosters
                logger.debug("Trained booster: target=%s q=%.2f seeds=%d", target, alpha, len(_SEEDS))

        # Mondrian CQR calibration on validation set (stratified by station × hour bin)
        bundle_val = self._predict_bundle(X_val)
        y_hi_val = _compute_hi_array(y_val["temp_c"].values, y_val["rh"].values)
        station_ids_val = np.full(len(X_val), station_id)
        angles_val = np.arctan2(X_val["local_hour_sin"].values, X_val["local_hour_cos"].values)
        local_hours_val = np.round(angles_val * 24 / (2 * np.pi)).astype(int) % 24
        self._calibrator = MondianCQRCalibrator()
        self._calibrator.fit(
            y_hi_val, bundle_val.hi_lower, bundle_val.hi_upper,
            station_ids=station_ids_val,
            local_hours=local_hours_val,
            alpha=0.10,
        )

        # Compute median PI width for low_confidence threshold
        lo_cal, hi_cal = self._calibrate(X_val, bundle_val.hi_lower, bundle_val.hi_upper)
        self._metadata["median_pi_width"] = float(np.median(hi_cal - lo_cal))

        # Danger gate (train on train+val combined)
        X_tv = pd.concat([X_train, X_val])
        y_hi_tv = _compute_hi_array(
            pd.concat([y_train["temp_c"], y_val["temp_c"]]).values,
            pd.concat([y_train["rh"], y_val["rh"]]).values,
        )
        self._gate = DangerGate()
        self._gate.fit(
            X_tv, pd.Series(y_hi_tv, index=X_tv.index),
            val_X=X_val,
            val_y_hi=pd.Series(y_hi_val, index=X_val.index),
        )

        # Test set metrics
        if len(X_test) > 0:
            bundle_test = self._predict_bundle_calibrated(X_test)
            y_hi_test = _compute_hi_array(y_test["temp_c"].values, y_test["rh"].values)
            mae = float(np.mean(np.abs(bundle_test.hi_mean - y_hi_test)))
            danger_mask = y_hi_test >= 40
            if danger_mask.sum() > 0:
                danger_recall = float(
                    ((bundle_test.hi_mean[danger_mask] >= 40).sum()) / danger_mask.sum()
                )
            else:
                danger_recall = float("nan")
            self._metadata.update({
                "mae": round(mae, 3),
                "danger_recall": round(danger_recall, 3) if not np.isnan(danger_recall) else None,
                "n_test": len(X_test),
                "n_train": n_train,
                "split_metadata": split.metadata,
            })
            logger.info("Test metrics: MAE=%.3f Danger recall=%.3f", mae, danger_recall)
        self._metadata["train_seconds"] = float(time.perf_counter() - started)

    def _tune(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        weights: np.ndarray,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> dict:
        """Optuna search over LightGBM hyperparameters (quantile q50, temp_c proxy)."""
        import lightgbm as lgb
        import optuna

        def objective(trial: optuna.Trial) -> float:
            params = {
                **_DEFAULT_PARAMS,
                "alpha": 0.50,
                "num_leaves": trial.suggest_int("num_leaves", 20, 200),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
                "seed": self.random_state,
            }
            dtrain = lgb.Dataset(X_train, label=y_train, weight=weights)
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
            booster = lgb.train(
                params, dtrain, num_boost_round=1000,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
            )
            return float(np.mean(np.abs(booster.predict(X_val) - y_val.values)))

        # Change 6: persistent Optuna study with warm-start
        _OPTUNA_STORAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
        study_name = f"lgbm_{self.target_kind}_h{self._horizon_h}"
        try:
            _storage = optuna.storages.RDBStorage(f"sqlite:///{_OPTUNA_STORAGE_PATH}")
        except Exception:
            _storage = None

        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
            storage=_storage,
            load_if_exists=True,
        )
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        logger.info("Optuna best: MAE=%.4f params=%s", study.best_value, study.best_params)
        self._metadata["n_trials_completed"] = sum(t.state.name == "COMPLETE" for t in study.trials)
        self._metadata["n_trials_pruned"] = sum(t.state.name == "PRUNED" for t in study.trials)
        return study.best_params

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _calibrate(
        self, X: pd.DataFrame, lo: np.ndarray, hi: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply calibrator — Mondrian CQR (per-stratum) or EnbPI fallback."""
        if self._calibrator is None:
            return lo, hi
        if isinstance(self._calibrator, MondianCQRCalibrator):
            station_ids = np.full(len(X), self._station_id)
            angles = np.arctan2(X["local_hour_sin"].values, X["local_hour_cos"].values)
            local_hours = np.round(angles * 24 / (2 * np.pi)).astype(int) % 24
            return self._calibrator.adjust(lo, hi, station_ids=station_ids, local_hours=local_hours)
        return self._calibrator.adjust(lo, hi)

    def _predict_th(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        """Predict q05/q50/q95/q97 for temp_c and rh. Returns dict with keys like 'temp_c_q05'."""
        # Change 3c: iterates _QUANTILES so q97 keys auto-appear
        X = self._align(X)
        result = {}
        for target in _TARGETS:
            for q_idx, alpha in enumerate(_QUANTILES):
                key = f"{target}_q{int(alpha * 100):02d}"
                preds = np.stack(
                    [b.predict(X) for b in self._boosters[target][q_idx]], axis=0
                ).mean(axis=0)
                result[key] = preds
        return result

    def _compose_hi(self, preds: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """4-corner Rothfusz composition for mean/lower/upper/q97 HI."""
        # Change 3d: return hi_q97 as fourth element
        t_lo, t_med, t_hi = preds["temp_c_q05"], preds["temp_c_q50"], preds["temp_c_q95"]
        r_lo, r_med, r_hi = preds["rh_q05"], preds["rh_q50"], preds["rh_q95"]

        hi_mean = _compute_hi_array(t_med, r_med)
        # Diagonal composition: vary T along its quantile axis, hold RH at median.
        # Full 2D corner min/max was too conservative (over-covered by ~3-4%);
        # diagonal keeps the physical T-driven range without RH amplification.
        hi_lower = _compute_hi_array(t_lo, r_med)
        hi_upper = _compute_hi_array(t_hi, r_med)

        # q97 composed from the q97 quantile of both heads
        t_q97 = preds["temp_c_q97"]
        r_q97 = preds["rh_q97"]
        hi_q97 = _compute_hi_array(t_q97, r_q97)

        return hi_mean, hi_lower, hi_upper, hi_q97

    def _predict_bundle_raw(self, X: pd.DataFrame) -> dict:
        """Raw (uncalibrated) dict — carries hi_q97 through for predict_with_pi."""
        # Change 3e: renamed from _predict_bundle (inner raw logic), returns plain dict
        preds = self._predict_th(X)
        hi_mean, hi_lower, hi_upper, hi_q97 = self._compose_hi(preds)
        danger_proba = self._gate.predict_proba(self._align(X)) if self._gate else None
        return {
            "hi_mean": hi_mean,
            "hi_lower": hi_lower,
            "hi_upper": hi_upper,
            "hi_q97": hi_q97,
            "temp_mean": preds["temp_c_q50"],
            "rh_mean": preds["rh_q50"],
            "danger_proba": danger_proba,
        }

    def _predict_bundle(self, X: pd.DataFrame) -> PredictionBundle:
        """Raw (uncalibrated) PredictionBundle — used internally during fit."""
        raw = self._predict_bundle_raw(self._align(X))
        return PredictionBundle(
            hi_mean=raw["hi_mean"],
            hi_lower=raw["hi_lower"],
            hi_upper=raw["hi_upper"],
            temp_mean=raw["temp_mean"],
            rh_mean=raw["rh_mean"],
            danger_proba=raw["danger_proba"],
        )

    def _predict_bundle_calibrated(self, X: pd.DataFrame) -> PredictionBundle:
        """Calibrated bundle (Mondrian CQR-adjusted)."""
        X = self._align(X)
        bundle = self._predict_bundle(X)
        lo_cal, hi_cal = self._calibrate(X, bundle.hi_lower, bundle.hi_upper)
        return PredictionBundle(
            hi_mean=bundle.hi_mean, hi_lower=lo_cal, hi_upper=hi_cal,
            temp_mean=bundle.temp_mean, rh_mean=bundle.rh_mean,
            danger_proba=bundle.danger_proba,
        )

    def _align(self, X: pd.DataFrame) -> pd.DataFrame:
        """Align X columns to training feature list, fill missing with medians."""
        out = X.reindex(columns=self.feature_list)
        for col, med in self._feature_medians.items():
            if col in out.columns:
                out[col] = out[col].fillna(med)
        return out

    # ------------------------------------------------------------------
    # BaseForecaster Protocol
    # ------------------------------------------------------------------

    def predict_with_pi(
        self, X: pd.DataFrame, alpha: float = 0.10
    ) -> PredictionBundle:
        """Return calibrated PredictionBundle for rows in X.

        DangerGate fires hi_mean escalation to 42.5°C when gate probability
        exceeds its threshold. Mondrian CQR calibration widens intervals in
        hot afternoon strata and tightens in cool regimes.
        """
        X_aligned = self._align(X)
        raw = self._predict_bundle_raw(X_aligned)
        hi_mean = raw["hi_mean"].copy()
        hi_lower = raw["hi_lower"].copy()
        hi_upper = raw["hi_upper"].copy()
        hi_q97 = raw["hi_q97"]
        danger_proba = raw["danger_proba"]

        # Mondrian CQR calibration
        hi_lower, hi_upper = self._calibrate(X_aligned, hi_lower, hi_upper)

        # Gate override — soft escalation when Danger probability fires.
        # Cap at 42.5°C (just above the alert boundary) instead of pushing to q97,
        # which caused ~13°C over-predictions on non-extreme hot-season samples.
        _DANGER_ALERT_THRESH = 42.5
        if self._gate is not None and danger_proba is not None:
            gate_mask = danger_proba >= self._gate.threshold
            if gate_mask.any():
                hi_mean[gate_mask] = np.maximum(hi_mean[gate_mask], _DANGER_ALERT_THRESH)

        return PredictionBundle(
            hi_mean=hi_mean,
            hi_lower=hi_lower,
            hi_upper=hi_upper,
            temp_mean=raw["temp_mean"],
            rh_mean=raw["rh_mean"],
            danger_proba=danger_proba,
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, dir_path: Path) -> None:
        """Persist all boosters + calibrator + gate + metadata."""
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        # Change 3f: iterates _QUANTILES — auto-handles 4 quantiles now
        for target in _TARGETS:
            for q_idx, alpha in enumerate(_QUANTILES):
                for s_idx, seed in enumerate(_SEEDS):
                    fname = f"{target}_q{int(alpha*100):02d}_s{seed}.txt"
                    booster = self._boosters[target][q_idx][s_idx]
                    if booster is not None:
                        booster.save_model(str(dir_path / fname))

        if self._calibrator:
            self._calibrator.save(dir_path / "calibrator.json")

        if self._gate:
            gate_dir = dir_path / "danger_gate"
            self._gate.save(gate_dir)

        meta = {
            "backend_name": self.backend_name,
            "target_kind": self.target_kind,
            "feature_list": self.feature_list,
            "feature_medians": self._feature_medians,
            "best_params": self._best_params,
            "station_id": self._station_id,
            "horizon_h": self._horizon_h,
            **self._metadata,
        }
        (dir_path / "bundle.json").write_text(json.dumps(meta, indent=2, default=str))
        logger.info("Saved LGBMForecaster to %s", dir_path)

    @classmethod
    def load(cls, dir_path: Path) -> "LGBMForecaster":
        """Reconstruct from a saved directory."""
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError("pip install lightgbm") from exc

        dir_path = Path(dir_path)
        bundle = json.loads((dir_path / "bundle.json").read_text())

        obj = cls()
        obj.feature_list = bundle["feature_list"]
        obj._feature_medians = bundle.get("feature_medians", {})
        obj._best_params = bundle.get("best_params", {})
        obj._station_id = bundle.get("station_id", "")
        obj._horizon_h = bundle.get("horizon_h", 24)
        obj._metadata = {k: v for k, v in bundle.items()
                         if k not in ("feature_list", "feature_medians", "best_params",
                                      "backend_name", "target_kind", "station_id", "horizon_h")}

        # Change 3f: load iterates _QUANTILES — handles q97 automatically
        for target in _TARGETS:
            for q_idx, alpha in enumerate(_QUANTILES):
                seed_boosters = []
                for seed in _SEEDS:
                    fname = dir_path / f"{target}_q{int(alpha*100):02d}_s{seed}.txt"
                    if fname.exists():
                        seed_boosters.append(lgb.Booster(model_file=str(fname)))
                obj._boosters[target][q_idx] = seed_boosters

        cal_path = dir_path / "calibrator.json"
        if cal_path.exists():
            cal_data = json.loads(cal_path.read_text())
            if "stratum_q" in cal_data:
                obj._calibrator = MondianCQRCalibrator.from_dict(cal_data)
            else:
                obj._calibrator = EnbPICalibrator.from_dict(cal_data)

        gate_dir = dir_path / "danger_gate"
        if gate_dir.exists() and (gate_dir / "gate_meta.json").exists():
            obj._gate = DangerGate.load(gate_dir)

        return obj


class LGBMDirectHIForecaster:
    """Direct heat-index LightGBM quantile forecaster."""

    backend_name: str = "lightgbm_hi_quantile"
    target_kind: Literal["hi", "th"] = "hi"

    def __init__(self, n_trials: int = 50, random_state: int = 42) -> None:
        self.n_trials = n_trials
        self.random_state = random_state
        self.feature_list: list[str] = []
        self._station_id: str = ""
        self._horizon_h: int = 24
        # Change 3g: added q97 slot
        self._boosters: dict[str, list] = {"q05": [], "q50": [], "q95": [], "q97": []}
        self._best_params: dict = {}
        self._feature_medians: dict = {}
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
            import lightgbm as lgb
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError as exc:
            raise ImportError("pip install lightgbm optuna") from exc

        started = time.perf_counter()
        if isinstance(y, pd.DataFrame):
            if "heat_index_c" not in y.columns:
                raise ValueError("Direct HI training requires a Series target or y['heat_index_c']")
            y = y["heat_index_c"]

        self._station_id = station_id
        self._horizon_h = horizon_h
        self.feature_list = list(X.columns)

        valid = y.notna()
        X, y = X[valid].copy(), y[valid].copy()
        split = split_xy(X, y, horizon_h=horizon_h)
        X_train, y_train = split.X_train, split.y_train
        X_val, y_val = split.X_val, split.y_val
        X_test, y_test = split.X_test, split.y_test
        self._feature_medians = fit_feature_medians(X_train)
        X_train = apply_feature_medians(X_train, self._feature_medians)
        X_val = apply_feature_medians(X_val, self._feature_medians)
        X_test = apply_feature_medians(X_test, self._feature_medians)

        from app.ml.forecast.train import _make_sample_weight
        weights = _make_sample_weight(y_train.to_numpy(dtype=float))
        self._best_params = self._tune(X_train, y_train, weights, X_val, y_val)

        # Change 3g: added q97 to quantile_map
        quantile_map = {"q05": 0.05, "q50": 0.50, "q95": 0.95, "q97": 0.97}
        for role, alpha in quantile_map.items():
            seed_boosters = []
            for seed in _SEEDS:
                params = {
                    **_DEFAULT_PARAMS,
                    **self._best_params,
                    "alpha": alpha,
                    "seed": seed,
                }
                dtrain = lgb.Dataset(X_train, label=y_train, weight=weights)
                dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
                booster = lgb.train(
                    params,
                    dtrain,
                    num_boost_round=2000,
                    valid_sets=[dval],
                    callbacks=[
                        lgb.early_stopping(50, verbose=False),
                        lgb.log_evaluation(-1),
                    ],
                )
                seed_boosters.append(booster)
            self._boosters[role] = seed_boosters

        if len(X_test) > 0:
            bundle = self.predict_with_pi(X_test)
            y_true = y_test.to_numpy(dtype=float)
            mae = float(np.mean(np.abs(bundle.hi_mean - y_true)))
            coverage = float(((y_true >= bundle.hi_lower) & (y_true <= bundle.hi_upper)).mean())
            self._metadata.update({
                "mae": round(mae, 3),
                "pi_coverage_90": round(coverage, 3),
                "n_test": len(X_test),
                "n_train": len(X_train),
                "split_metadata": split.metadata,
            })
        self._metadata["train_seconds"] = float(time.perf_counter() - started)

    def _tune(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        weights: np.ndarray,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> dict:
        import lightgbm as lgb
        import optuna

        def objective(trial: optuna.Trial) -> float:
            params = {
                **_DEFAULT_PARAMS,
                "alpha": 0.50,
                "num_leaves": trial.suggest_int("num_leaves", 20, 200),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
                "seed": self.random_state,
            }
            dtrain = lgb.Dataset(X_train, label=y_train, weight=weights)
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
            booster = lgb.train(
                params,
                dtrain,
                num_boost_round=1000,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
            )
            return float(np.mean(np.abs(booster.predict(X_val) - y_val.values)))

        # Change 6: persistent Optuna study with warm-start (LGBMDirectHIForecaster)
        _OPTUNA_STORAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
        study_name = f"lgbm_{self.target_kind}_h{self._horizon_h}"
        try:
            _storage = optuna.storages.RDBStorage(f"sqlite:///{_OPTUNA_STORAGE_PATH}")
        except Exception:
            _storage = None

        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
            storage=_storage,
            load_if_exists=True,
        )
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        self._metadata["n_trials_completed"] = sum(t.state.name == "COMPLETE" for t in study.trials)
        self._metadata["n_trials_pruned"] = sum(t.state.name == "PRUNED" for t in study.trials)
        return study.best_params

    def _align(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.reindex(columns=self.feature_list)
        for col, med in self._feature_medians.items():
            if col in out.columns:
                out[col] = out[col].fillna(med)
        return out.fillna(0.0)

    def _predict_role(self, X: pd.DataFrame, role: str) -> np.ndarray:
        # Change 3g: role can now be "q97" in addition to q05/q50/q95
        aligned = self._align(X)
        return np.stack([b.predict(aligned) for b in self._boosters[role]], axis=0).mean(axis=0)

    def predict_with_pi(self, X: pd.DataFrame, alpha: float = 0.10) -> PredictionBundle:
        return PredictionBundle(
            hi_mean=self._predict_role(X, "q50"),
            hi_lower=self._predict_role(X, "q05"),
            hi_upper=self._predict_role(X, "q95"),
        )

    def save(self, dir_path: Path) -> None:
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        # Change 3g: iterates all roles in _boosters including q97
        for role, boosters in self._boosters.items():
            for booster, seed in zip(boosters, _SEEDS):
                booster.save_model(str(dir_path / f"hi_{role}_s{seed}.txt"))
        meta = {
            "backend_name": self.backend_name,
            "target_kind": self.target_kind,
            "feature_list": self.feature_list,
            "feature_medians": self._feature_medians,
            "best_params": self._best_params,
            "station_id": self._station_id,
            "horizon_h": self._horizon_h,
            **self._metadata,
        }
        (dir_path / "bundle.json").write_text(json.dumps(meta, indent=2, default=str))

    @classmethod
    def load(cls, dir_path: Path) -> "LGBMDirectHIForecaster":
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError("pip install lightgbm") from exc

        dir_path = Path(dir_path)
        bundle = json.loads((dir_path / "bundle.json").read_text())
        obj = cls()
        obj.feature_list = bundle["feature_list"]
        obj._feature_medians = bundle.get("feature_medians", {})
        obj._best_params = bundle.get("best_params", {})
        obj._station_id = bundle.get("station_id", "")
        obj._horizon_h = bundle.get("horizon_h", 24)
        obj._metadata = {k: v for k, v in bundle.items()
                         if k not in ("feature_list", "feature_medians", "best_params",
                                      "backend_name", "target_kind", "station_id", "horizon_h")}
        # Change 3g: load iterates all keys in _boosters (includes q97) — no hardcoded list
        for role in obj._boosters:
            boosters = []
            for seed in _SEEDS:
                path = dir_path / f"hi_{role}_s{seed}.txt"
                if path.exists():
                    boosters.append(lgb.Booster(model_file=str(path)))
            obj._boosters[role] = boosters
        return obj
