"""LightGBM two-head quantile forecaster for HeatShield AI v3.

Predicts (temp_c, rh) jointly via 6 quantile boosters, composes to heat_index_c
via the Rothfusz formula, then applies EnbPI conformal calibration for valid PIs.
An optional DangerGate classifier routes samples ≥ 40°C through a tail regressor.
"""
from __future__ import annotations

import json
import logging
import gc
import os
import time
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import TimeSeriesSplit

from app.ml.forecast.base import BaseForecaster, PredictionBundle
from app.ml.forecast.conformal import EnbPICalibrator, MondianCQRCalibrator
from app.ml.forecast.danger_gate import DangerGate
from app.ml.forecast._optuna_utils import feature_signature, forecast_model_root, write_heartbeat
from app.ml.forecast.splitting import apply_feature_medians, fit_feature_medians, split_xy, split_xy_4way

logger = logging.getLogger(__name__)

_QUANTILES = [0.05, 0.50, 0.95, 0.97]
_TARGETS = ["temp_c", "rh"]
_SEEDS = [42, 123, 7]
_CV_SPLITS = 3
_CV_GAP_HOURS = 72
_LGBM_DEVICE_SUPPORT: dict[str, bool] = {}
_HW_DEVICE: str | None = None

# Detect best LightGBM device.
# Hardware probe runs once at import; env overrides are re-read on every call.
# Priority: LGBM_DEVICE env > HEATSHIELD_FORCE_CPU env > CUDA > OpenCL GPU > CPU

def _probe_lgbm_hardware() -> str:
    """Run the slow lgb.train probes once to discover hardware capabilities."""
    import lightgbm as lgb
    # Each probe gets a fresh Dataset — reusing one Dataset across probes risks
    # hitting inconsistent internal state after a failed lgb.train call.
    try:
        lgb.train({"objective": "regression", "device_type": "cuda", "verbose": -1,
                   "num_leaves": 4},
                  lgb.Dataset(np.zeros((4, 2)), np.zeros(4)),
                  num_boost_round=1)
        logger.info("LightGBM CUDA detected — training will use CUDA")
        _LGBM_DEVICE_SUPPORT["cuda"] = True
        _LGBM_DEVICE_SUPPORT["gpu"] = True  # if CUDA works, OpenCL certainly does
        return "cuda"
    except Exception:
        _LGBM_DEVICE_SUPPORT["cuda"] = False
    try:
        lgb.train({"objective": "regression", "device_type": "gpu", "verbose": -1,
                   "num_leaves": 4},
                  lgb.Dataset(np.zeros((4, 2)), np.zeros(4)),
                  num_boost_round=1)
        logger.info("LightGBM GPU (OpenCL) detected — training will use GPU")
        _LGBM_DEVICE_SUPPORT["gpu"] = True
        return "gpu"
    except Exception:
        _LGBM_DEVICE_SUPPORT["gpu"] = False
        return "cpu"


def _supports_lgbm_device(device_type: str) -> bool:
    """Return whether a specific LightGBM device type is usable."""
    key = device_type.lower().strip()
    if key in _LGBM_DEVICE_SUPPORT:
        return _LGBM_DEVICE_SUPPORT[key]
    try:
        import lightgbm as lgb
        lgb.train(
            {"objective": "regression", "device_type": key, "verbose": -1, "num_leaves": 4},
            lgb.Dataset(np.zeros((8, 2)), np.zeros(8)),
            num_boost_round=1,
        )
        _LGBM_DEVICE_SUPPORT[key] = True
    except Exception:
        _LGBM_DEVICE_SUPPORT[key] = False
    return _LGBM_DEVICE_SUPPORT[key]


def _detect_device() -> str:
    """Return best LightGBM device, re-reading env vars on every call.

    Hardware capability is probed once at import (_HW_DEVICE); env overrides
    (LGBM_DEVICE, HEATSHIELD_FORCE_CPU) are re-read each call so test fixtures
    and shell changes made after import are respected.
    """
    global _HW_DEVICE
    import os
    env = os.environ.get("LGBM_DEVICE", "").lower()
    if env:
        if _supports_lgbm_device(env):
            logger.info("LightGBM device forced via LGBM_DEVICE=%s", env)
            return env
        logger.warning("LGBM_DEVICE=%s is not supported by this LightGBM build. Falling back to auto-detect.", env)
    if os.environ.get("HEATSHIELD_FORCE_CPU") == "1":
        logger.info("LightGBM forced to CPU (HEATSHIELD_FORCE_CPU=1)")
        return "cpu"
    if _HW_DEVICE is None:
        _HW_DEVICE = _probe_lgbm_hardware()
    return _HW_DEVICE


_DEVICE_TYPE = "auto"  # kept for external references; actual device is resolved lazily

# Default LightGBM params (overridden by Optuna)
_DEFAULT_PARAMS = {
    "objective": "quantile",
    "metric": "quantile",
    "verbose": -1,
}

# Change 6: persistent Optuna study path
def _optuna_storage_path() -> Path:
    return forecast_model_root() / "optuna_studies.db"


def _lgbm_thread_count(device: str, parallel_jobs: int = 1) -> int | None:
    """Avoid CPU oversubscription when multiple boosters train in parallel."""
    if device != "cpu":
        return None
    if os.environ.get("LGBM_N_JOBS"):
        return max(1, int(os.environ["LGBM_N_JOBS"]))
    cores = os.cpu_count() or 1
    return max(1, cores // max(1, parallel_jobs))


def _lgbm_device_params(device: str, parallel_jobs: int = 1) -> dict:
    params = {"device_type": device}
    n_jobs = _lgbm_thread_count(device, parallel_jobs=parallel_jobs)
    if n_jobs is not None:
        params["n_jobs"] = n_jobs
    return params


def _remaining_trials(study, requested_trials: int) -> int:
    """Run Optuna up to requested complete trials, not requested extra trials."""
    complete = sum(t.state.name == "COMPLETE" for t in study.trials)
    return max(0, requested_trials - complete)


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


def _compute_dense_weights(
    y_hi: np.ndarray,
    *,
    alpha: float = 0.8,
    min_weight: float = 0.25,
    max_weight: float = 6.0,
) -> np.ndarray:
    """Compute density-based sample weights with clipping.

    Lower-density (rarer) samples get higher weights. Weights are normalized
    to mean 1 and clipped to avoid gradient explosions.
    """
    y = np.asarray(y_hi, dtype=float)
    if len(y) < 20 or np.nanstd(y) < 1e-6:
        return np.ones(len(y), dtype=float)
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(y)
        density = np.clip(kde(y), 1e-9, None)
        weight = 1.0 / (density ** float(alpha))
        weight = weight / max(float(np.mean(weight)), 1e-9)
        return np.clip(weight, min_weight, max_weight)
    except Exception:
        return np.ones(len(y), dtype=float)


def _expanding_window_splits(
    n_rows: int,
    *,
    gap: int,
    n_splits: int = _CV_SPLITS,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return expanding-window folds with purge gap."""
    if n_rows <= gap + 48:
        return []
    # Ensure fold count remains valid for short series.
    split_count = max(2, min(n_splits, (n_rows // max(gap + 24, 48))))
    splitter = TimeSeriesSplit(n_splits=split_count, gap=gap)
    return list(splitter.split(np.arange(n_rows)))


def _train_single_booster(
    target: str,
    alpha: float,
    seed: int,
    X_train: pd.DataFrame,
    y_target: np.ndarray,
    X_val: pd.DataFrame,
    y_val_target: np.ndarray,
    weights: np.ndarray,
    best_params: dict,
    parallel_jobs: int,
):
    """Train a single LightGBM booster for one (target, quantile, seed) combination.

    This function is defined at module level so joblib can pickle it.
    """
    import lightgbm as lgb

    params = {
        **_DEFAULT_PARAMS,
        **best_params,
        **_lgbm_device_params(_detect_device(), parallel_jobs=parallel_jobs),
        "alpha": alpha,
        "seed": seed,
    }
    dtrain = lgb.Dataset(X_train, label=y_target, weight=weights)
    dval = lgb.Dataset(X_val, label=y_val_target, reference=dtrain)
    return lgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )


def _make_lgb_dataset_pair(X_train: pd.DataFrame, X_val: pd.DataFrame):
    """Construct reusable LightGBM bins for serial booster training."""
    import lightgbm as lgb

    dtrain = lgb.Dataset(X_train, free_raw_data=False)
    dtrain.construct()
    dval = lgb.Dataset(X_val, reference=dtrain, free_raw_data=False)
    dval.construct()
    return dtrain, dval


def _train_cached_lgb_booster(
    dtrain,
    dval,
    y_train: np.ndarray,
    y_val: np.ndarray,
    weights: np.ndarray,
    params: dict,
):
    import lightgbm as lgb

    dtrain.set_label(y_train)
    dtrain.set_weight(weights)
    dval.set_label(y_val)
    return lgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )


def _build_monotone_constraints(feature_list: list[str]) -> list[int]:
    """Return +1 for lag columns of warm-weather variables, 0 otherwise.
    Enforces: higher lag value → higher forecast (physics prior).
    """
    WARM_PREFIXES = (
        "temp_c_lag", "heat_index_c_lag", "dewpoint_c_lag",
        "vpd_kpa_lag", "wbgt_stull_c_lag", "rh_lag",
    )
    return [1 if any(col.startswith(p) for p in WARM_PREFIXES) else 0 for col in feature_list]


def _refit_single_booster(
    target: str,
    alpha: float,
    seed: int,
    X_tv: pd.DataFrame,
    y_tv_target: np.ndarray,
    weights_tv: np.ndarray,
    best_params: dict,
    best_iter: int,
    parallel_jobs: int,
):
    """Refit a single booster on train+val_es at best_iter (no early-stop)."""
    import lightgbm as lgb

    params = {
        **_DEFAULT_PARAMS,
        **best_params,
        **_lgbm_device_params(_detect_device(), parallel_jobs=parallel_jobs),
        "alpha": alpha,
        "seed": seed,
    }
    constraints = _build_monotone_constraints(list(X_tv.columns))
    if any(c != 0 for c in constraints):
        params["monotone_constraints"] = constraints
    dtrain = lgb.Dataset(X_tv, label=y_tv_target, weight=weights_tv)
    return lgb.train(params, dtrain, num_boost_round=best_iter)


class LGBMForecaster:
    """Two-head LightGBM quantile forecaster satisfying BaseForecaster Protocol."""

    backend_name: str = "lightgbm_quantile"
    target_kind: Literal["hi", "th"]

    def __init__(
        self,
        n_trials: int = 50,
        random_state: int = 42,
        gate_backend: str = "lightgbm",
    ) -> None:
        self.n_trials = n_trials
        self.random_state = random_state
        self._gate_backend = gate_backend
        self.target_kind: str = "th"
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

        Split: train(70%) / val_es(7.5%) / val_cal(7.5%) / test(15%).
        Boosters are early-stopped on val_es, refitted on train+val_es at best_iter,
        then cal/gate sweep on val_cal — so test metrics and coverage are unbiased.
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

        valid = y.notna().all(axis=1)
        X, y = X[valid].copy(), y[valid].copy()

        split = split_xy_4way(X, y, horizon_h=horizon_h)
        X_train, y_train = split.X_train, split.y_train
        X_val_es, y_val_es = split.X_val_es, split.y_val_es
        X_val_cal, y_val_cal = split.X_val_cal, split.y_val_cal
        X_test, y_test = split.X_test, split.y_test
        n = len(X)
        n_train = len(X_train)
        n_val_es = len(X_val_es)
        n_val_cal = len(X_val_cal)
        self._feature_medians = fit_feature_medians(X_train)
        X_train = apply_feature_medians(X_train, self._feature_medians)
        X_val_es = apply_feature_medians(X_val_es, self._feature_medians)
        X_val_cal = apply_feature_medians(X_val_cal, self._feature_medians)
        X_test = apply_feature_medians(X_test, self._feature_medians)

        logger.info(
            "LGBMForecaster fit: station=%s h=%d rows=%d (train=%d val_es=%d val_cal=%d test=%d)",
            station_id, horizon_h, n, n_train, n_val_es, n_val_cal, len(X_test),
        )

        # Optuna tuning uses all non-test data so CV folds see maximum history.
        X_tune = pd.concat([X_train, X_val_es, X_val_cal], ignore_index=True)
        y_temp_tune = pd.concat(
            [y_train["temp_c"], y_val_es["temp_c"], y_val_cal["temp_c"]], ignore_index=True
        )
        y_rh_tune = pd.concat(
            [y_train["rh"], y_val_es["rh"], y_val_cal["rh"]], ignore_index=True
        )
        y_hi_tune = _compute_hi_array(y_temp_tune.values, y_rh_tune.values)

        dense_alpha_init = 0.8
        dense_weights = _compute_dense_weights(y_hi_tune, alpha=dense_alpha_init)
        self._best_params = self._tune(
            X_tune,
            y_temp_tune,
            y_hi_for_weights=y_hi_tune,
            dense_weights=dense_weights,
            dense_alpha_init=dense_alpha_init,
            horizon_h=horizon_h,
        )
        dense_alpha = float(self._best_params.pop("dense_alpha", dense_alpha_init))
        self._metadata["dense_alpha"] = dense_alpha
        del X_tune, y_temp_tune, y_rh_tune, y_hi_tune, dense_weights
        gc.collect()

        y_hi_train = _compute_hi_array(y_train["temp_c"].values, y_train["rh"].values)
        weights_train = _compute_dense_weights(y_hi_train, alpha=dense_alpha)

        device = _detect_device()
        is_gpu = "gpu" in device.lower() or "cuda" in device.lower()

        # Phase 1: pilot booster with early stopping on val_es to find best_iter.
        pilot_params = {
            **_DEFAULT_PARAMS,
            **self._best_params,
            **_lgbm_device_params(device),
            "alpha": 0.50,
            "seed": self.random_state,
        }
        pilot_booster = lgb.train(
            pilot_params,
            lgb.Dataset(X_train, label=y_train["temp_c"].values, weight=weights_train),
            num_boost_round=2000,
            valid_sets=[lgb.Dataset(X_val_es, label=y_val_es["temp_c"].values)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )
        best_iter = max(pilot_booster.best_iteration, 1)
        logger.info("Pilot early-stop: best_iter=%d", best_iter)
        del pilot_booster
        gc.collect()

        # Phase 2: refit ALL boosters on train+val_es at best_iter (no early-stop).
        X_tv = pd.concat([X_train, X_val_es], ignore_index=True)
        y_tv = pd.concat([y_train, y_val_es], ignore_index=True)
        y_hi_tv = _compute_hi_array(y_tv["temp_c"].values, y_tv["rh"].values)
        weights_tv = _compute_dense_weights(y_hi_tv, alpha=dense_alpha)

        if is_gpu:
            for t_idx, target in enumerate(_TARGETS):
                for q_idx, alpha in enumerate(_QUANTILES):
                    seed_boosters = []
                    for seed in _SEEDS:
                        params = {
                            **_DEFAULT_PARAMS,
                            **self._best_params,
                            **_lgbm_device_params(_detect_device()),
                            "alpha": alpha,
                            "seed": seed,
                        }
                        constraints = _build_monotone_constraints(list(X_tv.columns))
                        if any(c != 0 for c in constraints):
                            params["monotone_constraints"] = constraints
                        booster = lgb.train(
                            params,
                            lgb.Dataset(X_tv, label=y_tv[target].values, weight=weights_tv),
                            num_boost_round=best_iter,
                        )
                        seed_boosters.append(booster)
                    self._boosters[target][q_idx] = seed_boosters
                    logger.debug("Refit booster: target=%s q=%.2f seeds=%d", target, alpha, len(_SEEDS))
        else:
            n_jobs = max(1, min(int(os.environ.get("LGBM_PARALLEL_BOOSTERS", "3")), os.cpu_count() or 1))
            jobs = [
                (target, q_idx, alpha, seed)
                for t_idx, target in enumerate(_TARGETS)
                for q_idx, alpha in enumerate(_QUANTILES)
                for seed in _SEEDS
            ]
            results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_refit_single_booster)(
                    target, alpha, seed,
                    X_tv, y_tv[target].values,
                    weights_tv, self._best_params,
                    best_iter, n_jobs,
                )
                for target, q_idx, alpha, seed in jobs
            )
            result_idx = 0
            for t_idx, target in enumerate(_TARGETS):
                for q_idx, alpha in enumerate(_QUANTILES):
                    seed_boosters = []
                    for seed in _SEEDS:
                        seed_boosters.append(results[result_idx])
                        result_idx += 1
                    self._boosters[target][q_idx] = seed_boosters
                    logger.debug("Refit booster: target=%s q=%.2f seeds=%d", target, alpha, len(_SEEDS))

        # Danger gate: fit on train+val_es, sweep thresholds on val_cal.
        y_hi_tv_series = pd.Series(y_hi_tv, index=X_tv.index)
        y_hi_val_cal = _compute_hi_array(y_val_cal["temp_c"].values, y_val_cal["rh"].values)
        self._gate = DangerGate(gate_backend=self._gate_backend)
        self._gate.fit(
            X_tv, y_hi_tv_series,
            val_X=X_val_cal,
            val_y_hi=pd.Series(y_hi_val_cal, index=X_val_cal.index),
        )
        self._metadata["gate_warning_threshold"] = float(self._gate.warning_threshold)
        self._metadata["gate_danger_threshold"] = float(self._gate.danger_threshold)

        # Mondrian CQR calibration on val_cal — boosters and gate never saw this split.
        bundle_val_cal = self._predict_bundle(X_val_cal)
        station_ids_cal = np.full(len(X_val_cal), station_id)
        angles_cal = np.arctan2(X_val_cal["local_hour_sin"].values, X_val_cal["local_hour_cos"].values)
        local_hours_cal = np.round(angles_cal * 24 / (2 * np.pi)).astype(int) % 24
        danger_tier_cal = self._gate.predict_tier(X_val_cal) if self._gate else np.zeros(len(X_val_cal), dtype=int)
        self._calibrator = MondianCQRCalibrator()
        self._calibrator.fit(
            y_hi_val_cal,
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
        X_tune: pd.DataFrame,
        y_target: pd.Series,
        *,
        y_hi_for_weights: np.ndarray,
        dense_weights: np.ndarray | None = None,
        dense_alpha_init: float = 0.8,
        horizon_h: int,
    ) -> dict:
        """Optuna search over LightGBM hyperparameters with expanding-window CV."""
        import lightgbm as lgb
        import optuna

        # Detect device and set max_bin choices once (outside objective to avoid dynamic categorical space)
        dev = _detect_device()
        _max_bin_choices = [127, 255] if dev == "gpu" else [127, 255, 511]

        def objective(trial: optuna.Trial) -> float:
            dense_alpha = trial.suggest_float("dense_alpha", 0.3, 1.5)
            params = {
                **_DEFAULT_PARAMS,
                **_lgbm_device_params(dev),
                "alpha": 0.50,
                "num_leaves": trial.suggest_int("num_leaves", 31, 255),
                "max_depth": trial.suggest_int("max_depth", 4, 12),
                "max_bin": trial.suggest_categorical("max_bin", _max_bin_choices),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 0, 7),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
                "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
                "seed": self.random_state,
            }
            folds = _expanding_window_splits(
                len(X_tune),
                gap=max(horizon_h, _CV_GAP_HOURS),
                n_splits=_CV_SPLITS,
            )
            if not folds:
                return float("inf")

            fold_mae: list[float] = []
            for k, (tr_idx, va_idx) in enumerate(folds):
                X_train = X_tune.iloc[tr_idx]
                y_train = y_target.iloc[tr_idx]
                X_val = X_tune.iloc[va_idx]
                y_val = y_target.iloc[va_idx]
                if dense_weights is not None:
                    weights = dense_weights[tr_idx]
                else:
                    weights = _compute_dense_weights(y_hi_for_weights[tr_idx], alpha=dense_alpha)
                dtrain = lgb.Dataset(X_train, label=y_train, weight=weights)
                dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
                booster = lgb.train(
                    params,
                    dtrain,
                    num_boost_round=500,
                    valid_sets=[dval],
                    callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(-1)],
                )
                preds = booster.predict(X_val)
                fold_mae.append(float(np.mean(np.abs(preds - y_val.values))))
                trial.report(float(np.mean(fold_mae)), step=k)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return float(np.mean(fold_mae))

        # Change 6: persistent Optuna study with warm-start
        # station_id is included so parallel --workers runs don't share TPE state
        _feat_sig = feature_signature(
            list(X_tune.columns),
            extra="lags=[1,3,6,12,24]|roll=[3,6,24]",
        )
        storage_path = _optuna_storage_path()
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        study_name = f"lgbm_{self._station_id}_{self.target_kind}_h{self._horizon_h}_{_feat_sig}"
        try:
            _storage = optuna.storages.RDBStorage(f"sqlite:///{storage_path.as_posix()}?check_same_thread=False")
        except Exception:
            _storage = None

        try:
            study = optuna.create_study(
                study_name=study_name,
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=self.random_state),
                pruner=optuna.pruners.HyperbandPruner(min_resource=10, max_resource=1000, reduction_factor=3),
                storage=_storage,
                load_if_exists=True,
            )
        except Exception as exc:
            logger.warning("Optuna persistent storage unavailable (%s). Falling back to in-memory study.", exc)
            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=self.random_state),
                pruner=optuna.pruners.HyperbandPruner(min_resource=10, max_resource=1000, reduction_factor=3),
            )
        run_id = os.environ.get("HEATSHIELD_RUN_ID")
        slot = f"{self._station_id}:h{self._horizon_h}"

        def _heartbeat_cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            write_heartbeat(
                run_id,
                current_slot=slot,
                phase="tune",
                trials_done=sum(t.state.name == "COMPLETE" for t in study.trials),
            )

        remaining_trials = _remaining_trials(study, self.n_trials)
        if remaining_trials > 0:
            study.optimize(objective, n_trials=remaining_trials, show_progress_bar=False, callbacks=[_heartbeat_cb])
        else:
            logger.info("Optuna study already has %d complete trials; reusing best params", self.n_trials)
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
            danger_tier = self._gate.predict_tier(X) if self._gate else np.zeros(len(X), dtype=int)
            return self._calibrator.adjust(
                lo,
                hi,
                station_ids=station_ids,
                local_hours=local_hours,
                danger_tiers=danger_tier,
            )
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
        return out.fillna(0.0)

    # ------------------------------------------------------------------
    # BaseForecaster Protocol
    # ------------------------------------------------------------------

    def predict_with_pi(
        self, X: pd.DataFrame, alpha: float = 0.10
    ) -> PredictionBundle:
        """Return calibrated PredictionBundle for rows in X.

        DangerGate predicts 3 tiers (safe/warning/danger). We apply tier-aware
        escalation on hi_mean and keep calibrated intervals from Mondrian CQR.
        """
        X_aligned = self._align(X)
        raw = self._predict_bundle_raw(X_aligned)
        hi_mean = raw["hi_mean"].copy()
        hi_lower = raw["hi_lower"].copy()
        hi_upper = raw["hi_upper"].copy()
        hi_q97 = raw["hi_q97"]
        danger_proba = raw["danger_proba"]
        danger_tier = self._gate.predict_tier(X_aligned) if self._gate else np.zeros(len(X_aligned), dtype=int)

        # Mondrian CQR calibration
        hi_lower, hi_upper = self._calibrate(X_aligned, hi_lower, hi_upper)

        # Tier-aware gate override.
        warning_mask = danger_tier == 1
        danger_mask = danger_tier == 2
        if warning_mask.any():
            hi_mean[warning_mask] = np.maximum(hi_mean[warning_mask], 0.8 * hi_mean[warning_mask] + 0.2 * hi_q97[warning_mask])
        if danger_mask.any():
            hi_mean[danger_mask] = np.maximum(hi_mean[danger_mask], hi_q97[danger_mask])
            hi_upper[danger_mask] = np.maximum(hi_upper[danger_mask], hi_q97[danger_mask])

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
        obj.target_kind = bundle.get("target_kind", "th")
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

    def __init__(self, n_trials: int = 50, random_state: int = 42, gate_backend: str = "lightgbm") -> None:
        self.n_trials = n_trials
        self.random_state = random_state
        self._gate_backend = gate_backend  # accepted for API symmetry with LGBMForecaster; not used in fit
        self.feature_list: list[str] = []
        self._station_id: str = ""
        self._horizon_h: int = 24
        # Change 3g: added q97 slot
        self._boosters: dict[str, list] = {"q05": [], "q50": [], "q95": [], "q97": []}
        self._best_params: dict = {}
        self._feature_medians: dict = {}
        self._metadata: dict = {}
        self._calibrator: EnbPICalibrator | None = None

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

        from app.ml.forecast.train import _make_sample_weight
        weights = _make_sample_weight(y_train.to_numpy(dtype=float))
        self._best_params = self._tune(X_train, y_train, weights, X_val_es, y_val_es)

        # Phase 1: pilot with early-stop on val_es → best_iter
        import lightgbm as lgb
        pilot_params = {
            **_DEFAULT_PARAMS,
            **self._best_params,
            **_lgbm_device_params(_detect_device()),
            "alpha": 0.50,
            "seed": self.random_state,
        }
        pilot_booster = lgb.train(
            pilot_params,
            lgb.Dataset(X_train, label=y_train.to_numpy(dtype=float), weight=weights),
            num_boost_round=2000,
            valid_sets=[lgb.Dataset(X_val_es, label=y_val_es.to_numpy(dtype=float))],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )
        best_iter = max(pilot_booster.best_iteration, 1)
        logger.info("DirectHI pilot early-stop: best_iter=%d", best_iter)
        del pilot_booster
        gc.collect()

        # Phase 2: refit ALL boosters on train+val_es at best_iter (no early-stop).
        X_tv = pd.concat([X_train, X_val_es], ignore_index=True)
        y_tv = pd.concat([y_train, y_val_es], ignore_index=True)
        weights_tv = _make_sample_weight(y_tv.to_numpy(dtype=float))

        quantile_map = {"q05": 0.05, "q50": 0.50, "q95": 0.95, "q97": 0.97}
        for role, alpha in quantile_map.items():
            seed_boosters = []
            for seed in _SEEDS:
                params = {
                    **_DEFAULT_PARAMS,
                    **self._best_params,
                    **_lgbm_device_params(_detect_device()),
                    "alpha": alpha,
                    "seed": seed,
                }
                constraints = _build_monotone_constraints(list(X_tv.columns))
                if any(c != 0 for c in constraints):
                    params["monotone_constraints"] = constraints
                booster = lgb.train(
                    params,
                    lgb.Dataset(X_tv, label=y_tv.to_numpy(dtype=float), weight=weights_tv),
                    num_boost_round=best_iter,
                )
                seed_boosters.append(booster)
            self._boosters[role] = seed_boosters

        # Mondrian CQR calibration on val_cal — boosters never saw this split.
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
            bundle = self._predict_bundle_calibrated(X_test)
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
            dev = _detect_device()
            _max_bin_choices = [127, 255] if dev == "gpu" else [127, 255, 511]
            params = {
                **_DEFAULT_PARAMS,
                **_lgbm_device_params(dev),
                "alpha": 0.50,
                "num_leaves": trial.suggest_int("num_leaves", 31, 255),
                "max_depth": trial.suggest_int("max_depth", 4, 12),
                "max_bin": trial.suggest_categorical("max_bin", _max_bin_choices),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 0, 7),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
                "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
                "seed": self.random_state,
            }

            _should_prune = [False]

            def _pruning_cb(env) -> None:
                if not env.evaluation_result_list:
                    return
                val_loss = env.evaluation_result_list[0][2]
                trial.report(float(val_loss), step=env.iteration)
                if trial.should_prune():
                    _should_prune[0] = True

            dtrain = lgb.Dataset(X_train, label=y_train, weight=weights)
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
            booster = lgb.train(
                params,
                dtrain,
                num_boost_round=500,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(-1), _pruning_cb],
            )
            if _should_prune[0]:
                raise optuna.TrialPruned()
            return float(np.mean(np.abs(booster.predict(X_val) - y_val.values)))

        # Change 6: persistent Optuna study with warm-start (LGBMDirectHIForecaster)
        # station_id is included so parallel --workers runs don't share TPE state
        _feat_sig = feature_signature(
            list(X_train.columns),
            extra="lags=[1,3,6,12,24]|roll=[3,6,24]",
        )
        storage_path = _optuna_storage_path()
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        study_name = f"lgbm_{self._station_id}_{self.target_kind}_h{self._horizon_h}_{_feat_sig}"
        try:
            _storage = optuna.storages.RDBStorage(f"sqlite:///{storage_path.as_posix()}?check_same_thread=False")
        except Exception:
            _storage = None

        try:
            study = optuna.create_study(
                study_name=study_name,
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=self.random_state),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
                storage=_storage,
                load_if_exists=True,
            )
        except Exception as exc:
            logger.warning("Optuna persistent storage unavailable (%s). Falling back to in-memory study.", exc)
            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=self.random_state),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
            )
        run_id = os.environ.get("HEATSHIELD_RUN_ID")
        slot = f"{self._station_id}:h{self._horizon_h}"

        def _heartbeat_cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            write_heartbeat(
                run_id,
                current_slot=slot,
                phase="tune",
                trials_done=sum(t.state.name == "COMPLETE" for t in study.trials),
            )

        remaining_trials = _remaining_trials(study, self.n_trials)
        if remaining_trials > 0:
            study.optimize(objective, n_trials=remaining_trials, show_progress_bar=False, callbacks=[_heartbeat_cb])
        else:
            logger.info("Optuna study already has %d complete trials; reusing best params", self.n_trials)
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

    def _calibrate(
        self, X: pd.DataFrame, lo: np.ndarray, hi: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply Mondrian CQR calibrator to prediction intervals."""
        if self._calibrator is None:
            return lo, hi
        station_ids = np.full(len(X), self._station_id)
        angles = np.arctan2(X["local_hour_sin"].values, X["local_hour_cos"].values)
        local_hours = np.round(angles * 24 / (2 * np.pi)).astype(int) % 24
        return self._calibrator.adjust(
            lo, hi, station_ids=station_ids, local_hours=local_hours,
        )

    def _predict_bundle_calibrated(
        self, X: pd.DataFrame, alpha: float = 0.10
    ) -> PredictionBundle:
        """Predict with calibrated prediction intervals."""
        raw = self.predict_with_pi(X, alpha)
        lo, hi = self._calibrate(X, raw.hi_lower, raw.hi_upper)
        return PredictionBundle(
            hi_mean=raw.hi_mean, hi_lower=lo, hi_upper=hi,
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

        if self._calibrator:
            self._calibrator.save(dir_path / "calibrator.json")

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
        obj.target_kind = bundle.get("target_kind", "hi")
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

        cal_path = dir_path / "calibrator.json"
        if cal_path.exists():
            cal_data = json.loads(cal_path.read_text())
            if "stratum_q" in cal_data:
                obj._calibrator = MondianCQRCalibrator.from_dict(cal_data)
            else:
                obj._calibrator = EnbPICalibrator.from_dict(cal_data)

        return obj
