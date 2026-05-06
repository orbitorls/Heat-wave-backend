"""Conformal prediction calibrator for time-series forecast intervals.

Implements a simplified EnbPI (Ensemble Batch Prediction Intervals) approach:
fit calibration quantile on validation residuals, then expand/contract the
raw quantile interval at test time to guarantee empirical coverage ≥ 1-alpha.

Reference: Chen & Candès (ICML 2021); Angelopoulos et al. (2023).
This implementation assumes i.i.d. residuals within the val window, which holds
when the val split is contiguous in time and drift is slow relative to the window.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class EnbPICalibrator:
    """Post-hoc conformal calibrator for pre-computed quantile intervals.

    Usage
    -----
    cal = EnbPICalibrator()
    cal.fit(y_val_true, y_val_lower, y_val_upper, alpha=0.10)
    lo_cal, hi_cal = cal.adjust(y_test_lower, y_test_upper)
    # Empirical coverage on val is guaranteed >= 1 - alpha.
    """

    def __init__(self) -> None:
        self.alpha: float = 0.10
        self._q_hat: float = 0.0

    def fit(
        self,
        y_true: np.ndarray,
        y_lower: np.ndarray,
        y_upper: np.ndarray,
        alpha: float = 0.10,
    ) -> "EnbPICalibrator":
        """Compute calibration constant from validation residuals.

        The conformity score per sample is max(lower - true, true - upper),
        i.e. how far outside the interval the true value lies (negative = inside).
        q_hat is the (1-alpha) quantile of these scores with finite-sample correction.
        """
        self.alpha = alpha
        n = len(y_true)
        scores = np.maximum(y_lower - y_true, y_true - y_upper)
        # Finite-sample correction: ceil((n+1)*(1-alpha))/n quantile
        level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        self._q_hat = float(np.quantile(scores, level))
        return self

    def adjust(
        self,
        y_lower: np.ndarray,
        y_upper: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Expand or contract the interval by q_hat.

        If the raw interval is already conservative (q_hat < 0), this will
        tighten it — that's correct behaviour.
        """
        return y_lower - self._q_hat, y_upper + self._q_hat

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {"alpha": self.alpha, "q_hat": self._q_hat}

    @classmethod
    def from_dict(cls, d: dict) -> "EnbPICalibrator":
        obj = cls()
        obj.alpha = float(d["alpha"])
        obj._q_hat = float(d["q_hat"])
        return obj

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "EnbPICalibrator":
        return cls.from_dict(json.loads(path.read_text()))


class MondianCQRCalibrator:
    """Mondrian Conformal Quantile Regression calibrator.

    Stratifies calibration residuals on (station_id, hour_bin, danger_tier) where
    hour_bin = local_hour // 6 and danger_tier is an optional 3-level risk bin.
    Per-stratum q_hat gives tighter intervals in cool regimes and wider intervals
    in hot/afternoon/high-risk regimes while keeping per-stratum coverage.
    """

    def __init__(self) -> None:
        self.alpha: float = 0.10
        self._stratum_q: dict[tuple, float] = {}   # (station_id, hour_bin, danger_tier) -> q_hat
        self._global_q: float = 0.0                # fallback for unseen strata

    def fit(
        self,
        y_true: np.ndarray,
        y_lower: np.ndarray,
        y_upper: np.ndarray,
        station_ids: np.ndarray,
        local_hours: np.ndarray,
        danger_tiers: np.ndarray | None = None,
        alpha: float = 0.10,
    ) -> "MondianCQRCalibrator":
        """Fit per-stratum conformal q_hat from validation residuals.

        Parameters
        ----------
        y_true, y_lower, y_upper : arrays of shape (N,)
        station_ids : string array of shape (N,) — station identifier per row
        local_hours : int array of shape (N,) — local hour (0-23) per row
        danger_tiers : int array of shape (N,) — optional {0,1,2} gate tier per row
        alpha : miscoverage rate (default 0.10 → 90% PI)
        """
        self.alpha = alpha
        scores = np.maximum(y_lower - y_true, y_true - y_upper)
        hour_bins = np.asarray(local_hours, dtype=int) // 6
        tiers = np.zeros(len(scores), dtype=int) if danger_tiers is None else np.asarray(danger_tiers, dtype=int)

        # Global fallback
        n = len(y_true)
        level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        self._global_q = float(np.quantile(scores, level))

        # Per-stratum
        station_arr = np.asarray(station_ids)
        unique_strata = set(zip(station_arr, hour_bins, tiers))
        for sid, hbin, tier in unique_strata:
            mask = (station_arr == sid) & (hour_bins == hbin) & (tiers == tier)
            s = scores[mask]
            n_s = len(s)
            if n_s < 10:
                continue  # too few samples — rely on global fallback
            lv = min(1.0, np.ceil((n_s + 1) * (1 - alpha)) / n_s)
            self._stratum_q[(str(sid), int(hbin), int(tier))] = float(np.quantile(s, lv))

        return self

    def adjust(
        self,
        y_lower: np.ndarray,
        y_upper: np.ndarray,
        station_ids: np.ndarray,
        local_hours: np.ndarray,
        danger_tiers: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply per-stratum q_hat. Falls back to global for unseen strata."""
        y_lower = np.asarray(y_lower, dtype=float).copy()
        y_upper = np.asarray(y_upper, dtype=float).copy()
        hour_bins = np.asarray(local_hours, dtype=int) // 6
        tiers = np.zeros(len(y_lower), dtype=int) if danger_tiers is None else np.asarray(danger_tiers, dtype=int)

        for i, (sid, hbin, tier) in enumerate(zip(station_ids, hour_bins, tiers)):
            key = (str(sid), int(hbin), int(tier))
            q = self._stratum_q.get(key, self._global_q)
            y_lower[i] -= q
            y_upper[i] += q

        return y_lower, y_upper

    # ------------------------------------------------------------------
    # Serialisation — same pattern as EnbPICalibrator
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "global_q": self._global_q,
            "stratum_q": {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in self._stratum_q.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MondianCQRCalibrator":
        obj = cls()
        obj.alpha = float(d["alpha"])
        obj._global_q = float(d["global_q"])
        obj._stratum_q = {}
        for k, v in d.get("stratum_q", {}).items():
            parts = k.split("|")
            if len(parts) == 3:
                obj._stratum_q[(parts[0], int(parts[1]), int(parts[2]))] = float(v)
            else:
                # Backward compatibility with older (station|hour_bin) artifacts.
                obj._stratum_q[(parts[0], int(parts[1]), 0)] = float(v)
        return obj

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "MondianCQRCalibrator":
        return cls.from_dict(json.loads(path.read_text()))
