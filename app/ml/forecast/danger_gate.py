"""Danger-class gate classifier for HeatShield AI forecast v3.

Trains a class-weighted LightGBM binary classifier to detect Danger-class
heat events (heat index ≥ danger_thresh °C). Used by lgbm_backend.py to
route samples through a Danger-conditional regressor when proba > threshold.

Why a separate gate? The two-head (T, RH) regressor optimises MSE, which
de-emphasises the rare tail. The gate classifier directly optimises for
Danger detection and can be tuned independently.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


class DangerGate:
    """Binary gate: P(heat_index >= danger_thresh) for a given forecast horizon.

    Attributes
    ----------
    threshold : float
        Decision boundary for proba → "Danger" (swept on validation F1).
    danger_thresh : float
        HI threshold defining the positive class (default 40°C — slightly
        below the 42°C Danger boundary to give the classifier more positives).
    """

    def __init__(self) -> None:
        self.threshold: float = 0.35
        self.danger_thresh: float = 40.0
        self._clf = None  # lgb.Booster

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y_hi: pd.Series,
        *,
        danger_thresh: float = 40.0,
        val_X: pd.DataFrame | None = None,
        val_y_hi: pd.Series | None = None,
        n_estimators: int = 500,
        random_state: int = 42,
    ) -> "DangerGate":
        """Train the binary gate classifier.

        Parameters
        ----------
        X, y_hi : training features and heat-index target series.
        danger_thresh : HI boundary for positive class.
        val_X, val_y_hi : held-out validation set for threshold sweep.
        """
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError("pip install lightgbm") from exc

        self.danger_thresh = danger_thresh
        y_bin = (y_hi >= danger_thresh).astype(int)

        n_pos = int(y_bin.sum())
        n_neg = int(len(y_bin) - n_pos)
        scale_pos = max(1.0, n_neg / max(n_pos, 1))

        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "scale_pos_weight": scale_pos,
            "num_leaves": 63,
            "learning_rate": 0.05,
            "n_estimators": n_estimators,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": random_state,
            "verbose": -1,
        }

        dtrain = lgb.Dataset(X, label=y_bin)

        callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
        eval_sets = []

        if val_X is not None and val_y_hi is not None:
            val_bin = (val_y_hi >= danger_thresh).astype(int)
            dval = lgb.Dataset(val_X, label=val_bin, reference=dtrain)
            eval_sets = [dval]

        self._clf = lgb.train(
            params,
            dtrain,
            valid_sets=eval_sets if eval_sets else None,
            callbacks=callbacks if eval_sets else [lgb.log_evaluation(-1)],
        )

        # Sweep threshold on validation set if provided
        if val_X is not None and val_y_hi is not None:
            self._sweep_threshold(val_X, val_y_hi >= danger_thresh)

        return self

    def _sweep_threshold(
        self, X: pd.DataFrame, y_bin: pd.Series
    ) -> None:
        """Find the threshold that maximises recall subject to a precision constraint."""
        PRECISION_MIN = 0.55
        RECALL_MIN = 0.40

        probas = self.predict_proba(X)

        # First pass: find thresholds that meet both constraints
        valid_thresholds = []
        for t in np.arange(0.10, 0.80, 0.05):
            preds = (probas >= t).astype(int)
            tp = int(((preds == 1) & (y_bin == 1)).sum())
            fp = int(((preds == 1) & (y_bin == 0)).sum())
            fn = int(((preds == 0) & (y_bin == 1)).sum())
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            if prec >= PRECISION_MIN and rec >= RECALL_MIN:
                valid_thresholds.append((t, rec, prec))

        if valid_thresholds:
            # Pick the threshold with highest recall (break ties by highest precision)
            best = max(valid_thresholds, key=lambda x: (x[1], x[2]))
            self.threshold = best[0]
        else:
            # Fallback: no threshold meets both constraints — maximise recall
            # with precision >= 0.40
            fallback = []
            for t in np.arange(0.10, 0.80, 0.05):
                preds = (probas >= t).astype(int)
                tp = int(((preds == 1) & (y_bin == 1)).sum())
                fp = int(((preds == 1) & (y_bin == 0)).sum())
                fn = int(((preds == 0) & (y_bin == 1)).sum())
                prec = tp / max(tp + fp, 1)
                rec = tp / max(tp + fn, 1)
                if prec >= 0.40:
                    fallback.append((t, rec))
            if fallback:
                self.threshold = max(fallback, key=lambda x: x[1])[0]
            else:
                self.threshold = 0.35  # last resort default

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(Danger) for each row in X, shape (N,)."""
        if self._clf is None:
            raise RuntimeError("DangerGate not fitted — call .fit() first")
        return self._clf.predict(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return binary Danger prediction using self.threshold."""
        return (self.predict_proba(X) >= self.threshold).astype(int)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Save to a directory: gate_clf.txt (LightGBM) + gate_meta.json."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self._clf is not None:
            self._clf.save_model(str(path / "gate_clf.txt"))
        meta = {
            "threshold": self.threshold,
            "danger_thresh": self.danger_thresh,
        }
        (path / "gate_meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: Path) -> "DangerGate":
        """Load from a previously saved directory."""
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError("pip install lightgbm") from exc

        path = Path(path)
        obj = cls()
        meta = json.loads((path / "gate_meta.json").read_text())
        obj.threshold = float(meta["threshold"])
        obj.danger_thresh = float(meta["danger_thresh"])
        clf_path = path / "gate_clf.txt"
        if clf_path.exists():
            obj._clf = lgb.Booster(model_file=str(clf_path))
        return obj
