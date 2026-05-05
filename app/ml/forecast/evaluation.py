"""Reusable forecast evaluation metrics and artifacts."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
)

_CATEGORY_LABELS = ["Caution", "Extreme Caution", "Danger", "Extreme Danger"]


def categorize_heat_index(values: np.ndarray | pd.Series) -> np.ndarray:
    return pd.cut(
        pd.Series(values),
        bins=[-np.inf, 33, 42, 52, np.inf],
        labels=[0, 1, 2, 3],
    ).astype(int).to_numpy()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _hours_from_features(X: pd.DataFrame, n: int) -> np.ndarray:
    if {"hour_sin", "hour_cos"} <= set(X.columns):
        return (np.round(np.arctan2(X["hour_sin"], X["hour_cos"]) / (2 * np.pi / 24)) % 24).astype(int).to_numpy()
    return np.arange(n) % 24


def _station_names(X: pd.DataFrame, labels: Mapping[int, str] | None) -> pd.Series:
    if "station_enc" not in X.columns:
        return pd.Series(["all"] * len(X))
    codes = X["station_enc"].astype(int)
    if labels:
        return codes.map(lambda c: labels.get(int(c), f"station_{int(c)}"))
    return codes.map(lambda c: f"station_{int(c)}")


def _baseline_metrics(y_true: np.ndarray, X: pd.DataFrame) -> dict:
    if "heat_index_c_lag1h" in X.columns:
        persistence = X["heat_index_c_lag1h"].to_numpy(dtype=float)
    else:
        persistence = np.roll(y_true, 1)
        persistence[0] = y_true[0]
    persistence_mae = float(mean_absolute_error(y_true, persistence))

    if {"station_enc", "month", "hour_sin", "hour_cos"} <= set(X.columns):
        frame = pd.DataFrame({
            "y": y_true,
            "station": X["station_enc"].astype(int).to_numpy(),
            "month": X["month"].astype(int).to_numpy(),
            "hour": _hours_from_features(X, len(y_true)),
        })
        climatology = frame.groupby(["station", "month", "hour"])["y"].transform("mean").to_numpy()
    else:
        climatology = np.full_like(y_true, float(np.mean(y_true)), dtype=float)
    climatology_mae = float(mean_absolute_error(y_true, climatology))

    return {
        "persistence_mae": persistence_mae,
        "climatology_mae": climatology_mae,
    }


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    X: pd.DataFrame,
    *,
    q05: np.ndarray | None = None,
    q95: np.ndarray | None = None,
    horizon_h: int,
    runtime: dict | None = None,
    split_metadata: dict | None = None,
) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y_pred - y_true
    abs_err = np.abs(residual)
    baselines = _baseline_metrics(y_true, X)
    best_baseline = min(baselines["persistence_mae"], baselines["climatology_mae"])

    true_cat = categorize_heat_index(y_true)
    pred_cat = categorize_heat_index(y_pred)
    cm = confusion_matrix(true_cat, pred_cat, labels=[0, 1, 2, 3])
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, np.where(row_sums == 0, 1, row_sums), where=True)
    report = classification_report(
        true_cat,
        pred_cat,
        labels=[0, 1, 2, 3],
        target_names=_CATEGORY_LABELS,
        output_dict=True,
        zero_division=0,
    )

    def danger(threshold: float) -> dict:
        mask = y_true >= threshold
        support = int(mask.sum())
        if support == 0:
            return {"support": 0, "recall": float("nan"), "false_negative_rate": float("nan")}
        recall = float((y_pred[mask] >= threshold).mean())
        return {"support": support, "recall": recall, "false_negative_rate": 1.0 - recall}

    pi_metrics = {"available": False}
    if q05 is not None and q95 is not None:
        q05 = np.asarray(q05, dtype=float)
        q95 = np.asarray(q95, dtype=float)
        width = q95 - q05
        pi_metrics = {
            "available": True,
            "coverage_90": float(((y_true >= q05) & (y_true <= q95)).mean()),
            "mean_width": float(np.mean(width)),
            "p25_width": float(np.percentile(width, 25)),
            "p75_width": float(np.percentile(width, 75)),
            "width_ratio": float(np.percentile(width, 75) / max(np.percentile(width, 25), 1e-6)),
            "pinball_q05": float(np.mean(np.maximum(0.05 * (y_true - q05), (0.05 - 1) * (y_true - q05)))),
            "pinball_q95": float(np.mean(np.maximum(0.95 * (y_true - q95), (0.95 - 1) * (y_true - q95)))),
        }

    return {
        "horizon_h": horizon_h,
        "regression": {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "bias": float(np.mean(residual)),
            "correlation": float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else float("nan"),
            "p50_abs_error": float(np.percentile(abs_err, 50)),
            "p90_abs_error": float(np.percentile(abs_err, 90)),
            "p95_abs_error": float(np.percentile(abs_err, 95)),
        },
        "baselines": {
            **baselines,
            "skill_score": float(1.0 - mean_absolute_error(y_true, y_pred) / max(best_baseline, 1e-6)),
        },
        "risk": {
            "labels": _CATEGORY_LABELS,
            "classification_report": report,
            "confusion_matrix": cm,
            "confusion_matrix_normalized": cm_norm,
        },
        "safety": {
            "danger_40": danger(40.0),
            "danger_42": danger(42.0),
        },
        "prediction_interval": pi_metrics,
        "runtime": {
            "feature_rows": int(len(X)),
            **(runtime or {}),
        },
        "split": split_metadata or {},
    }


def _save_confusion(metrics: dict, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(metrics["risk"]["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
                xticklabels=_CATEGORY_LABELS, yticklabels=_CATEGORY_LABELS, ax=axes[0])
    axes[0].set_title("Risk Category Counts")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    sns.heatmap(metrics["risk"]["confusion_matrix_normalized"], annot=True, fmt=".2f", cmap="Blues",
                xticklabels=_CATEGORY_LABELS, yticklabels=_CATEGORY_LABELS, ax=axes[1])
    axes[1].set_title("Risk Category Recall")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def _save_classification(metrics: dict, out_dir: Path) -> None:
    report = metrics["risk"]["classification_report"]
    rows = [report[label] for label in _CATEGORY_LABELS]
    x = np.arange(len(_CATEGORY_LABELS))
    fig, ax = plt.subplots(figsize=(10, 5))
    for offset, key in [(-0.25, "precision"), (0.0, "recall"), (0.25, "f1-score")]:
        ax.bar(x + offset, [r[key] for r in rows], width=0.25, label=key)
    ax.set_xticks(x)
    ax.set_xticklabels(_CATEGORY_LABELS, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "classification_report.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def _save_error_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    X: pd.DataFrame,
    out_dir: Path,
    station_labels: Mapping[int, str] | None,
) -> None:
    residual = y_pred - y_true
    abs_err = np.abs(residual)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(y_true, y_pred, s=12, alpha=0.55)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    axes[0].plot([lo, hi], [lo, hi], "k--", linewidth=1)
    axes[0].set_xlabel("Actual HI")
    axes[0].set_ylabel("Predicted HI")
    axes[1].hist(residual, bins=min(40, max(5, len(y_true) // 2)))
    axes[1].axvline(0, color="black", linestyle="--")
    axes[1].set_xlabel("Prediction Error")
    fig.tight_layout()
    fig.savefig(out_dir / "pred_vs_actual.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    hours = _hours_from_features(X, len(y_true))
    by_hour = pd.DataFrame({"hour": hours, "abs_err": abs_err, "err": residual}).groupby("hour").mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    by_hour["abs_err"].reindex(range(24), fill_value=0).plot(kind="bar", ax=ax)
    ax.set_ylabel("MAE")
    fig.tight_layout()
    fig.savefig(out_dir / "error_by_hour.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    stations = _station_names(X, station_labels)
    by_station = pd.DataFrame({"station": stations, "abs_err": abs_err}).groupby("station").mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    by_station["abs_err"].plot(kind="bar", ax=ax)
    ax.set_ylabel("MAE")
    fig.tight_layout()
    fig.savefig(out_dir / "error_by_station.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    heat = pd.DataFrame({"station": stations, "hour": hours, "abs_err": abs_err})
    pivot = heat.pivot_table(index="station", columns="hour", values="abs_err", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(12, max(3, 0.8 * len(pivot))))
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "error_heatmap.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def _save_pi_plots(y_true: np.ndarray, q05: np.ndarray | None, q95: np.ndarray | None, out_dir: Path) -> None:
    if q05 is None or q95 is None:
        return
    coverage = ((y_true >= q05) & (y_true <= q95)).astype(int)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["outside", "inside"], [(coverage == 0).mean(), coverage.mean()])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share")
    fig.tight_layout()
    fig.savefig(out_dir / "pi_calibration.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(q95 - q05, bins=min(40, max(5, len(y_true) // 2)))
    ax.set_xlabel("q95 - q05")
    fig.tight_layout()
    fig.savefig(out_dir / "pi_width.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def _write_summary(metrics: dict, out_dir: Path) -> None:
    reg = metrics["regression"]
    base = metrics["baselines"]
    pi = metrics["prediction_interval"]
    lines = [
        "# Forecast Evaluation",
        "",
        f"- Horizon: h{metrics['horizon_h']}",
        f"- MAE: {reg['mae']:.3f}",
        f"- RMSE: {reg['rmse']:.3f}",
        f"- Skill score: {base['skill_score']:.3f}",
        f"- Danger recall >=42C: {metrics['safety']['danger_42']['recall']}",
        f"- PI available: {pi['available']}",
    ]
    if pi.get("available"):
        lines.append(f"- PI coverage 90: {pi['coverage_90']:.3f}")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_predictions(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    X: pd.DataFrame,
    *,
    out_dir: str | Path,
    horizon_h: int,
    station_labels: Mapping[int, str] | None = None,
    q05: np.ndarray | pd.Series | None = None,
    q95: np.ndarray | pd.Series | None = None,
    runtime: dict | None = None,
    split_metadata: dict | None = None,
) -> dict:
    """Compute metrics and write standard evaluation artifacts."""
    started = time.perf_counter()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    q05_arr = None if q05 is None else np.asarray(q05, dtype=float)
    q95_arr = None if q95 is None else np.asarray(q95, dtype=float)
    runtime_data = dict(runtime or {})

    metrics = compute_metrics(
        y_true_arr,
        y_pred_arr,
        X,
        q05=q05_arr,
        q95=q95_arr,
        horizon_h=horizon_h,
        runtime=runtime_data,
        split_metadata=split_metadata,
    )
    metrics["runtime"]["eval_seconds"] = metrics["runtime"].get(
        "eval_seconds", float(time.perf_counter() - started)
    )

    (out_path / "metrics.json").write_text(
        json.dumps(_json_safe(metrics), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_summary(metrics, out_path)
    _save_confusion(metrics, out_path)
    _save_classification(metrics, out_path)
    _save_error_plots(y_true_arr, y_pred_arr, X, out_path, station_labels)
    _save_pi_plots(y_true_arr, q05_arr, q95_arr, out_path)
    return _json_safe(metrics)
