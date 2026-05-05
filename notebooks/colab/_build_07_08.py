"""Build 07_evaluate.ipynb and 08_register.ipynb for the HeatShield AI Colab pipeline.

Run with `python notebooks/colab/_build_07_08.py` to (re)generate both notebooks.
This file is only kept around for traceability — the notebooks themselves are the
artifacts the user opens; you can delete this builder if you prefer not to keep it.
"""
from __future__ import annotations

import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent


def _md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def _code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def _nb(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10"},
            "colab": {"provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# ---------------------------------------------------------------------------
# 07_evaluate.ipynb
# ---------------------------------------------------------------------------

EVAL_CELLS: list[dict] = []

EVAL_CELLS.append(_md(
"""# 07 — Comprehensive Evaluation / ประเมินโมเดลแบบครบวงจร

**EN.** Safety-oriented evaluation of the v3 forecast pipeline. Loads every
artifact produced by notebooks 04 (regression + quantile q05/q50/q95), 05
(danger / high-watch classifier), and 06 (calibration). Applies calibration and
`app.core.risk_fusion.fuse_risk` per row, then renders **all metrics tables and
charts inline** so a reviewer can sign off without opening any other file.

**TH.** ประเมินโมเดลแบบเน้นความปลอดภัย โหลด artifact ทั้งหมดจาก notebook 04
(regression + quantile q05/q50/q95), 05 (classifier P(High-Watch)/P(Danger)),
และ 06 (calibration). ใช้ calibration + `fuse_risk` ทีละแถว แล้วแสดง
**ตารางและกราฟทุกอันแบบ inline** เพื่อให้ตรวจรับงานได้โดยไม่ต้องเปิดไฟล์อื่น.

Sections / หัวข้อ

1. Setup
2. Load artifacts (per station × horizon)
3. Apply calibration + risk fusion
4. Regression metrics table
5. Classification metrics table
6. Uncertainty / PI metrics table
7. Charts (all inline)
8. Run-to-run delta vs previous `runs/`
9. Markdown summary (5 bullets)
"""
))

EVAL_CELLS.append(_md("## 1. Setup / ตั้งค่า"))

EVAL_CELLS.append(_code(
"""# --- bootstrap (re-run if you opened this notebook before 00_setup) -------
import os, sys
REPO_DIR = "/content/Heat-wave-backend"
if not os.path.exists(REPO_DIR):
    !bash {REPO_DIR}/scripts/colab_bootstrap.sh || true
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)
print(os.getcwd())
"""
))

EVAL_CELLS.append(_code(
"""# --- imports + matplotlib inline -----------------------------------------
%matplotlib inline
import json
from datetime import date, timedelta, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

from app.data.stations import STATIONS
from app.data.loaders import read_observations
from app.core.risk_fusion import fuse_risk, DEFAULT_THRESHOLDS
from app.core.calibration import Calibration
from app.ml.forecast.features import build_X_once
from app.ml.forecast.splitting import chronological_split
from app.ml.registry import load_latest_v3, list_v3_models

V3_DIR = Path("app/models/forecast_v3")
RUNS_DIR = Path("runs")
RUNS_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [6, 12, 24, 48, 72]
PEAK_HOURS = set(range(10, 19))  # 10:00-18:00 local
HI_HIGH_THRESHOLD = 40.0
DANGER_THRESHOLD = 42.0
HIGH_WATCH_THRESHOLD = 38.0

sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.float_format", lambda v: f"{v:.3f}" if isinstance(v, float) else str(v))
print("setup OK | stations:", list(STATIONS.keys()), "| horizons:", HORIZONS)
"""
))

EVAL_CELLS.append(_code(
"""# --- load risk thresholds (override fuse_risk defaults where applicable) -
THRESHOLDS_PATH = Path("configs/risk/thresholds.yaml")
risk_cfg = yaml.safe_load(THRESHOLDS_PATH.read_text(encoding="utf-8"))

FUSE_THRESHOLDS = dict(DEFAULT_THRESHOLDS)
alerts = risk_cfg.get("alerts", {})
if "danger_watch" in alerts:
    FUSE_THRESHOLDS["q90_danger"] = float(alerts["danger_watch"]["rule"]["q90_hi_c_gte"])
if "danger_alert" in alerts:
    FUSE_THRESHOLDS["p_danger_gate"] = float(alerts["danger_alert"]["rule"]["p_danger_gate"])
if "extreme_alert" in alerts:
    FUSE_THRESHOLDS["q50_danger"] = float(alerts["extreme_alert"]["rule"]["q50_hi_c_gte"])

print("Active fuse thresholds:")
for k, v in FUSE_THRESHOLDS.items():
    print(f"  {k:<22s} {v}")
"""
))

EVAL_CELLS.append(_md(
"""## 2. Load artifacts / โหลด artifact

For each `(station, horizon)` we load the v3 forecaster (regressor + quantile
heads), the danger gate classifier (lives inside the forecaster), and any
calibration sidecar. Missing artifacts are tolerated: that station/horizon
simply drops out of the unified val-fold prediction frame.
"""
))

EVAL_CELLS.append(_code(
"""# --- discover available (station, horizon) slots ---------------------------
choice_matrix = list_v3_models()
slots: list[tuple[str, int]] = []
for sid in STATIONS:
    for h in HORIZONS:
        slot_dir = V3_DIR / sid / f"h{h}"
        if (slot_dir / "bundle.json").exists():
            slots.append((sid, h))
print(f"Discovered {len(slots)} slots out of {len(STATIONS) * len(HORIZONS)} expected.")
for sid, h in slots[:8]:
    print(f"  {sid} h={h:>2d}  -> {choice_matrix.get(sid, {}).get(str(h), '?')}")
if len(slots) > 8:
    print(f"  ... +{len(slots) - 8} more")
"""
))

EVAL_CELLS.append(_code(
"""# --- helpers: build the val fold for one (station, horizon) ---------------
def _hours_from_X(X: pd.DataFrame, n: int) -> np.ndarray:
    if {"hour_sin", "hour_cos"} <= set(X.columns):
        ang = np.arctan2(X["hour_sin"], X["hour_cos"])
        return (np.round(ang / (2 * np.pi / 24)) % 24).astype(int).to_numpy()
    return np.arange(n) % 24


def _months_from_X(X: pd.DataFrame, n: int) -> np.ndarray:
    if "month" in X.columns:
        return X["month"].astype(int).to_numpy()
    return np.zeros(n, dtype=int)


def build_val_fold(station_id: str, horizon_h: int, days: int = 365) -> pd.DataFrame | None:
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=days)
    try:
        df_obs = read_observations(station_id, start, end)
    except Exception as exc:
        print(f"[{station_id} h{horizon_h}] read_observations failed: {exc}")
        return None
    if df_obs.empty or len(df_obs) < 60:
        print(f"[{station_id} h{horizon_h}] insufficient observations ({len(df_obs)})")
        return None

    X, df_aug = build_X_once(df_obs)
    if X.empty:
        print(f"[{station_id} h{horizon_h}] feature matrix is empty")
        return None

    # Chronological split — use the val partition for evaluation, no leakage.
    split = chronological_split(df_aug, horizon_h=horizon_h)
    val_ts = split.val.get("ts_utc")
    if val_ts is None or len(val_ts) == 0:
        return None
    val_mask = X.join(df_aug[["ts_utc"]], how="left")["ts_utc"].isin(val_ts)
    X_val = X[val_mask].copy()
    if X_val.empty:
        return None

    # Build target hi at t+horizon
    df_target = df_aug.set_index("ts_utc").sort_index()
    if "heat_index_c" not in df_target.columns:
        return None
    ts_val = df_aug.loc[X_val.index, "ts_utc"].to_numpy()
    target_ts = pd.to_datetime(ts_val) + pd.Timedelta(hours=horizon_h)
    y_true = df_target["heat_index_c"].reindex(target_ts).to_numpy()

    keep = ~np.isnan(y_true)
    X_val = X_val.loc[keep].reset_index(drop=True)
    y_true = y_true[keep]
    ts_val = ts_val[keep]
    if len(X_val) == 0:
        return None

    return pd.DataFrame({
        "ts_utc": pd.to_datetime(ts_val),
        "station_id": station_id,
        "horizon_h": horizon_h,
        "hour": _hours_from_X(X_val, len(X_val)),
        "month": _months_from_X(X_val, len(X_val)),
        "y_true": y_true,
        "_X_index": X_val.index,
    }).assign(_X=[X_val.iloc[[i]].reset_index(drop=True) for i in range(len(X_val))])
"""
))

EVAL_CELLS.append(_code(
"""# --- run prediction across all slots; build a unified val-fold frame ------
unified_rows: list[pd.DataFrame] = []
slot_status: list[dict] = []

for sid, h in slots:
    fold = build_val_fold(sid, h, days=365)
    if fold is None or len(fold) == 0:
        slot_status.append({"station": sid, "horizon": h, "rows": 0, "status": "no-data"})
        continue
    try:
        forecaster = load_latest_v3(sid, h)
    except FileNotFoundError as exc:
        slot_status.append({"station": sid, "horizon": h, "rows": 0, "status": f"missing:{exc}"})
        continue

    # Stack the per-row Xs back into a single frame for one batch predict.
    X_batch = pd.concat([row.iloc[0] for row in fold["_X"]], ignore_index=True)
    bundle = forecaster.predict_with_pi(X_batch, alpha=0.10)

    fold = fold.drop(columns=["_X", "_X_index"])
    fold["q05_raw"] = bundle.hi_lower
    fold["q50_raw"] = bundle.hi_mean
    fold["q95_raw"] = bundle.hi_upper
    fold["p_danger"] = bundle.danger_proba if bundle.danger_proba is not None else np.nan
    unified_rows.append(fold)
    slot_status.append({"station": sid, "horizon": h, "rows": len(fold), "status": "ok"})

status_df = pd.DataFrame(slot_status)
display(status_df)

if not unified_rows:
    raise RuntimeError("No predictions produced — check ingest/training notebooks first.")

predictions = pd.concat(unified_rows, ignore_index=True)
print(f"\\nUnified val-fold predictions: {len(predictions)} rows across "
      f"{predictions['station_id'].nunique()} stations and "
      f"{predictions['horizon_h'].nunique()} horizons.")
predictions.head()
"""
))

EVAL_CELLS.append(_md(
"""## 3. Apply calibration + risk fusion / Calibrate และ fuse risk

If a calibration sidecar (`calibration.json` from notebook 06) exists for a
station/horizon, we apply it to `q50_raw` to produce `q50`. Otherwise `q50 =
q50_raw`. We then call `fuse_risk` per row using the thresholds loaded above.
"""
))

EVAL_CELLS.append(_code(
"""# --- discover calibration sidecars ----------------------------------------
calibrations: dict[tuple[str, int], Calibration] = {}
for sid in STATIONS:
    for h in HORIZONS:
        cal_path = V3_DIR / sid / f"h{h}" / "calibration.json"
        if cal_path.exists():
            try:
                calibrations[(sid, h)] = Calibration.from_json(json.loads(cal_path.read_text()))
            except Exception as exc:
                print(f"[{sid} h{h}] calibration load failed: {exc}")
print(f"Loaded {len(calibrations)} calibration sidecars.")
"""
))

EVAL_CELLS.append(_code(
"""# --- apply calibration to q50_raw (per-station/horizon batch) -------------
predictions["q50"] = predictions["q50_raw"].astype(float)
for (sid, h), calib in calibrations.items():
    mask = (predictions["station_id"] == sid) & (predictions["horizon_h"] == h)
    if not mask.any():
        continue
    sub = predictions.loc[mask]
    corrected = calib.apply(
        sub["q50_raw"].to_numpy(),
        sub["station_id"].to_numpy(),
        sub["horizon_h"].to_numpy(),
        sub["hour"].to_numpy(),
    )
    predictions.loc[mask, "q50"] = corrected

# Quantile bands pass through (unless a band-specific calibration exists later).
predictions["q05"] = predictions["q05_raw"].astype(float)
predictions["q95"] = predictions["q95_raw"].astype(float)

# q90 alias used by fuse_risk (per thresholds.yaml convention)
sigma_hat = (predictions["q95"] - predictions["q50"]).clip(lower=0.0) / 1.6449
predictions["q90"] = np.maximum(predictions["q95"], predictions["q50"] + 1.28 * sigma_hat)
predictions["pi_width"] = (predictions["q95"] - predictions["q05"]).clip(lower=0.0)
predictions["abs_err"] = (predictions["q50"] - predictions["y_true"]).abs()
predictions["err"] = predictions["q50"] - predictions["y_true"]

# A no-classifier fallback: if p_danger is NaN, use a heuristic from q90.
predictions["p_danger"] = predictions["p_danger"].fillna(
    np.clip((predictions["q90"] - 36.0) / 8.0, 0.0, 1.0)
)
# P(High-Watch) heuristic: probability mass above 38°C (gaussian approx)
from scipy.stats import norm  # noqa: E402
sigma_safe = sigma_hat.replace(0, 0.5).fillna(0.5)
predictions["p_high_watch"] = 1.0 - norm.cdf(
    HIGH_WATCH_THRESHOLD, loc=predictions["q50"], scale=sigma_safe
)

predictions.head()
"""
))

EVAL_CELLS.append(_code(
"""# --- compute hi_anomaly_p95_excess per station × month × hour -------------
clim = predictions.groupby(["station_id", "month", "hour"])["y_true"].quantile(0.95)
clim = clim.rename("hi_p95_clim").reset_index()
predictions = predictions.merge(clim, on=["station_id", "month", "hour"], how="left")
predictions["hi_anomaly_p95_excess"] = (
    predictions["q50"] - predictions["hi_p95_clim"].fillna(predictions["q50"])
).clip(lower=0.0)
"""
))

EVAL_CELLS.append(_code(
"""# --- call fuse_risk per row -----------------------------------------------
def _fuse_row(row) -> dict:
    out = fuse_risk(
        hi_q50=float(row["q50"]),
        hi_q90=float(row["q90"]),
        p_high_watch=float(row["p_high_watch"]),
        p_danger=float(row["p_danger"]),
        hi_anomaly_p95_excess=float(row["hi_anomaly_p95_excess"]),
        horizon_h=int(row["horizon_h"]),
        pi_width=float(row["pi_width"]),
        thresholds=FUSE_THRESHOLDS,
    )
    return {
        "risk_level": out.risk_level,
        "confidence": out.confidence,
        "uncertainty_flag": out.uncertainty_flag,
        "horizon_type": out.horizon_type,
        "reason_codes": "|".join(out.reason_codes) if out.reason_codes else "",
    }

fusion = pd.DataFrame([_fuse_row(r) for r in predictions.to_dict("records")])
predictions = pd.concat([predictions.reset_index(drop=True), fusion], axis=1)

# Predicted high-watch / danger flags from the fused output.
predictions["pred_high_watch"] = predictions["risk_level"].isin(["Caution", "Watch", "Warning", "Danger"])
predictions["pred_danger"] = predictions["risk_level"].isin(["Warning", "Danger"])
predictions["true_high_watch"] = predictions["y_true"] >= HIGH_WATCH_THRESHOLD
predictions["true_danger"] = predictions["y_true"] >= DANGER_THRESHOLD

display(predictions[["ts_utc", "station_id", "horizon_h", "y_true", "q50", "q05", "q95",
                     "risk_level", "confidence", "uncertainty_flag", "reason_codes"]].head())
"""
))

EVAL_CELLS.append(_md("## 4. Regression metrics / ตัวชี้วัด regression"))

EVAL_CELLS.append(_code(
"""# --- compute 4 baselines per (station, horizon) --------------------------
def _baselines(group: pd.DataFrame) -> dict:
    g = group.sort_values("ts_utc")
    y = g["y_true"].to_numpy()
    n = len(y)
    if n < 25:
        nan = float("nan")
        return {"persistence_1h_mae": nan, "persistence_24h_mae": nan,
                "climatology_mae": nan, "seasonal_naive_mae": nan, "best_baseline_mae": nan}

    pers_1h = np.r_[y[0], y[:-1]]
    pers_24h = np.r_[y[:24], y[:-24]] if n > 24 else pers_1h
    clim = g.groupby(["month", "hour"])["y_true"].transform("mean").to_numpy()
    sn = np.r_[y[:min(168, n)], y[:-168]] if n > 168 else pers_24h
    bases = {
        "persistence_1h_mae": float(np.mean(np.abs(y - pers_1h))),
        "persistence_24h_mae": float(np.mean(np.abs(y - pers_24h))),
        "climatology_mae": float(np.mean(np.abs(y - clim))),
        "seasonal_naive_mae": float(np.mean(np.abs(y - sn))),
    }
    bases["best_baseline_mae"] = min(bases.values())
    return bases


def regression_metrics(group: pd.DataFrame) -> pd.Series:
    err = group["q50"] - group["y_true"]
    abs_err = err.abs()
    peak_mask = group["hour"].isin(PEAK_HOURS)
    high_mask = group["y_true"] >= HI_HIGH_THRESHOLD
    bases = _baselines(group)
    mae = float(abs_err.mean())
    skill = float(1.0 - mae / max(bases["best_baseline_mae"], 1e-6)) if not np.isnan(bases["best_baseline_mae"]) else float("nan")
    return pd.Series({
        "n": int(len(group)),
        "mae": mae,
        "rmse": float(np.sqrt((err ** 2).mean())),
        "bias": float(err.mean()),
        "peak_mae": float(abs_err[peak_mask].mean()) if peak_mask.any() else float("nan"),
        "high_hi_mae": float(abs_err[high_mask].mean()) if high_mask.any() else float("nan"),
        "best_baseline_mae": bases["best_baseline_mae"],
        "skill_score": skill,
    })


reg_per = predictions.groupby(["station_id", "horizon_h"]).apply(regression_metrics).reset_index()
overall = regression_metrics(predictions).to_frame().T
overall.insert(0, "horizon_h", "all")
overall.insert(0, "station_id", "ALL")
reg_table = pd.concat([reg_per, overall], ignore_index=True)
print("Regression metrics (per station × horizon, plus overall row):")
display(reg_table.style.format({
    "mae": "{:.3f}", "rmse": "{:.3f}", "bias": "{:+.3f}",
    "peak_mae": "{:.3f}", "high_hi_mae": "{:.3f}",
    "best_baseline_mae": "{:.3f}", "skill_score": "{:+.3f}"
}))
"""
))

EVAL_CELLS.append(_md("## 5. Classification metrics / ตัวชี้วัด classifier"))

EVAL_CELLS.append(_code(
"""# --- per-(station, horizon) classifier metrics + lead time ----------------
def classification_metrics(group: pd.DataFrame) -> pd.Series:
    tp_hw = ((group["pred_high_watch"]) & (group["true_high_watch"])).sum()
    fn_hw = ((~group["pred_high_watch"]) & (group["true_high_watch"])).sum()
    fp_hw = ((group["pred_high_watch"]) & (~group["true_high_watch"])).sum()
    pos_hw = group["true_high_watch"].sum()
    pos_d = group["true_danger"].sum()
    tp_d = ((group["pred_danger"]) & (group["true_danger"])).sum()

    high_recall = float(tp_hw / pos_hw) if pos_hw else float("nan")
    danger_recall = float(tp_d / pos_d) if pos_d else float("nan")
    fn_rate = float(fn_hw / pos_hw) if pos_hw else float("nan")
    precision = float(tp_hw / (tp_hw + fp_hw)) if (tp_hw + fp_hw) else float("nan")
    over_warn = float(fp_hw / max(len(group) - pos_hw, 1))

    # Lead time: for each true_high_watch event, how far ahead the model first
    # raised pred_high_watch. We treat horizon_h as the implicit lead time and
    # measure how often the warning persisted across recent hours.
    lead_times: list[int] = []
    g = group.sort_values("ts_utc").reset_index(drop=True)
    pred = g["pred_high_watch"].to_numpy()
    truth = g["true_high_watch"].to_numpy()
    for i, t in enumerate(truth):
        if not t:
            continue
        # walk backwards while pred stays True
        j = i
        while j > 0 and pred[j - 1]:
            j -= 1
        lead_times.append(int(group["horizon_h"].iloc[0]) + (i - j))
    lt_mean = float(np.mean(lead_times)) if lead_times else float("nan")
    lt_med = float(np.median(lead_times)) if lead_times else float("nan")

    return pd.Series({
        "n": int(len(group)),
        "pos_high_watch": int(pos_hw),
        "pos_danger": int(pos_d),
        "high_watch_recall": high_recall,
        "danger_recall": danger_recall,
        "false_negative_rate": fn_rate,
        "precision": precision,
        "over_warning_rate": over_warn,
        "lead_time_mean_h": lt_mean,
        "lead_time_median_h": lt_med,
    })


cls_per = predictions.groupby(["station_id", "horizon_h"]).apply(classification_metrics).reset_index()
cls_overall = classification_metrics(predictions).to_frame().T
cls_overall.insert(0, "horizon_h", "all")
cls_overall.insert(0, "station_id", "ALL")
cls_table = pd.concat([cls_per, cls_overall], ignore_index=True)
print("Classification metrics (per station × horizon, plus overall row):")
display(cls_table.style.format({
    "high_watch_recall": "{:.3f}", "danger_recall": "{:.3f}",
    "false_negative_rate": "{:.3f}", "precision": "{:.3f}",
    "over_warning_rate": "{:.3f}",
    "lead_time_mean_h": "{:.1f}", "lead_time_median_h": "{:.1f}"
}))
"""
))

EVAL_CELLS.append(_md("## 6. Uncertainty / PI metrics / ตัวชี้วัดความไม่แน่นอน"))

EVAL_CELLS.append(_code(
"""def uncertainty_metrics(group: pd.DataFrame) -> pd.Series:
    inside = (group["y_true"] >= group["q05"]) & (group["y_true"] <= group["q95"])
    day_mask = group["hour"].isin(PEAK_HOURS)
    high_mask = group["y_true"] >= HI_HIGH_THRESHOLD
    return pd.Series({
        "n": int(len(group)),
        "coverage_90_overall": float(inside.mean()),
        "coverage_90_daytime": float(inside[day_mask].mean()) if day_mask.any() else float("nan"),
        "coverage_90_high_hi": float(inside[high_mask].mean()) if high_mask.any() else float("nan"),
        "pi_width_median": float(group["pi_width"].median()),
        "pi_width_p90": float(group["pi_width"].quantile(0.90)),
    })


unc_per = predictions.groupby(["station_id", "horizon_h"]).apply(uncertainty_metrics).reset_index()
unc_overall = uncertainty_metrics(predictions).to_frame().T
unc_overall.insert(0, "horizon_h", "all")
unc_overall.insert(0, "station_id", "ALL")
unc_table = pd.concat([unc_per, unc_overall], ignore_index=True)
print("PI / uncertainty metrics (target = 0.90 coverage):")
display(unc_table.style.format({
    "coverage_90_overall": "{:.3f}", "coverage_90_daytime": "{:.3f}",
    "coverage_90_high_hi": "{:.3f}", "pi_width_median": "{:.3f}",
    "pi_width_p90": "{:.3f}"
}))
"""
))

EVAL_CELLS.append(_md(
"""## 7. Charts / กราฟ

Every chart is rendered inline. They cover: predicted-vs-actual scatter
(faceted across stations at h=24), error timeline, error-by-hour heatmap,
error-by-month, bias heatmap (station × horizon), PI coverage curve vs target,
ROC + PR for the classifier, threshold sensitivity, and a lead-time histogram.
"""
))

EVAL_CELLS.append(_code(
"""# 7.1 predicted vs actual scatter — h=24 panel per station ----------------
sub = predictions[predictions["horizon_h"] == 24]
stations_present = sorted(sub["station_id"].unique())
ncols = min(3, max(1, len(stations_present)))
nrows = int(np.ceil(len(stations_present) / ncols)) or 1
fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows), squeeze=False)
for ax, sid in zip(axes.flat, stations_present):
    g = sub[sub["station_id"] == sid]
    ax.scatter(g["y_true"], g["q50"], s=8, alpha=0.4)
    lo = float(min(g["y_true"].min(), g["q50"].min()))
    hi = float(max(g["y_true"].max(), g["q50"].max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.axhline(DANGER_THRESHOLD, color="red", ls=":", lw=1, alpha=0.6)
    ax.axvline(DANGER_THRESHOLD, color="red", ls=":", lw=1, alpha=0.6)
    ax.set_title(f"{sid} (h=24)")
    ax.set_xlabel("actual HI (°C)")
    ax.set_ylabel("predicted HI (°C)")
for ax in axes.flat[len(stations_present):]:
    ax.axis("off")
fig.suptitle("Predicted vs Actual — h=24 / พยากรณ์เทียบจริง", y=1.02)
fig.tight_layout()
plt.show()
"""
))

EVAL_CELLS.append(_code(
"""# 7.2 error timeline — single station / single horizon --------------------
sid_pick = stations_present[0] if stations_present else None
if sid_pick is not None:
    g = predictions[(predictions["station_id"] == sid_pick) & (predictions["horizon_h"] == 24)].sort_values("ts_utc")
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(g["ts_utc"], g["err"], lw=0.8, color="#444", label="q50 - actual")
    ax.fill_between(g["ts_utc"], g["q05"] - g["y_true"], g["q95"] - g["y_true"],
                    alpha=0.15, color="#1f77b4", label="q05-q95 band (residual)")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_title(f"Error timeline — {sid_pick} h=24 / ความคลาดเคลื่อนตามเวลา")
    ax.set_ylabel("error (°C)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    plt.show()
"""
))

EVAL_CELLS.append(_code(
"""# 7.3 error by hour-of-day — station × hour heatmap -----------------------
heat = (
    predictions.groupby(["station_id", "hour"])["abs_err"].mean().unstack("hour")
).reindex(columns=range(24))
fig, ax = plt.subplots(figsize=(12, 0.55 * max(3, len(heat))))
sns.heatmap(heat, cmap="YlOrRd", annot=True, fmt=".2f", ax=ax, cbar_kws={"label": "MAE (°C)"})
ax.set_title("MAE by station × hour of day / MAE ตามสถานี × ชั่วโมง")
fig.tight_layout()
plt.show()
"""
))

EVAL_CELLS.append(_code(
"""# 7.4 error by month -------------------------------------------------------
by_month = predictions.groupby(["station_id", "month"])["abs_err"].mean().unstack("month").reindex(columns=range(1, 13))
fig, ax = plt.subplots(figsize=(11, 0.55 * max(3, len(by_month))))
sns.heatmap(by_month, cmap="YlOrRd", annot=True, fmt=".2f", ax=ax, cbar_kws={"label": "MAE (°C)"})
ax.set_title("MAE by station × month / MAE ตามสถานี × เดือน")
fig.tight_layout()
plt.show()
"""
))

EVAL_CELLS.append(_code(
"""# 7.5 bias heatmap — station × horizon -------------------------------------
bias_pivot = predictions.groupby(["station_id", "horizon_h"])["err"].mean().unstack("horizon_h")
fig, ax = plt.subplots(figsize=(8, 0.6 * max(3, len(bias_pivot))))
sns.heatmap(bias_pivot, cmap="RdBu_r", center=0, annot=True, fmt="+.2f", ax=ax,
            cbar_kws={"label": "bias = q50 - actual (°C)"})
ax.set_title("Bias by station × horizon / อคติของพยากรณ์")
fig.tight_layout()
plt.show()
"""
))

EVAL_CELLS.append(_code(
"""# 7.6 PI coverage curve — coverage by horizon vs target band --------------
cov_curve = (
    predictions.assign(inside=lambda d: ((d["y_true"] >= d["q05"]) & (d["y_true"] <= d["q95"])).astype(int))
    .groupby("horizon_h")["inside"].mean().reset_index(name="coverage_90")
)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(cov_curve["horizon_h"], cov_curve["coverage_90"], marker="o", lw=1.5, label="empirical 90% PI coverage")
ax.axhline(0.90, color="green", ls="--", label="target 0.90")
ax.fill_between(cov_curve["horizon_h"], 0.85, 0.95, color="green", alpha=0.10, label="±0.05 band")
ax.set_xlabel("horizon (h)")
ax.set_ylabel("coverage")
ax.set_ylim(0.5, 1.0)
ax.set_title("Prediction-interval coverage vs target / ความครอบคลุมของช่วงพยากรณ์")
ax.legend()
fig.tight_layout()
plt.show()
"""
))

EVAL_CELLS.append(_code(
"""# 7.7 ROC + PR for the high-watch classifier (one station) ----------------
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

if sid_pick is not None:
    g = predictions[(predictions["station_id"] == sid_pick) & (predictions["horizon_h"] == 24)]
    if g["true_high_watch"].sum() >= 5 and (~g["true_high_watch"]).sum() >= 5:
        fpr, tpr, _ = roc_curve(g["true_high_watch"], g["p_high_watch"])
        roc_auc = auc(fpr, tpr)
        prec, rec, _ = precision_recall_curve(g["true_high_watch"], g["p_high_watch"])
        ap = average_precision_score(g["true_high_watch"], g["p_high_watch"])
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
        axes[0].set_xlabel("False Positive Rate")
        axes[0].set_ylabel("True Positive Rate")
        axes[0].set_title(f"ROC — {sid_pick} h=24")
        axes[0].legend()
        axes[1].plot(rec, prec, label=f"AP = {ap:.3f}")
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        axes[1].set_title(f"PR — {sid_pick} h=24")
        axes[1].legend()
        fig.tight_layout()
        plt.show()
    else:
        print(f"[{sid_pick} h=24] insufficient class balance for ROC/PR")
"""
))

EVAL_CELLS.append(_code(
"""# 7.8 threshold sensitivity — overall classifier across thresholds --------
thresholds_sweep = np.linspace(0.05, 0.95, 19)
rows = []
for t in thresholds_sweep:
    pred = (predictions["p_high_watch"] >= t)
    truth = predictions["true_high_watch"]
    tp = int(((pred) & (truth)).sum())
    fp = int(((pred) & (~truth)).sum())
    fn = int(((~pred) & (truth)).sum())
    tn = int(((~pred) & (~truth)).sum())
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)) if (precision and recall) else float("nan")
    rows.append({"threshold": t, "recall": recall, "precision": precision, "f1": f1})
sweep = pd.DataFrame(rows)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(sweep["threshold"], sweep["recall"], label="recall", marker="o")
ax.plot(sweep["threshold"], sweep["precision"], label="precision", marker="s")
ax.plot(sweep["threshold"], sweep["f1"], label="F1", marker="^")
ax.set_xlabel("P(High-Watch) threshold")
ax.set_ylabel("score")
ax.set_title("Threshold sensitivity — high-watch / ผลของ threshold ต่อตัวชี้วัด")
ax.legend()
fig.tight_layout()
plt.show()
"""
))

EVAL_CELLS.append(_code(
"""# 7.9 lead-time histogram --------------------------------------------------
def collect_lead_times(group: pd.DataFrame) -> list[int]:
    g = group.sort_values("ts_utc").reset_index(drop=True)
    pred = g["pred_high_watch"].to_numpy()
    truth = g["true_high_watch"].to_numpy()
    h = int(group["horizon_h"].iloc[0])
    out: list[int] = []
    for i, t in enumerate(truth):
        if not t:
            continue
        j = i
        while j > 0 and pred[j - 1]:
            j -= 1
        out.append(h + (i - j))
    return out

all_leads: list[int] = []
for (sid, h), g in predictions.groupby(["station_id", "horizon_h"]):
    all_leads.extend(collect_lead_times(g))

fig, ax = plt.subplots(figsize=(8, 4))
if all_leads:
    ax.hist(all_leads, bins=min(40, max(5, len(all_leads) // 5)), color="#1f77b4", alpha=0.85)
    ax.axvline(np.median(all_leads), color="red", ls="--", label=f"median = {np.median(all_leads):.1f}h")
    ax.legend()
ax.set_xlabel("lead time (hours before event)")
ax.set_ylabel("count")
ax.set_title("Lead-time distribution / การกระจายของเวลานำหน้า")
fig.tight_layout()
plt.show()
"""
))

EVAL_CELLS.append(_md(
"""## 8. Run-to-run delta vs previous `runs/` / เปรียบเทียบกับการรันก่อนหน้า

We persist this run's per-(station,horizon) metrics into `runs/{utc_ts}/` and
compare against the most recent prior run if any. Skipped gracefully on the
first run.
"""
))

EVAL_CELLS.append(_code(
"""# --- write current run + diff vs prior ------------------------------------
this_run = RUNS_DIR / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
this_run.mkdir(parents=True, exist_ok=True)
reg_table.to_csv(this_run / "regression.csv", index=False)
cls_table.to_csv(this_run / "classification.csv", index=False)
unc_table.to_csv(this_run / "uncertainty.csv", index=False)
print(f"current run dir: {this_run}")

prior_dirs = sorted([p for p in RUNS_DIR.iterdir() if p.is_dir() and p != this_run])
if not prior_dirs:
    print("No prior run found — skipping delta comparison.")
    delta_summary = None
else:
    prior = prior_dirs[-1]
    print(f"comparing against: {prior}")
    try:
        prev_reg = pd.read_csv(prior / "regression.csv")
        prev_cls = pd.read_csv(prior / "classification.csv")
        prev_unc = pd.read_csv(prior / "uncertainty.csv")
    except Exception as exc:
        print(f"could not read prior run: {exc}")
        delta_summary = None
    else:
        merged_reg = reg_table.merge(prev_reg, on=["station_id", "horizon_h"], suffixes=("", "_prev"))
        merged_cls = cls_table.merge(prev_cls, on=["station_id", "horizon_h"], suffixes=("", "_prev"))
        merged_unc = unc_table.merge(prev_unc, on=["station_id", "horizon_h"], suffixes=("", "_prev"))
        merged_reg["delta_mae"] = merged_reg["mae"] - merged_reg["mae_prev"]
        merged_cls["delta_high_watch_recall"] = merged_cls["high_watch_recall"] - merged_cls["high_watch_recall_prev"]
        merged_unc["delta_coverage_90"] = merged_unc["coverage_90_overall"] - merged_unc["coverage_90_overall_prev"]
        delta_summary = {
            "delta_mae": merged_reg[["station_id", "horizon_h", "mae", "mae_prev", "delta_mae"]],
            "delta_recall": merged_cls[["station_id", "horizon_h", "high_watch_recall",
                                        "high_watch_recall_prev", "delta_high_watch_recall"]],
            "delta_coverage": merged_unc[["station_id", "horizon_h", "coverage_90_overall",
                                          "coverage_90_overall_prev", "delta_coverage_90"]],
        }
        print("\\nDelta MAE (current - previous):")
        display(delta_summary["delta_mae"])
        print("\\nDelta High-Watch Recall:")
        display(delta_summary["delta_recall"])
        print("\\nDelta 90% PI Coverage:")
        display(delta_summary["delta_coverage"])
"""
))

EVAL_CELLS.append(_md(
"""## 9. Summary / สรุป

5-bullet summary describing what improved / regressed vs the previous run.
This cell is auto-generated; copy + paste into the PR description.
"""
))

EVAL_CELLS.append(_code(
"""# --- 5-bullet summary -----------------------------------------------------
overall_row = reg_table[reg_table["station_id"] == "ALL"].iloc[0]
overall_cls = cls_table[cls_table["station_id"] == "ALL"].iloc[0]
overall_unc = unc_table[unc_table["station_id"] == "ALL"].iloc[0]

bullets: list[str] = []
bullets.append(
    f"- Overall MAE = {overall_row['mae']:.3f}°C, RMSE = {overall_row['rmse']:.3f}°C, "
    f"bias = {overall_row['bias']:+.3f}°C, skill vs best baseline = {overall_row['skill_score']:+.3f}."
)
bullets.append(
    f"- Peak-hour (10-18 local) MAE = {overall_row['peak_mae']:.3f}°C; "
    f"high-HI (≥{HI_HIGH_THRESHOLD:.0f}°C) MAE = {overall_row['high_hi_mae']:.3f}°C."
)
bullets.append(
    f"- High-Watch recall = {overall_cls['high_watch_recall']:.3f}, "
    f"Danger recall = {overall_cls['danger_recall']:.3f}, "
    f"FN-rate = {overall_cls['false_negative_rate']:.3f}, "
    f"over-warning = {overall_cls['over_warning_rate']:.3f}."
)
bullets.append(
    f"- 90% PI coverage = {overall_unc['coverage_90_overall']:.3f} (target 0.90), "
    f"daytime = {overall_unc['coverage_90_daytime']:.3f}, "
    f"high-HI = {overall_unc['coverage_90_high_hi']:.3f}, "
    f"median PI width = {overall_unc['pi_width_median']:.2f}°C."
)
if delta_summary is not None:
    dmae = delta_summary["delta_mae"]["delta_mae"].mean()
    drec = delta_summary["delta_recall"]["delta_high_watch_recall"].mean()
    dcov = delta_summary["delta_coverage"]["delta_coverage_90"].mean()
    bullets.append(
        f"- vs previous run: ΔMAE = {dmae:+.3f}°C, ΔHigh-Watch recall = {drec:+.3f}, "
        f"Δ90%-coverage = {dcov:+.3f}."
    )
else:
    bullets.append("- vs previous run: not available (first run in `runs/`).")

print("\\n".join(bullets))
"""
))

EVAL_CELLS.append(_md(
"""### End of 07_evaluate / จบการประเมิน

**EN.** Hand the inline tables and charts to the reviewer; once they sign off,
proceed to `08_register.ipynb` to publish the artifacts.

**TH.** ส่งตารางและกราฟ inline ทั้งหมดให้ผู้ตรวจรับ เมื่อผ่านแล้วไปต่อที่
`08_register.ipynb` เพื่อ publish artifact.
"""
))


# ---------------------------------------------------------------------------
# 08_register.ipynb
# ---------------------------------------------------------------------------

REG_CELLS: list[dict] = []

REG_CELLS.append(_md(
"""# 08 — Register Artifacts / ลงทะเบียน artifact กลับเข้า repo

**EN.** Verifies the v3 layout, refreshes `choice_matrix.json`, validates that
the registry sidecar does **not** clobber the backend's `bundle.json` (per
`CLAUDE.md`), prints the exact `git`/`git lfs` commands to push the artifacts
back to the repo from a local machine, and writes `MODEL_CARD.md` if missing.

**TH.** ตรวจ layout v3, อัปเดต `choice_matrix.json`, ยืนยันว่า `registry.json`
ไม่ทับ `bundle.json` ของ backend (ตาม `CLAUDE.md`), แสดงคำสั่ง `git` / `git lfs`
ที่ใช้ push artifact ขึ้น repo จากเครื่อง local, และเขียน `MODEL_CARD.md`
ถ้ายังไม่มี.
"""
))

REG_CELLS.append(_md("## 1. Setup / ตั้งค่า"))

REG_CELLS.append(_code(
"""# --- bootstrap ------------------------------------------------------------
import os, sys
REPO_DIR = "/content/Heat-wave-backend"
if not os.path.exists(REPO_DIR):
    !bash {REPO_DIR}/scripts/colab_bootstrap.sh || true
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)
print(os.getcwd())
"""
))

REG_CELLS.append(_code(
"""# --- imports --------------------------------------------------------------
import json
from datetime import datetime, timezone
from pathlib import Path

from app.data.stations import STATIONS

V3_DIR = Path("app/models/forecast_v3")
HORIZONS = [6, 12, 24, 48, 72]
EXPECTED_BUNDLE = "bundle.json"
EXPECTED_REGISTRY = "registry.json"
OPTIONAL_FILES = ["classifier.json", "calibration.json"]

print("v3 root:", V3_DIR.resolve())
print("stations:", list(STATIONS.keys()))
print("horizons:", HORIZONS)
"""
))

REG_CELLS.append(_md(
"""## 2. Verify v3 layout / ตรวจสอบ layout

For each `(station, horizon)` we expect:

- `bundle.json` — backend-owned metadata (must exist).
- `registry.json` — registry sidecar from `save_model_v3`.
- A regressor head + q05/q50/q95 quantile heads (LightGBM `.txt` boosters
  for `lightgbm_quantile`, or `.ubj` boosters for the XGBoost backend).
- `classifier.json` — danger gate classifier (optional but expected).
- `calibration.json` — bias-correction sidecar from notebook 06 (optional).
"""
))

REG_CELLS.append(_code(
"""# --- walk the v3 tree and tabulate which files are present ---------------
import pandas as pd

rows = []
for sid in STATIONS:
    for h in HORIZONS:
        slot = V3_DIR / sid / f"h{h}"
        if not slot.exists():
            rows.append({"station": sid, "horizon": h, "exists": False, "files": ""})
            continue
        files = sorted(p.name for p in slot.iterdir() if p.is_file())
        rows.append({
            "station": sid,
            "horizon": h,
            "exists": True,
            "has_bundle": EXPECTED_BUNDLE in files,
            "has_registry": EXPECTED_REGISTRY in files,
            "has_classifier": "classifier.json" in files,
            "has_calibration": "calibration.json" in files,
            "n_boosters": sum(1 for f in files if f.endswith((".txt", ".ubj"))),
            "files": ", ".join(files),
        })
checklist = pd.DataFrame(rows)
display(checklist[["station", "horizon", "exists", "has_bundle", "has_registry",
                   "has_classifier", "has_calibration", "n_boosters"]])

missing = checklist[(~checklist["exists"]) | (~checklist.get("has_bundle", False))]
if len(missing) == 0:
    print("OK — every (station, horizon) slot has a bundle.json")
else:
    print("WARNING — missing slots:")
    display(missing)
"""
))

REG_CELLS.append(_md(
"""## 3. Update `choice_matrix.json` / อัปเดตเมทริกซ์เลือก backend

We pick `lightgbm_quantile` when the bundle's `backend_name` is one of the
LightGBM variants, else fall back to `xgboost`. The matrix is rewritten only
if the inferred mapping differs from disk.
"""
))

REG_CELLS.append(_code(
"""# --- read each bundle's backend_name and rebuild choice_matrix -----------
matrix_path = V3_DIR / "choice_matrix.json"
existing_matrix = json.loads(matrix_path.read_text()) if matrix_path.exists() else {}

inferred: dict[str, dict[str, str]] = {}
for sid in STATIONS:
    for h in HORIZONS:
        slot = V3_DIR / sid / f"h{h}"
        bundle_path = slot / EXPECTED_BUNDLE
        if not bundle_path.exists():
            continue
        try:
            bundle = json.loads(bundle_path.read_text())
        except Exception as exc:
            print(f"[{sid} h{h}] could not read bundle.json: {exc}")
            continue
        backend = bundle.get("backend_name")
        if not backend:
            # Fallback heuristic: prefer lightgbm_quantile if quantile heads exist.
            quantile_files = [p for p in slot.glob("q*.txt")] + [p for p in slot.glob("q*.ubj")]
            backend = "lightgbm_quantile" if quantile_files else "lightgbm"
        inferred.setdefault(sid, {})[str(h)] = backend

if inferred == existing_matrix:
    print("choice_matrix.json is already up-to-date.")
else:
    matrix_path.write_text(json.dumps(inferred, indent=2))
    print(f"Rewrote {matrix_path}")
    print(json.dumps(inferred, indent=2))
"""
))

REG_CELLS.append(_md(
"""## 4. Validate metadata / ตรวจสอบ metadata

Per `CLAUDE.md`, the registry sidecar (`registry.json`) MUST NOT overwrite the
backend-owned fields stored in `bundle.json`. We assert that no key written
into `registry.json` collides with a key in `bundle.json` whose value differs.
"""
))

REG_CELLS.append(_code(
"""# --- detect overlap between bundle.json and registry.json -----------------
issues: list[dict] = []
for sid in STATIONS:
    for h in HORIZONS:
        slot = V3_DIR / sid / f"h{h}"
        bp = slot / EXPECTED_BUNDLE
        rp = slot / EXPECTED_REGISTRY
        if not bp.exists() or not rp.exists():
            continue
        bundle = json.loads(bp.read_text())
        sidecar = json.loads(rp.read_text())
        for k, v in sidecar.items():
            if k in bundle and bundle[k] != v:
                # backend_name / feature_list / target_kind appear in both by
                # design — that's expected and they should match. Anything else
                # overlapping is a bug.
                if k in {"backend_name", "feature_list", "target_kind",
                         "station_id", "horizon_h"}:
                    if bundle[k] != v:
                        issues.append({"station": sid, "horizon": h, "key": k,
                                       "bundle": bundle[k], "registry": v,
                                       "severity": "design-mismatch"})
                else:
                    issues.append({"station": sid, "horizon": h, "key": k,
                                   "bundle": bundle[k], "registry": v,
                                   "severity": "clobber"})
if issues:
    print("Found metadata-overlap issues:")
    display(pd.DataFrame(issues))
else:
    print("OK — registry.json never clobbers backend-owned bundle.json fields.")
"""
))

REG_CELLS.append(_md(
"""## 5. Git push / คำสั่ง git ที่ต้องรัน local

We do NOT have credentials in Colab — these commands are printed for the
operator to run on their local machine after pulling the Drive copy. The
`git lfs track` lines cover both XGBoost binary (`*.ubj`) and LightGBM text
boosters (`*.txt`). Adjust globs if you decide LightGBM `.txt` files are small
enough to commit without LFS.
"""
))

REG_CELLS.append(_code(
"""# --- print the local commit/push recipe -----------------------------------
print("# 1. Sync from Drive to your local checkout (after `gcloud rsync`/`rclone`):")
print("git lfs install")
print('git lfs track "*.ubj" "*.txt"')
print("git add .gitattributes")
print("git add app/models/forecast_v3/")
print("git add configs/risk/thresholds.yaml runs/")
print('git commit -m "feat(forecast): retrain v3 forecasters $(date -u +%Y-%m-%d)"')
print("git push origin main")
print()
print("# 2. (Optional) dry-run from local to see what would change:")
print("git status --porcelain | head -50")
"""
))

REG_CELLS.append(_code(
"""# --- optional: live `git status --porcelain` if running on local Linux ----
import shutil, subprocess
git_bin = shutil.which("git")
if git_bin and Path(".git").exists():
    out = subprocess.run([git_bin, "status", "--porcelain"], capture_output=True, text=True)
    head = "\\n".join(out.stdout.splitlines()[:50])
    print(head if head else "(working tree clean)")
else:
    print("git not available or not a git repo (running in Colab is normal — skipping).")
"""
))

REG_CELLS.append(_md(
"""## 6. Local smoke test / smoke test บนเครื่อง local

After pulling the artifacts and committing locally, run the slow test suite
to confirm `load_latest_v3` resolves every (station, horizon):

```powershell
# Windows PowerShell (matches CLAUDE.md guidance)
.\\.venv\\Scripts\\Activate.ps1
pytest -m slow tests/                   # full slow suite
pytest -m slow tests/test_forecast.py   # forecast-only sanity
```

```bash
# macOS / Linux
source .venv/bin/activate
pytest -m slow tests/
```

Verify the FastAPI app picks up the new artifacts at startup:

```bash
uvicorn app.main:app --reload
# inspect logs for "Pre-warmed v3 forecaster ... h=24" lines per station.
```
"""
))

REG_CELLS.append(_md(
"""## 7. Model card / เขียน Model Card

`MODEL_CARD.md` is created **only if missing** so we never overwrite hand-edited
content. Edit the file in subsequent runs by hand or delete it before re-running
this cell.
"""
))

REG_CELLS.append(_code(
"""# --- write MODEL_CARD.md (idempotent: skipped if it exists) ---------------
card_path = V3_DIR / "MODEL_CARD.md"
if card_path.exists():
    print(f"{card_path} already exists — leaving it untouched.")
else:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    discovered_backends = sorted({
        json.loads(p.read_text()).get("backend_name", "?")
        for p in V3_DIR.glob("*/h*/bundle.json")
    })
    n_slots = sum(1 for _ in V3_DIR.glob("*/h*/bundle.json"))
    body = f'''# HeatShield AI — v3 Forecast Model Card / การ์ดโมเดล

_Last updated_ {today}

## Intended use / การใช้งานที่ตั้งใจ

**EN.** Hourly heat-index nowcasts and 6/12/24/48/72-hour forecasts for the
five Thai TMD stations registered in `app/data/stations.py` (BKK_01, CNX_01,
KKN_01, HYI_01, RYG_01). Outputs feed the FastAPI risk endpoint and the
public `risk_level` / action-card surface.

**TH.** ใช้สำหรับพยากรณ์ heat index รายชั่วโมงและล่วงหน้า 6/12/24/48/72
ชั่วโมง สำหรับ 5 สถานี TMD ที่ขึ้นทะเบียนใน `app/data/stations.py`.
ผลลัพธ์ป้อนเข้า FastAPI endpoint สำหรับความเสี่ยงและ action card.

## Data scope / ขอบเขตข้อมูล

- TMD station observations (1–2 y window).
- ERA5 reanalysis at 0.25° (3 y window via CDS API).
- NASA POWER (5 y window, no auth).
- Loader prefers ERA5 > NASA POWER for the same `(station_id, ts_utc)`.

## Backends in use / backend ที่ใช้งานจริง

- {", ".join(discovered_backends) or "(none discovered)"}
- {n_slots} (station, horizon) slots populated.
- Choice matrix lives at `app/models/forecast_v3/choice_matrix.json`.

## Known failure modes / ข้อจำกัดที่ทราบ

- **48 / 72 h horizons are awareness only.** `fuse_risk` flags
  `horizon_type="awareness"` for those horizons; consumers must surface them
  as situational guidance, not operational alerts. — โหมด 48/72h เป็นเพียง
  ข้อมูลเตือนล่วงหน้า ไม่ใช้เป็น operational alert.
- **Low-confidence guard.** When the most recent observation is older than 60
  minutes or the PI width exceeds 4°C (or 1.6× the v3 median PI width), the
  pipeline marks the prediction `low_confidence`. — มี guard ที่ตั้งสถานะ
  low_confidence อัตโนมัติ.
- **Closed station registry.** Only the five Thai TMD stations are supported.
  Adding a new station requires updating `app/data/stations.py` and rerunning
  ingest + training. — ระบบรองรับ 5 สถานีเท่านั้น.
- **Calibration coverage.** A station/horizon without `calibration.json`
  passes through uncalibrated; `fuse_risk` still works but bias may be larger.
  — ถ้าไม่มีไฟล์ calibration.json จะใช้ค่าดิบ.

## Retraining cadence / รอบการ retrain

- **Monthly** full retrain across all stations × horizons (driven by Colab
  notebooks 02–08 in order). — เทรนใหม่ทุกเดือน.
- **Ad-hoc** retrain when station coverage drops below 90% for the trailing
  30 days, or when the regression skill score versus the best baseline turns
  negative (see `07_evaluate.ipynb` overall row).
- After every retrain, the user requires the full evaluation report (metrics
  + all charts) inline before publishing — see `notebooks/colab/07_evaluate.ipynb`.

## Provenance / แหล่งที่มา

- Code: `app/ml/forecast/`, `app/core/risk_fusion.py`,
  `app/core/calibration.py`.
- Configs: `configs/risk/thresholds.yaml`,
  `configs/train/classifier.yaml` (if present).
- Tests: run `pytest -m slow tests/` after registration.
'''
    card_path.write_text(body, encoding="utf-8")
    print(f"wrote {card_path} ({len(body)} chars)")
"""
))

REG_CELLS.append(_md(
"""### End of 08_register / จบการลงทะเบียน

**EN.** All artifacts have been verified, the backend choice matrix is current,
metadata sidecars are conflict-free, and the model card is in place. Run the
printed `git` / `git lfs` commands locally to publish.

**TH.** Artifact ผ่านการตรวจครบ, choice matrix อัปเดตแล้ว, ไม่มี metadata ซ้อนทับ,
และ model card พร้อมใช้งาน. ใช้คำสั่ง `git` / `git lfs` ที่แสดงไว้บน local
เพื่อ push.
"""
))


# ---------------------------------------------------------------------------
# Write + validate
# ---------------------------------------------------------------------------

def main() -> None:
    eval_path = NB_DIR / "07_evaluate.ipynb"
    reg_path = NB_DIR / "08_register.ipynb"

    eval_nb = _nb(EVAL_CELLS)
    reg_nb = _nb(REG_CELLS)

    eval_path.write_text(json.dumps(eval_nb, indent=1, ensure_ascii=False), encoding="utf-8")
    reg_path.write_text(json.dumps(reg_nb, indent=1, ensure_ascii=False), encoding="utf-8")

    # Sanity: re-parse and count cells.
    for path in (eval_path, reg_path):
        nb = json.loads(path.read_text(encoding="utf-8"))
        assert nb["nbformat"] == 4
        print(f"{path.name}: {len(nb['cells'])} cells, valid JSON.")


if __name__ == "__main__":
    main()
