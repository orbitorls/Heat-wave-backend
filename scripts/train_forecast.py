"""CLI training script for the HeatShield AI heat-index forecast model.

Usage:
    # Multi-horizon (default, trains all 5 horizons):
    python scripts/train_forecast.py [--station BKK_01] [--trials 30]

    # Custom horizons:
    python scripts/train_forecast.py --horizons 6,24,48

    # Legacy single-horizon (falls back to single-horizon save):
    python scripts/train_forecast.py --horizon 24 [--station BKK_01] [--trials 30]

Reads parquet data from data/raw/, builds features, trains XGBoost,
saves model to app/models/forecast_v{n}/ directory.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import subprocess
import sys
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from threading import Lock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd

from app.data.loaders import read_observations
from app.data.stations import STATIONS
from app.ml.forecast.features import build_features, build_X_once, build_y_for_horizon, _DEFAULT_LAGS_H, _DEFAULT_ROLLING_H
from app.ml.forecast.train import train
from app.ml.forecast.evaluation import evaluate_predictions
from app.ml.forecast._optuna_utils import write_heartbeat
from app.ml.forecast.splitting import split_xy
from app.ml.registry import save_model
from app.ml.viz import generate_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _can_use_lgbm_device(device_type: str) -> bool:
    """Probe whether LightGBM can run on a requested device."""
    try:
        import lightgbm as lgb
        import numpy as np
        lgb.train(
            {"objective": "regression", "device_type": device_type, "verbose": -1, "num_leaves": 4},
            lgb.Dataset(np.zeros((8, 2)), np.zeros(8)),
            num_boost_round=1,
        )
        return True
    except Exception:
        return False


def _configure_training_device(requested: str) -> None:
    """Configure env vars so training prefers the requested device."""
    req = requested.lower().strip()
    if req == "cpu":
        os.environ["HEATSHIELD_FORCE_CPU"] = "1"
        os.environ.pop("LGBM_DEVICE", None)
        logger.info("Training device configured: CPU")
        return

    os.environ["HEATSHIELD_FORCE_CPU"] = "0"
    if req == "auto":
        os.environ.pop("LGBM_DEVICE", None)
        logger.info("Training device configured: auto-detect")
        return

    if req in {"gpu", "cuda"} and _can_use_lgbm_device(req):
        os.environ["LGBM_DEVICE"] = req
        logger.info("Training device configured: LGBM_DEVICE=%s", req)
        return

    alternate = "gpu" if req == "cuda" else "cuda"
    if req in {"gpu", "cuda"} and _can_use_lgbm_device(alternate):
        os.environ["LGBM_DEVICE"] = alternate
        logger.warning(
            "Requested LightGBM device '%s' unavailable. Falling back to '%s'.",
            req,
            alternate,
        )
        return

    os.environ.pop("LGBM_DEVICE", None)
    logger.warning(
        "Requested LightGBM device '%s' unavailable. Falling back to auto-detect.",
        req,
    )


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return "unknown"


def _fmt_metrics(metrics: dict) -> dict:
    return {k: f"{v:.4f}" if not (isinstance(v, float) and v != v) else "nan" for k, v in metrics.items()}


def _quality_status(metrics: dict) -> str:
    skill = metrics.get("baselines", {}).get("skill_score", metrics.get("skill_score", 0.0))
    danger = metrics.get("safety", {}).get("danger_42", {}).get("recall", metrics.get("danger_recall", 0.0))
    pi = metrics.get("prediction_interval", {})
    coverage = pi.get("coverage_90") if pi.get("available") else None
    # Production minimum: all slots must at least beat baselines.
    if skill < 0.0:
        return "not_ready"
    if coverage is not None and not (0.85 <= coverage <= 0.93):
        return "candidate"
    if skill >= 0.55 and (danger is None or danger != danger or danger >= 0.40):
        return "ready"
    return "candidate"


def _row_from_existing_slot(slot_dir: Path, station_id: str, horizon_h: int) -> dict:
    """Build a leaderboard row from existing v3 artifacts for resume mode."""
    bundle_path = slot_dir / "bundle.json"
    registry_path = slot_dir / "registry.json"
    backend = "unknown"
    status = "skipped_existing"
    mae = float("nan")
    skill = float("nan")
    danger_recall = float("nan")
    eval_dir = ""

    if bundle_path.exists():
        try:
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            backend = bundle.get("backend_name", backend)
        except Exception:
            pass

    if registry_path.exists():
        try:
            registry_meta = json.loads(registry_path.read_text(encoding="utf-8"))
            backend = registry_meta.get("backend_name", backend)
            status = registry_meta.get("status", status)
            metrics = registry_meta.get("evaluation", {})
            mae = metrics.get("regression", {}).get("mae", mae)
            skill = metrics.get("baselines", {}).get("skill_score", skill)
            danger_recall = metrics.get("safety", {}).get("danger_42", {}).get("recall", danger_recall)
            eval_dir = metrics.get("paths", {}).get("out_dir", eval_dir)
        except Exception:
            pass

    return {
        "backend": backend,
        "station": station_id,
        "horizon_h": horizon_h,
        "mae": float(mae) if mae is not None else float("nan"),
        "skill_score": float(skill) if skill is not None else float("nan"),
        "danger_recall_42": float(danger_recall) if danger_recall is not None else float("nan"),
        "status": status,
        "eval_dir": eval_dir,
    }


_EVAL_CHART_NAMES = [
    "confusion_matrix.png",
    "classification_report.png",
    "pred_vs_actual.png",
    "error_by_hour.png",
    "error_by_station.png",
    "error_heatmap.png",
    "pi_calibration.png",
    "pi_width.png",
]

def _next_version_dir() -> Path:
    """Return the next logs/eval/v{n}/ folder that does not yet exist."""
    base = Path("logs") / "eval"
    base.mkdir(parents=True, exist_ok=True)
    existing = [
        int(d.name[1:])
        for d in base.iterdir()
        if d.is_dir() and d.name.startswith("v") and d.name[1:].isdigit()
    ]
    n = (max(existing) + 1) if existing else 1
    return base / f"v{n}"


def _export_run_to_versioned_dir(run_dir: Path, primary_station: str = "BKK_01", primary_h: int = 24) -> Path:
    """Copy run charts into a new logs/eval/v{n}/ folder — flat layout matching v1/v2.

    Structure written:
      logs/eval/v{n}/confusion_matrix.png
      logs/eval/v{n}/classification_report.png
      logs/eval/v{n}/pred_vs_actual.png
      logs/eval/v{n}/error_by_hour.png
      logs/eval/v{n}/error_by_station.png
      logs/eval/v{n}/error_heatmap.png
      logs/eval/v{n}/pi_calibration.png
      logs/eval/v{n}/pi_width.png
      logs/eval/v{n}/summary.png
      logs/eval/v{n}/leaderboard.md
      logs/eval/v{n}/leaderboard.json

    The 8 per-slot charts come from the primary slot (primary_station/h{primary_h}).
    Falls back to the first available slot if primary is missing.
    """
    import shutil

    dest = _next_version_dir()
    dest.mkdir(parents=True, exist_ok=True)

    # Locate primary slot dir: run_dir/{backend}/{station}/h{H}/
    primary_dir: Path | None = None
    for candidate in sorted(run_dir.rglob(f"h{primary_h}")):
        if candidate.is_dir() and candidate.parent.name == primary_station:
            primary_dir = candidate
            break
    if primary_dir is None:
        # Fall back to first slot found
        for name in _EVAL_CHART_NAMES:
            hits = sorted(run_dir.rglob(name))
            if hits:
                primary_dir = hits[0].parent
                break

    # Copy the 8 charts flat into dest
    if primary_dir:
        for name in _EVAL_CHART_NAMES:
            src = primary_dir / name
            if src.exists():
                shutil.copy2(src, dest / name)

    # Copy summary + leaderboard
    for fname in ("summary.png", "leaderboard.md", "leaderboard.json"):
        src = run_dir / fname
        if src.exists():
            shutil.copy2(src, dest / fname)

    logger.info("Run exported to %s", dest.resolve())
    return dest


def _display_eval_charts(run_dir: Path) -> None:
    """Print all eval chart paths and open via OS image viewer (Windows)."""
    found: list[Path] = []
    for name in _EVAL_CHART_NAMES:
        found.extend(sorted(run_dir.rglob(name)))
    if not found:
        return

    logger.info("=" * 60)
    logger.info("POST-TRAINING EVAL CHARTS (%d files)", len(found))
    logger.info("=" * 60)
    prev_slot = None
    for p in found:
        slot = str(p.parent.relative_to(run_dir))
        if slot != prev_slot:
            logger.info("")
            logger.info("[%s]", slot)
            prev_slot = slot
        logger.info("  EVAL_CHART: %s", p.resolve())

    logger.info("")
    logger.info("=" * 60)

    if sys.platform == "win32":
        for p in found:
            try:
                os.startfile(str(p))
            except Exception:
                pass


def _generate_summary_chart(run_dir: Path, rows: list[dict]) -> Path:
    """Create a single-page summary: MAE / skill / danger-recall heatmaps across station × horizon."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np

    stations = sorted({r["station"] for r in rows})
    horizons = sorted({r["horizon_h"] for r in rows})

    def _grid(metric: str, default=float("nan")) -> np.ndarray:
        def _safe(v):
            return float("nan") if v is None else float(v)
        lookup = {(r["station"], r["horizon_h"]): _safe(r.get(metric, default)) for r in rows}
        return np.array([[lookup.get((s, h), float("nan")) for h in horizons] for s in stations], dtype=float)

    mae_grid = _grid("mae")
    skill_grid = _grid("skill_score")
    danger_grid = _grid("danger_recall_42")

    fig, axes = plt.subplots(1, 3, figsize=(16, max(3, len(stations) * 1.2 + 2)))
    fig.suptitle(f"Training Run Summary  —  {run_dir.name}", fontsize=13, fontweight="bold")

    h_labels = [f"h{h}" for h in horizons]

    def _heatmap(ax, data, title, cmap, vmin, vmax, fmt=".2f", bad_color="#cccccc"):
        masked = np.ma.masked_invalid(data)
        im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(horizons))); ax.set_xticklabels(h_labels, fontsize=9)
        ax.set_yticks(range(len(stations))); ax.set_yticklabels(stations, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        plt.colorbar(im, ax=ax, shrink=0.8)
        for i in range(len(stations)):
            for j in range(len(horizons)):
                v = data[i, j]
                if not np.isnan(v):
                    ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=8,
                            color="white" if abs(v - (vmin + vmax) / 2) > (vmax - vmin) * 0.2 else "black")

    _heatmap(axes[0], mae_grid,    "MAE (°C)\nlow = good",      "RdYlGn_r", 1.0, 4.0)
    _heatmap(axes[1], skill_grid,  "Skill Score\nhigh = good",  "RdYlGn",   0.0, 1.0)
    _heatmap(axes[2], danger_grid, "Danger Recall ≥42°C\nhigh = good", "RdYlGn", 0.0, 1.0)

    plt.tight_layout()
    out = run_dir / "summary.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    return out


def _write_leaderboard(run_dir: Path, rows: list[dict]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    json_path = run_dir / "leaderboard.json"
    json_tmp = json_path.with_suffix(".json.tmp")
    json_tmp.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    os.replace(json_tmp, json_path)
    lines = [
        "| Backend | Station | Horizon | MAE | Skill | Danger Recall >=42C | Status |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['backend']} | {row['station']} | {row['horizon_h']} | "
            f"{row.get('mae', float('nan')):.3f} | {row.get('skill_score', float('nan')):.3f} | "
            f"{(row['danger_recall_42'] if row.get('danger_recall_42') is not None else float('nan')):.3f} | {row['status']} |"
        )
    md_path = run_dir / "leaderboard.md"
    md_tmp = md_path.with_suffix(".md.tmp")
    md_tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(md_tmp, md_path)


def _read_champion_mae(slot_dir: Path) -> float | None:
    """Return the existing slot's test MAE from registry.json, or None if no slot exists."""
    registry_path = slot_dir / "registry.json"
    if not registry_path.exists():
        return None
    try:
        data = json.loads(registry_path.read_text(encoding="utf-8"))
        mae = data.get("evaluation", {}).get("regression", {}).get("mae")
        return float(mae) if mae is not None else None
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HeatShield AI forecast model")
    parser.add_argument(
        "--horizon", type=int, default=None,
        help="Legacy single-horizon flag in hours. If set, overrides --horizons and uses single-horizon save.",
    )
    parser.add_argument(
        "--horizons", type=str, default="6,12,24,48,72",
        help="Comma-separated forecast horizons in hours (default: 6,12,24,48,72)",
    )
    parser.add_argument("--station", type=str, default=None, help="Single station to train on (default: all)")
    parser.add_argument("--trials", type=int, default=100, help="Number of optuna trials (default: 100)")
    parser.add_argument("--start", type=str, default=None, help="Data start date YYYY-MM-DD (default: 5 years ago)")
    parser.add_argument("--end", type=str, default=None, help="Data end date YYYY-MM-DD (default: yesterday)")
    parser.add_argument("--backend", type=str, default="lightgbm",
                        choices=["xgboost", "lightgbm", "lightgbm_hi"],
                        help="Model backend to train (default: lightgbm)")
    parser.add_argument("--target-kind", type=str, default="th",
                        choices=["hi", "th"],
                        dest="target_kind",
                        help="Prediction target: hi=direct heat-index, th=temp+rh two-head (default: th)")
    parser.add_argument("--gate-backend", type=str, default="lightgbm",
                        choices=["lightgbm", "brf"],
                        dest="gate_backend",
                        help="DangerGate backend: lightgbm (default) or brf (BalancedRandomForest)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel station training workers (default: 1). "
                             "GPU training with workers>1 may cause device contention.")
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run id for logs/eval/runs/<run_id>. Default: UTC timestamp.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain slots even when app/models/forecast_<ver>/{station}/h{H}/bundle.json already exists.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="gpu",
        choices=["auto", "cpu", "gpu", "cuda"],
        help="Training device preference (default: gpu).",
    )
    parser.add_argument(
        "--model-version", type=str, default="v3",
        dest="model_version",
        help="Registry layout version under app/models/forecast_<ver>/ (default: v3).",
    )
    parser.add_argument(
        "--backends",
        default="lgbm",
        help="Comma-separated list of backends to train: lgbm,catboost,xgb (default: lgbm)",
    )
    args = parser.parse_args()
    os.environ["HEATSHIELD_FORECAST_VERSION"] = args.model_version
    _configure_training_device(args.device)

    end_date = date.fromisoformat(args.end) if args.end else date.today() - timedelta(days=1)
    start_date = date.fromisoformat(args.start) if args.start else end_date - timedelta(days=365 * 5)

    # Resolve horizons list — legacy --horizon overrides --horizons
    if args.horizon is not None:
        horizons = [args.horizon]
        legacy_single = True
        logger.info("Legacy --horizon mode: training single horizon=%dh", args.horizon)
    else:
        horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]
        legacy_single = len(horizons) == 1
        logger.info("Multi-horizon mode: training horizons=%s", horizons)

    station_ids = [args.station] if args.station else list(STATIONS.keys())
    logger.info("Training for stations=%s trials=%d", station_ids, args.trials)
    try:
        from app.ml.forecast.train import _gpu_xgb_params
        logger.info("XGBoost device mode: %s", _gpu_xgb_params().get("device", "cpu"))
    except Exception:
        pass
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    eval_run_dir = Path("logs") / "eval" / "runs" / run_id
    leaderboard: list[dict] = []
    leaderboard_lock = Lock()
    os.environ["HEATSHIELD_RUN_ID"] = run_id
    write_heartbeat(run_id, phase="start", extra={"model_version": args.model_version})

    def _record_leaderboard_row(row: dict) -> None:
        with leaderboard_lock:
            leaderboard.append(row)
            _write_leaderboard(eval_run_dir, leaderboard)
            drive_root = os.environ.get("HEATSHIELD_DRIVE_ROOT")
            if drive_root:
                try:
                    import shutil
                    dst = Path(drive_root) / "logs" / "train" / run_id
                    dst.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(eval_run_dir / "leaderboard.json", dst / "leaderboard.json")
                    shutil.copy2(eval_run_dir / "leaderboard.md", dst / "leaderboard.md")
                except Exception:
                    pass

    if args.backend in {"lightgbm", "lightgbm_hi"}:
        from app.ml.forecast.backends.lgbm_backend import LGBMDirectHIForecaster, LGBMForecaster
        from app.ml import registry
        target_kind = "hi" if args.backend == "lightgbm_hi" else args.target_kind

        from app.ml.forecast.backends.lgbm_backend import _compute_hi_array

        # GPU workers contend for the same device memory in Colab. Force serial
        # station scheduling instead of letting multiple processes OOM mid-run.
        if args.workers > 1 and os.getenv("HEATSHIELD_FORCE_CPU") != "1":
            from app.ml.forecast.train import _gpu_xgb_params
            from app.ml.forecast.backends.lgbm_backend import _detect_device as _lgbm_detect_device
            if _gpu_xgb_params().get("device") == "cuda" or _lgbm_detect_device() in {"cuda", "gpu"}:
                logger.warning(
                    "--workers %d requested with GPU training. Forcing --workers 1 to avoid GPU memory contention.",
                    args.workers,
                )
                args.workers = 1

        def _train_station(sid: str) -> list[dict]:
            """Train all horizons for one station; returns leaderboard rows."""
            rows: list[dict] = []
            obs = read_observations(sid, start_date, end_date)
            if len(obs) < 500:
                logger.warning("Too few rows (%d) for station=%s — skipping", len(obs), sid)
                return rows

            # Build features ONCE per station — avoids rebuilding X for every horizon
            X_full, df_aug = build_X_once(obs)

            for h in horizons:
                current_slot = f"{sid}:h{h}"
                write_heartbeat(run_id, current_slot=current_slot, phase="slot_start")
                slot_dir = Path("app") / "models" / f"forecast_{args.model_version}" / sid / f"h{h}"
                bundle_path = slot_dir / "bundle.json"
                registry_path = slot_dir / "registry.json"
                if not args.force and bundle_path.exists() and registry_path.exists():
                    logger.warning(
                        "SKIP station=%s h=%d — bundle already exists. Pass --force to retrain. (%s)",
                        sid,
                        h,
                        slot_dir,
                    )
                    row = _row_from_existing_slot(slot_dir, sid, h)
                    rows.append(row)
                    _record_leaderboard_row(row)
                    write_heartbeat(run_id, current_slot=current_slot, phase="skipped_existing")
                    continue

                logger.info("Training LightGBM backend: station=%s horizon=%d target=%s",
                            sid, h, target_kind)
                y_full = build_y_for_horizon(df_aug, X_full.index, h, target_kind=target_kind)

                # Apply valid mask (y must be non-null)
                if isinstance(y_full, pd.DataFrame):
                    valid = y_full.notna().all(axis=1)
                else:
                    valid = y_full.notna()
                X = X_full[valid]
                y = y_full[valid]

                if len(X) < 500:
                    logger.warning(
                        "Too few rows (%d) after target NaN-drop for station=%s h=%d — skipping",
                        len(X), sid, h,
                    )
                    continue
                if target_kind == "th":
                    y_hi_for_audit = _compute_hi_array(y["temp_c"].values, y["rh"].values)
                else:
                    y_hi_for_audit = y.values
                danger_support = int((y_hi_for_audit >= 42.0).sum())
                if danger_support < 20:
                    logger.warning(
                        "Low danger support for station=%s h=%d: only %d rows with HI>=42C",
                        sid, h, danger_support,
                    )

                forecaster_cls = LGBMDirectHIForecaster if args.backend == "lightgbm_hi" else LGBMForecaster
                forecaster = forecaster_cls(
                    n_trials=args.trials,
                    gate_backend=args.gate_backend,
                )
                write_heartbeat(run_id, current_slot=current_slot, phase="fit")
                forecaster.fit(X, y, station_id=sid, horizon_h=h)

                split = split_xy(X, y, horizon_h=h)
                bundle = forecaster.predict_with_pi(split.X_test)
                if target_kind == "th":
                    y_true = _compute_hi_array(split.y_test["temp_c"].values, split.y_test["rh"].values)
                else:
                    y_true = split.y_test.values
                out_dir = eval_run_dir / forecaster.backend_name / sid / f"h{h}"
                eval_metrics = evaluate_predictions(
                    y_true,
                    bundle.hi_mean,
                    split.X_test,
                    out_dir=out_dir,
                    horizon_h=h,
                    station_labels={0: sid},
                    q05=bundle.hi_lower,
                    q95=bundle.hi_upper,
                    runtime={"train_seconds": forecaster._metadata.get("train_seconds", 0.0)},
                    split_metadata=split.metadata,
                    station=sid,
                    eval_dpi=80,
                )
                write_heartbeat(run_id, current_slot=current_slot, phase="eval")
                status = _quality_status(eval_metrics)
                metrics = {
                    "run_id": run_id,
                    "status": status,
                    "n_train_rows": split.metadata["row_counts"]["train"],
                    "horizons": horizons,
                    "evaluation": eval_metrics,
                }
                new_mae = eval_metrics.get("regression", {}).get("mae", float("inf"))
                champion_mae = _read_champion_mae(slot_dir)
                champion_kept = False
                # When --force is set, bypass champion-challenger: the old champion's
                # MAE may be optimistic (trained before the val-leakage fix), so a
                # head-to-head comparison would incorrectly reject the corrected model.
                if not args.force and champion_mae is not None and new_mae >= champion_mae:
                    logger.info(
                        "Champion-challenger: keeping existing model (champion MAE=%.4f <= challenger MAE=%.4f) "
                        "for station=%s h=%d",
                        champion_mae, new_mae, sid, h,
                    )
                    champion_kept = True
                else:
                    registry.save_model_v3(forecaster, metrics, sid, h)
                    logger.info(
                        "Champion-challenger: saved challenger (MAE=%.4f%s) for station=%s h=%d",
                        new_mae,
                        f" < champion {champion_mae:.4f}" if champion_mae is not None else " — new slot",
                        sid, h,
                    )

                # When champion is kept, surface its stored metrics so the leaderboard
                # reflects the model that is actually on disk, not the rejected challenger.
                row_eval = eval_metrics
                row_status = status
                if champion_kept:
                    try:
                        champ_data = json.loads((slot_dir / "registry.json").read_text(encoding="utf-8"))
                        row_eval = champ_data.get("evaluation", eval_metrics)
                        row_status = champ_data.get("status", status)
                    except Exception:
                        pass

                row = {
                    "backend": forecaster.backend_name,
                    "station": sid,
                    "horizon_h": h,
                    "mae": row_eval.get("regression", {}).get("mae",
                             eval_metrics.get("regression", {}).get("mae", float("nan"))),
                    "skill_score": row_eval.get("baselines", {}).get("skill_score",
                                     eval_metrics.get("baselines", {}).get("skill_score", float("nan"))),
                    "danger_recall_42": row_eval.get("safety", {}).get("danger_42", {}).get("recall",
                                          eval_metrics.get("safety", {}).get("danger_42", {}).get("recall")),
                    "status": row_status,
                    "eval_dir": str(out_dir),
                    **({"champion_kept": True} if champion_kept else {}),
                }
                rows.append(row)
                _record_leaderboard_row(row)
                write_heartbeat(run_id, current_slot=current_slot, phase="slot_done")
                logger.info("Saved LightGBM v3: station=%s h=%d", sid, h)
                del forecaster, bundle, split
                gc.collect()

                # Optional CatBoost training when --backends includes "catboost"
                backends_to_train = [b.strip() for b in getattr(args, "backends", "lgbm").split(",")]
                if "catboost" in backends_to_train:
                    try:
                        from app.ml.forecast.backends.catboost_backend import CatBoostForecaster
                        logger.info("Training CatBoost backend: station=%s horizon=%d", sid, h)
                        catboost_forecaster = CatBoostForecaster(
                            n_trials=getattr(args, "trials", 10),
                            random_state=42,
                        )
                        catboost_forecaster.fit(X, y, station_id=sid, horizon_h=h)
                        catboost_path = slot_dir / "catboost"
                        catboost_forecaster.save(catboost_path)
                        logger.info("CatBoost model saved to %s", catboost_path)
                        del catboost_forecaster
                        gc.collect()
                    except Exception as exc:
                        logger.warning("CatBoost training failed (%s) — skipping", exc)
            return rows

        if args.workers > 1:
            logger.info("Parallel station training: workers=%d stations=%s", args.workers, station_ids)
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(_train_station, sid): sid for sid in station_ids}
                for fut in as_completed(futures):
                    try:
                        fut.result()
                    except Exception as exc:
                        logger.error("Station %s training failed: %s", futures[fut], exc)
        else:
            for sid in station_ids:
                _train_station(sid)

        _write_leaderboard(eval_run_dir, leaderboard)
        logger.info("LightGBM v3 training complete")
        try:
            summary_path = _generate_summary_chart(eval_run_dir, leaderboard)
            logger.info("Summary chart: %s", summary_path.resolve())
        except Exception as e:
            logger.warning("Summary chart failed: %s", e)
        try:
            _export_run_to_versioned_dir(eval_run_dir)
        except Exception as e:
            logger.warning("versioned export failed: %s", e)
        _display_eval_charts(eval_run_dir)

        return  # don't fall through to XGBoost path

    # Load and concatenate observations from all stations
    frames: list[pd.DataFrame] = []
    for sid in station_ids:
        df = read_observations(sid, start_date, end_date)
        if df.empty:
            logger.warning("No data found for station=%s — skipping", sid)
            continue
        df["station_id"] = sid
        frames.append(df)

    if not frames:
        logger.error("No data loaded. Run scripts/build_historical.py first to ingest data.")
        sys.exit(1)

    all_data = pd.concat(frames, ignore_index=True)
    logger.info("Loaded %d total observations across %d stations", len(all_data), len(frames))

    # ------------------------------------------------------------------ #
    # Train each horizon
    # ------------------------------------------------------------------ #
    all_horizon_results: dict[int, dict] = {}
    for horizon_h in horizons:
        write_heartbeat(run_id, current_slot=f"all:h{horizon_h}", phase="slot_start")
        logger.info("=" * 50)
        logger.info("Training horizon = %dh", horizon_h)

        X, y = build_features(
            all_data.copy(),
            horizon_h=horizon_h,
            lags_h=_DEFAULT_LAGS_H,
            rolling_h=_DEFAULT_ROLLING_H,
        )
        logger.info("Feature matrix shape for h%d: %s", horizon_h, X.shape)

        if len(X) < 500:
            logger.error(
                "Only %d training rows for horizon=%dh after feature engineering. "
                "Need at least 500. Ingest more historical data first.",
                len(X), horizon_h,
            )
            sys.exit(1)

        result = train(X, y, station_id=",".join(station_ids), horizon_h=horizon_h, n_trials=args.trials)
        write_heartbeat(run_id, current_slot=f"all:h{horizon_h}", phase="eval")
        all_horizon_results[horizon_h] = result
        logger.info("h%d metrics: %s", horizon_h, _fmt_metrics(result["metrics"]))
        preds = result["test_predictions"]
        test_data = result["test_data"]
        out_dir = eval_run_dir / "xgboost_hi" / "all" / f"h{horizon_h}"
        eval_metrics = evaluate_predictions(
            test_data["y"].values,
            preds["mean"],
            test_data["X"],
            out_dir=out_dir,
            horizon_h=horizon_h,
            station_labels={i: sid for i, sid in enumerate(sorted(station_ids))},
            q05=preds.get("q05"),
            q95=preds.get("q95"),
            runtime={"train_seconds": result["training"]["train_seconds"]},
            split_metadata=result["split_metadata"],
            eval_dpi=80,
        )
        status = _quality_status(eval_metrics)
        result["evaluation"] = eval_metrics
        result["status"] = status
        row = {
            "backend": "xgboost_hi",
            "station": "all",
            "horizon_h": horizon_h,
            "mae": eval_metrics["regression"]["mae"],
            "skill_score": eval_metrics["baselines"]["skill_score"],
            "danger_recall_42": eval_metrics["safety"]["danger_42"]["recall"],
            "status": status,
            "eval_dir": str(out_dir),
        }
        _record_leaderboard_row(row)
        write_heartbeat(run_id, current_slot=f"all:h{horizon_h}", phase="slot_done")
        gc.collect()

    logger.info("=" * 50)

    # ------------------------------------------------------------------ #
    # Save model
    # ------------------------------------------------------------------ #
    if legacy_single:
        # Single-horizon save (original v2 single-horizon behavior)
        only_h = horizons[0]
        result = all_horizon_results[only_h]
        feature_medians = result["feature_medians"]
        metadata = {
            "horizon_h": only_h,
            "stations": station_ids,
            "data_window": {"start": start_date.isoformat(), "end": end_date.isoformat()},
            "n_train_rows": len(X),
            "feature_list": result["feature_list"],
            "feature_medians": feature_medians,
            "hyperparams": result["hyperparams"],
            "metrics": result["metrics"],
            "evaluation": result["evaluation"],
            "status": result["status"],
            "git_sha": _git_sha(),
            "roles": ["mean", "q05", "q95"],
            "seeds": [42, 123, 7],
        }
        ensemble_payload = {"all_boosters": result["all_boosters"]}
        model_path = save_model(ensemble_payload, metadata, kind="forecast")
    else:
        # Multi-horizon save
        horizons_payload = {
            "horizons": {
                h: {"all_boosters": all_horizon_results[h]["all_boosters"]}
                for h in horizons
            }
        }

        # Use h24 as primary for backward compat, fall back to first horizon
        h24_result = all_horizon_results.get(24, next(iter(all_horizon_results.values())))
        primary_h = 24 if 24 in all_horizon_results else horizons[0]

        metadata = {
            "horizon_h": primary_h,
            "horizons": horizons,
            "stations": station_ids,
            "data_window": {"start": start_date.isoformat(), "end": end_date.isoformat()},
            "n_train_rows": sum(len(all_data) for _ in horizons),  # approximate
            "feature_list": h24_result["feature_list"],
            "feature_medians": h24_result["feature_medians"],
            "hyperparams": h24_result["hyperparams"],
            "metrics": h24_result["metrics"],
            "evaluation": h24_result["evaluation"],
            "status": h24_result["status"],
            "git_sha": _git_sha(),
            "roles": ["mean", "q05", "q95"],
            "seeds": [42, 123, 7],
            "per_horizon_metrics": {h: all_horizon_results[h]["metrics"] for h in horizons},
            "per_horizon_evaluation": {h: all_horizon_results[h]["evaluation"] for h in horizons},
        }
        model_path = save_model(horizons_payload, metadata, kind="forecast")

    logger.info("Model saved to %s", model_path)
    _write_leaderboard(eval_run_dir, leaderboard)
    logger.info("Evaluation artifacts saved to %s", eval_run_dir)

    # ------------------------------------------------------------------ #
    # Generate visualizations (use primary / only horizon)
    # ------------------------------------------------------------------ #
    primary_h = horizons[0] if legacy_single else (24 if 24 in all_horizon_results else horizons[0])
    metadata_path = model_path / "metadata.json" if model_path.is_dir() else model_path.with_suffix(".json")
    if model_path.is_dir() and (model_path / f"h{primary_h}").exists():
        booster_path = model_path / f"h{primary_h}" / "mean_s42.ubj"
    elif model_path.is_dir() and (model_path / "mean_s42.ubj").exists():
        booster_path = model_path / "mean_s42.ubj"
    else:
        booster_path = model_path

    try:
        image_paths = generate_report(metadata_path, booster_path)
        logger.info("Generated %d visualization(s):", len(image_paths))
        for img_path in image_paths:
            logger.info("  %s", img_path)
    except Exception as e:
        logger.error("Failed to generate visualizations: %s", e)

    # ------------------------------------------------------------------ #
    # Final quality gate (use primary horizon metrics)
    # ------------------------------------------------------------------ #
    primary_result = all_horizon_results.get(primary_h, next(iter(all_horizon_results.values())))
    skill = primary_result["metrics"]["skill_score"]
    if skill >= 0.15:
        logger.info("skill_score=%.3f >= 0.15 — model is READY to ship.", skill)
    else:
        logger.warning(
            "skill_score=%.3f < 0.15 — model does NOT beat baselines enough. "
            "Do not use in production. Ingest more data or tune features.",
            skill,
        )


if __name__ == "__main__":
    main()
