# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

HeatShield AI backend — Python 3.10+ FastAPI service that converts weather observations into heat-health risk scores, what-if scenarios, action cards, and 6/24/48 h heat-index forecasts for five hard-coded Thai TMD stations (`app/data/stations.py`). README and inline comments are bilingual (Thai + English); preserve that style when editing existing prose.

`AGENTS.md` is the human contributor guide; this file is the operational complement for Claude Code.

## Commands

```powershell
# Setup
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Run the API (pre-warms v3 forecasters at startup)
uvicorn app.main:app --reload

# Tests — pyproject sets `addopts = "-ra -m 'not slow'"`,
# so plain `pytest` skips anything marked @pytest.mark.slow.
pytest                                  # default suite (fast, no model/data needed)
pytest -m slow                          # tests that need ingested data + trained models
pytest tests/test_features.py::test_x   # single test

# Demo + data + training (in usual order)
python scripts/seed_demo_data.py        # synthetic scenario walk-through
python scripts/ingest_all.py            # ingest TMD + ERA5 + NASA POWER + MODIS
python scripts/train_forecast.py        # multi-horizon (default trains 6/12/24/48/72)
python scripts/train_forecast.py --horizon 24 --station BKK_01 --trials 30
python scripts/evaluate_model.py
python scripts/viz_training_results.py
```

Configuration lives in `.env` (see `.env.example`): `CDSAPI_KEY`, `TMD_API_KEY`/`TMD_API_TOKEN`, `HEATSHIELD_ENV`, `HEATSHIELD_LOG_LEVEL`. Two TMD base URLs coexist in the example (`/nwpapi/v1` vs `/api/Weather/v1`) — clients in `app/data/` pick the right one; do not collapse them.

## Architecture

**Layered packages.** Keep the boundaries strict:

- `app/api/` — FastAPI routers only; request handling, validation, HTTP errors. One module per `/heat-index`, `/events`, `/risk`, `/whatif`, `/action-card`, `/forecast` route group.
- `app/core/` — pure, ML-free domain logic (Rothfusz heat index, adaptive percentile event detection, vulnerability profiles, risk scoring, what-if simulator, action-card generator). New deterministic calculations belong here as small typed functions.
- `app/data/` — external clients (`tmd_client`, `era5_client`, `nasa_power_client`, `modis_client`), Pydantic schemas, parquet loaders, the station registry, data-quality utilities.
- `app/ml/` — forecasting features, training, prediction, viz, and the model registry. Backend implementations live under `app/ml/forecast/backends/` (LightGBM, XGBoost; TabPFN is loaded lazily by name).

`app/main.py` mounts each router with its prefix and runs a startup hook that pre-loads v3 h=24 forecasters for every station via `registry.load_latest_v3`; missing artifacts are tolerated (v2 fallback handles them).

**Model registry has three coexisting layouts** (`app/ml/registry.py`). When touching anything that loads/saves models, account for all three:

1. **v1** — single XGBoost booster: `app/models/forecast_v{n}.ubj` + `forecast_v{n}.json`.
2. **v2** — ensemble dir `app/models/forecast_v{n}/`. Either single-horizon (9 boosters: 3 roles × 3 seeds at the dir root) or multi-horizon (`h6/`, `h12/`, `h24/`, `h48/`, `h72/` subdirs, each holding the same 9 files), plus `metadata.json`.
3. **v3** — model-agnostic backends per `(station_id, horizon_h)` under `app/models/forecast_v3/{station}/h{H}/`. `choice_matrix.json` at the v3 root maps station+horizon to a backend name (`lightgbm_quantile`, `lightgbm_hi_quantile`, `xgboost`, `tabpfn`); `load_latest_v3` dispatches to the right backend class. Backends own their own `bundle.json`; registry-level fields are written to a separate `registry.json` sidecar so saves do not clobber backend metadata.

Persistence uses XGBoost's native `.ubj` format (and backend-specific files for v3). The registry's docstring explicitly avoids serialization formats that allow arbitrary code execution; stick to `xgb.Booster.save_model` / `load_model` and JSON sidecars.

**Forecast prediction has a v3 → v2 → v1 fallback chain** (`app/ml/forecast/predict.py`). `predict()` first tries `load_latest_v3` per requested horizon; if every horizon resolves it returns immediately, otherwise it falls through to the v2/v1 XGBoost path. The v2 path averages an ensemble across seeds for the mean and the q05/q95 quantile boosters; v1 falls back to a flat PI from training metadata. `low_confidence` is set when the most recent observation is older than 60 minutes or the PI width exceeds 4 °C (or 1.6× the v3 median PI width). At least 26 hourly observations are required to build features.

**Feature engineering invariant** (`app/ml/forecast/features.py`): every row in `X` may only use observations with `ts_utc <= row.ts_utc`; the target uses `ts_utc + horizon_h`. Adding new lag/rolling features must preserve this no-leakage rule. `_DEFAULT_LAGS_H = [1, 3, 6, 12, 24]`, `_DEFAULT_ROLLING_H = [3, 6, 24]`. Inference also pads any extended columns missing at predict time with `metadata["feature_medians"]` so the column order matches training exactly.

**Parquet observation store** (`app/data/loaders.py`): partitioned at `data/raw/station_id={id}/date={iso}/obs.parquet`. `read_observations` concatenates a date range and, when a partition contains rows from multiple sources, **prefers ERA5 over NASA POWER for the same `(station_id, ts_utc)`** via an ordered categorical drop-duplicates. Preserve that ordering when adding new sources.

**Stations are a closed registry.** `app/data/stations.py` defines five Thai TMD stations (BKK_01, CNX_01, KKN_01, HYI_01, RYG_01). API routes validate `station_id` against `STATIONS` and 400 on miss. New stations require updating that file plus any data-ingestion scripts that iterate `STATIONS`.

## Conventions

- Pydantic v2 schemas in `app/data/schemas.py` are the single source of truth for API I/O; route modules import them, they do not redefine shapes inline.
- Tests that need real ingested parquet or trained models must be marked `@pytest.mark.slow`. The default `pytest` run must stay green without any external data.
- Treat `data/`, `logs/`, and `app/models/` as artifact-heavy paths: only modify them when a change genuinely depends on new data, models, or evaluation output.
