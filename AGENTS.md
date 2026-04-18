# PROJECT KNOWLEDGE BASE

**Generated:** 2026-03-06 20:37:00 +07:00
**Commit:** (git unavailable)
**Branch:** main

## OVERVIEW
Python backend for heatwave forecasting over Thailand. Pipeline is ERA5 NetCDF ingestion -> ConvLSTM training -> Flask inference API.

## STRUCTURE
```text
./
|- api_server.py       # primary API entrypoint (Flask)
|- Train_Ai.py         # model training + checkpoint versioning
|- data_loader.py      # ERA5 crop/merge/normalize + sequence builder
|- heatwave_model.py   # ConvLSTM + physics-informed loss
|- download_era5.py    # CDS API downloader for raw ERA5 files
|- main.py             # legacy alternate API server (avoid)
|- heatwave_cli.py     # NEW: unified CLI (studio/control/trainer modes)
|- requirements.txt    # UTF-16 encoded (!)
|- era5_data/          # large NetCDF inputs (binary)
|- models/             # .pth checkpoints (binary)
`- output/             # generated artifacts (images)
```

## WHERE TO LOOK
| Task | Location | Notes |
|---|---|---|
| API routes and response schema | `api_server.py` | `/api/predict`, `/api/forecast`, `/api/map`, `/api/health` |
| Training flow and checkpoint save policy | `Train_Ai.py` | `get_next_version`, train loop, save naming |
| Data preprocessing and tensor shape contracts | `data_loader.py` | Returns `(Time, C, H, W)` then sequence windows |
| Model architecture and custom loss | `heatwave_model.py` | `HeatwaveConvLSTM`, `PhysicsInformedLoss` |
| ERA5 acquisition process | `download_era5.py` | Uses CDS API and writes to `era5_data/` |
| Legacy/alternate serving path | `main.py` | Avoid — overlaps api_server.py |
| Unified CLI | `heatwave_cli.py` | studio/control/trainer interactive modes |

## CODE MAP
| Symbol | Type | Location | Refs | Role |
|---|---|---|---|---|
| `HeatwaveConvLSTM` | class | `heatwave_model.py` | high | Core sequence model used by training and API |
| `PhysicsInformedLoss` | class | `heatwave_model.py` | medium | MSE + thermodynamic consistency penalty |
| `ConvLSTMCell` | class | `heatwave_model.py` | high | Base ConvLSTM cell |
| `load_era5_data` | function | `data_loader.py` | high | NetCDF ingestion, crop, merge, normalize |
| `create_sequences` | function | `data_loader.py` | high | Builds supervised windows for forecasting |
| `clean_data` | function | `data_loader.py` | medium | NaN fill + outlier clipping |
| `main` | function | `Train_Ai.py` | medium | End-to-end training orchestration |
| `load_resources` | function | `api_server.py` | medium | Loads data/checkpoint and initializes globals |
| `predict_summary` | function | `api_server.py` | medium | Dashboard-oriented prediction response |
| `forecast_summary` | function | `api_server.py` | medium | 7-day forecast payload (partially mocked) |
| `predict_map` | function | `api_server.py` | medium | GeoJSON polygon generation from grid |
| `get_latest_model` | function | `api_server.py` | medium | Checkpoint version discovery |
| `download_era5_data` | function | `download_era5.py` | low | Bulk historical dataset download |

## CONVENTIONS
- Flat module layout at repo root; no `src/` package split.
- Model checkpoints follow `heatwave_model_checkpoint_v{N}.pth`.
- Data directory constants are plain strings (`era5_data`, `models`) and expected relative to repo root.
- API code uses module-level globals for loaded model/data state (anti-pattern).
- Mixed language comments (English + Thai) in serving scripts.
- Ruff linter present (`.ruff_cache/` exists) but no config file.
- Train_Ai.py supports env-based config: `HW_BATCH_SIZE`, `HW_SEQ_LEN`, `HW_EPOCHS`, etc.

## ANTI-PATTERNS (THIS PROJECT)
- Do not commit real credentials or tokens; remove inline key examples from scripts before sharing.
- Do not expand production API behavior in `main.py`; align changes to `api_server.py` first.
- Do not change tensor shape order implicitly; keep `(Batch, Time, Channels, H, W)` contract.
- Do not place generated NetCDF/checkpoint artifacts in code review diffs unless explicitly required.
- Avoid module-level mutable globals in api_server.py (use app context or singletons).
- Do not mix Thai/English comments in new code; use English consistently.

## UNIQUE STYLES
- Risk classification thresholds are simple rule-based cutoffs on predicted max temperature.
- Forecast endpoint currently mixes model output (day 1) with synthetic variation (days 2-7).
- Data loader prioritizes spatial crop early to reduce memory before merge.
- heatwave_cli.py provides three interactive modes: studio (key shortcuts), control (arrow menu), trainer (training config UI).
- API has fallback for PyTorch 2.6+ `weights_only=True` default via `_safe_torch_load`.

## COMMANDS
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python download_era5.py
python Train_Ai.py
python api_server.py
python main.py

# NEW CLI
python heatwave_cli.py --help
python heatwave_cli.py studio     # interactive TUI
python heatwave_cli.py control    # arrow-key menu
python heatwave_cli.py trainer    # training config UI
```

## NOTES
- `requirements.txt` is UTF-16 encoded — may cause issues on some systems.
- Repo contains large binary assets under `era5_data/` and `models/`; avoid broad recursive tooling.
- No tests or CI workflow files present.
- Ruff linter present but no pyproject.toml config.
- heatwave_cli.py is 1237 lines — consider modularizing if it grows further.
