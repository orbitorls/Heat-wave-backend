# Copilot instructions for Heat-wave-backend

Purpose: help Copilot sessions quickly understand how to build, run, test, and change this repository.

---

## Quick commands

- Install dependencies:
  - pip install -r requirements.txt

- Run full test suite:
  - pytest

- Run a single test (example syntax):
  - pytest tests/<file_name>.py::test_function_name

- Train model:
  - python Train_Ai.py
  - or use the unified entrypoint: python main.py --mode train --config config/config.yaml

- Run API / dashboard:
  - python api_server.py
  - or: python main.py --mode dashboard --host 0.0.0.0 --port 5000

- Evaluate model (uses latest checkpoint by default):
  - python evaluate_model.py [--checkpoint <path>] [--output-json <path>] [--verbose]

- Run a one-off prediction (CLI entrypoint):
  - python main.py --mode predict --input ./data/sample.csv [--proba]

Notes: there is no repository-provided lint command or config detected.

---

## High-level architecture (concise)

- src/ is the canonical package. Key subpackages:
  - src.data.loader — ERA5 loading, preprocessing, sequence creation (primary data API).
  - src.models — model implementations (convlstm) and model manager (load/predict).
  - src.models.manager — runtime model manager used by CLI and APIs.
  - src.api (newer FastAPI modules) and top-level Flask app bootstrap (api_server.py) — both entry patterns exist.
  - train scripts: Train_Ai.py (ConvLSTM training), train_daily_xgboost.py (daily XGBoost training).

- Model types:
  - ConvLSTM spatial sequence model (primary forecasting model).
  - XGBoost "daily" model for single-day heatwave prediction (lighter-weight inference).

- Data & artifacts:
  - era5_data/ — NetCDF ERA5 inputs
  - models/ — checkpoint files (see naming conventions below)
  - output/ — evaluation and artifact outputs

- Entrypoints:
  - main.py — unified entrypoint with modes: train, dashboard, predict, cli, web
  - api_server.py — Flask production entrypoint, registers blueprints (including daily XGBoost blueprint)
  - Train_Ai.py — legacy/primary training script invoked by main.py in train mode

---

## Key conventions & patterns

- Configs: config/config.yaml (project YAML/JSON) is accepted. main.py maps config keys from `data`, `training`, and `model` sections into Train_Ai.py runtime keys via an internal mapping function. Expect compatibility mapping rather than 1:1 passthrough.

- Deprecated root modules: top-level files like data_loader.py and heatwave_model.py are thin compatibility wrappers that re-export src.* implementations and emit DeprecationWarning. Prefer importing from src.*.

- Checkpoint naming:
  - ConvLSTM / spatial model: models/heatwave_convlstm_v{N}.pth (or similar versioned names).
  - Daily XGBoost: models/heatwave_daily_xgboost_v{N}.pth
  - XGBoost checkpoints are expected to contain a dict with key "sklearn_model" and optional metadata (normalization_mean/std, temp_mean_celsius, heatwave_temp_threshold).

- Model loading:
  - src.models.manager provides load_model and inference helpers used by CLI and API. main.py and api_server.py call into these managers.
  - main.py --mode predict requires --input (CSV). It converts numeric columns to numpy arrays and tries predict_proba/predict.

- API / routes:
  - Daily XGBoost endpoints implemented in api_daily_predict.py blueprint:
    - POST /api/daily/predict
    - GET /api/daily/model_info
    - GET /api/daily/health
  - api_server.py registers blueprints and exposes the Flask app for production.

- Tests:
  - Tests are run with pytest. Test modules live under tests/ (conftest.py, test_model.py, etc.). Use pytest path::test_name to run a single test.

- Packaging / dependencies:
  - requirements.txt is the source of truth for Python libs; install into a virtualenv matching the Python version used by CI/developer.

---

## What to look for when making changes

- Prefer src/ paths (not top-level compatibility modules).
- When adding a new model checkpoint format, update model_manager and XGBoost load helpers to read/write the same metadata keys (feature names, normalization stats, threshold).
- If adding new CLI flags or config keys, update main.py's _extract_training_config mapping so PROJECT_CONTEXT-style config keys map into Train_Ai.py.

---

If this file exists already, append the missing items above rather than replacing the whole file.

---

(Generated automatically to help future Copilot sessions.)
