# Repository Guidelines

## Project Structure & Module Organization

This is a Python FastAPI backend for HeatShield AI. Core application code lives in `app/`: `app/api/` contains route modules, `app/core/` holds domain logic such as heat index, risk scoring, and what-if simulation, `app/data/` contains clients, schemas, loaders, and data quality utilities, and `app/ml/` contains forecasting features, training, prediction, and model registry code. Trained artifacts are stored under `app/models/`. Operational scripts live in `scripts/`, tests in `tests/`, documentation in `docs/`, raw/cache datasets in `data/`, and generated metrics or plots in `logs/`.

## Build, Test, and Development Commands

Set up a local environment with:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the API with `uvicorn app.main:app --reload`. Run the default test suite with `pytest`; this uses `pyproject.toml` defaults and excludes tests marked `slow`. Run slow, data/model-dependent checks with `pytest -m slow`. Common script entry points include `python scripts/seed_demo_data.py`, `python scripts/ingest_all.py`, `python scripts/train_forecast.py`, and `python scripts/evaluate_model.py`.

## Coding Style & Naming Conventions

Use Python 3.10+ and standard PEP 8 style with 4-space indentation. Prefer typed functions, Pydantic models for API schemas, and small pure functions in `app/core/` for reusable calculations. Name modules and functions with `snake_case`, classes with `PascalCase`, and constants with `UPPER_SNAKE_CASE`. Keep route modules focused on request handling; put business rules in `app/core/` or ML/data logic in the matching package.

## Testing Guidelines

Tests use `pytest` and `pytest-asyncio`. Add tests beside the existing suite as `tests/test_<feature>.py`, and name test functions `test_<behavior>`. Cover deterministic domain logic directly, mock external weather/data services, and mark tests that require real ingested data or trained models with `@pytest.mark.slow`.

## Commit & Pull Request Guidelines

This workspace does not include Git history, so follow a simple imperative convention such as `Add forecast regression test` or `Fix TMD retry handling`. Keep commits scoped to one concern. Pull requests should describe the change, list validation commands run, call out data/model artifact changes, link related issues, and include screenshots or metric plots when modifying generated visual outputs.

## Security & Configuration Tips

Do not commit `.env`, API keys, or large regenerated datasets unless explicitly required. Use `.env.example` for configuration documentation. Treat `data/`, `logs/`, and `app/models/` as artifact-heavy paths: update them only when the change depends on new data, models, or evaluation output.
