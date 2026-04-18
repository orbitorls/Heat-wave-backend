# Thailand Heatwave Upgrade Roadmap

## TL;DR
> **Summary**: Upgrade the backend from a temperature-first prototype into a research-backed heatwave forecasting system by keeping ConvLSTM as the canonical model path, adding high-value ERA5 predictors, switching evaluation to leakage-safe year-block heatwave metrics, and exposing calibrated risk outputs without breaking current API consumers.
> **Deliverables**:
> - Leakage-safe year-block training/evaluation workflow
> - Expanded configurable ERA5 variable pipeline and derived humid-heat features
> - ConvLSTM anomaly + exceedance training path with conservative architecture/loss upgrades
> - RandomForest kept as baseline only under the same evaluation harness
> - Additive API risk outputs and calibration metadata
> **Effort**: Large
> **Parallel**: YES - 3 waves
> **Critical Path**: 1 -> 2 -> 3 -> 6 -> 7 -> 8 -> 9 -> 10 -> F1-F4

## Context
### Original Request
- User asked for a detailed research review of papers/best practices, then requested a plan first.

### Interview Summary
- Repo is a Thailand heatwave backend built around ERA5 ingestion, a ConvLSTM path, a RandomForest path, and a Flask API.
- Research-backed priorities are humidity-aware features, soil-moisture-aware evaluation, percentile-based heatwave targets, probabilistic risk outputs, and stricter benchmark methodology.
- No test suite or CI exists, so verification must be command-driven and agent-executed.

### Metis Review (gaps addressed)
- Canonical path chosen: ConvLSTM is primary; RandomForest remains baseline-only.
- Guardrail added: climatology and thresholds must be computed from training-year blocks only.
- Guardrail added: API compatibility must be additive, not destructive.
- Guardrail added: channel expansion is limited to a fixed high-ROI ERA5 set in this phase; no open-ended variable creep.

## Work Objectives
### Core Objective
- Convert the current system into a decision-ready heatwave forecasting stack that predicts temperature anomalies plus heatwave exceedance risk for Thailand using a leakage-safe evaluation protocol aligned with current literature.

### Deliverables
- `download_era5.py` updated to retrieve the fixed Phase-1 variable set.
- `data_loader.py` updated for configurable channel assembly, derived humid-heat features, and train-only climatology support.
- `Train_ConvLSTM.py` upgraded into the canonical trainer with year-block splits, anomaly targets, exceedance head, calibration artifacts, and evaluation reports.
- `Train_Ai.py` retained as a baseline evaluator under the same split/metric logic.
- `heatwave_model.py` upgraded conservatively to a multiscale ConvLSTM + tail-aware physics-scheduled objective.
- `api_server.py` updated to preserve current fields while adding calibrated risk metadata and grid/map risk outputs.

### Definition of Done (verifiable conditions with commands)
- `python -c "from data_loader import load_era5_data; d,lats,lons,m,s=load_era5_data('era5_data', normalize=False); print(d.shape, len(lats), len(lons))"` prints the expected expanded channel count and completes without NaN-related exceptions.
- `python -c "from Train_ConvLSTM import build_year_block_splits; print(build_year_block_splits('era5_data'))"` returns deterministic year-block partitions with no overlap.
- `python Train_ConvLSTM.py` completes one configured training run, writes checkpoint metadata, and emits continuous + event metrics.
- `python Train_Ai.py` completes one baseline run under the same split/threshold logic and writes comparable metrics.
- `python api_server.py` starts successfully, `curl http://127.0.0.1:5000/api/health` returns `{"status":"ok"...}`, and `curl http://127.0.0.1:5000/api/predict` / `curl http://127.0.0.1:5000/api/forecast` include both legacy keys and new risk metadata.

### Must Have
- Year-block-only headline evaluation.
- Train-only climatology and percentile threshold generation.
- Fixed Phase-1 predictor set: `z500`, `t2m`, `swvl1`, `d2m`, `u10`, `v10`, `ssrd`, `strd`, `msl`, plus static geography and derived humid-heat features.
- ConvLSTM canonical training path with anomaly regression + exceedance probability output.
- RandomForest baseline retained for comparison, not deployment.
- Additive API outputs with calibrated risk metadata.

### Must NOT Have (guardrails, AI slop patterns, scope boundaries)
- No random train/validation/test splits for headline results.
- No use of full-dataset percentiles/climatology during training or evaluation.
- No replacement of `/api/predict` or `/api/forecast` response keys; only additive nested fields.
- No external datasets in this phase (no station data, no MJO/ENSO feeds, no urban satellite layers).
- No full architecture rewrite to GraphCast/Pangu/FourCastNet-class models.
- No implicit tensor reordering away from `(Batch, Time, Channels, H, W)`.

## Verification Strategy
> ZERO HUMAN INTERVENTION - all verification is agent-executed.
- Test decision: tests-after + command-driven checks; add lightweight deterministic unit-style coverage only for new split/label/calibration helpers.
- QA policy: Every task includes agent-executed scenarios.
- Evidence: `.sisyphus/evidence/task-{N}-{slug}.{ext}`

## Execution Strategy
### Parallel Execution Waves
> Target: 5-8 tasks per wave. <3 per wave (except final) = under-splitting.
> Extract shared dependencies as Wave-1 tasks for max parallelism.

Wave 1: task 1 foundation, task 2 data acquisition, task 3 split/climatology, task 4 labels/metrics, task 5 baseline harness
Wave 2: task 6 loader/channel refactor, task 7 ConvLSTM architecture, task 8 loss/objective, task 9 calibration/checkpoints
Wave 3: task 10 API/runtime integration, task 11 verification/reporting cleanup

### Dependency Matrix (full, all tasks)
- 1 blocks 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
- 2 blocks 6
- 3 blocks 4, 5, 8, 9, 11
- 4 blocks 5, 8, 9, 10, 11
- 5 blocks 11
- 6 blocks 7, 8, 9, 10, 11
- 7 blocks 8, 9, 10, 11
- 8 blocks 9, 10, 11
- 9 blocks 10, 11
- 10 blocks 11

### Agent Dispatch Summary (wave -> task count -> categories)
- Wave 1 -> 5 tasks -> `deep`, `unspecified-high`, `quick`
- Wave 2 -> 4 tasks -> `deep`, `unspecified-high`
- Wave 3 -> 2 tasks -> `unspecified-high`, `quick`

## TODOs
> Implementation + Test = ONE task. Never separate.
> EVERY task MUST have: Agent Profile + Parallelization + QA Scenarios.

- [ ] 1. Establish the canonical experiment contract

  **What to do**: Make `Train_ConvLSTM.py` the canonical training/inference path, keep `Train_Ai.py` as a baseline-only evaluator, and introduce one shared configuration contract for variable set, split mode, threshold percentile, persistence length, calibration toggle, and output report paths. Update runtime metadata so `api_server.py` can detect the upgraded ConvLSTM checkpoints without guessing channel semantics.
  **Must NOT do**: Do not delete `Train_Ai.py`, do not remove current checkpoint compatibility, and do not introduce random split defaults.

  **Recommended Agent Profile**:
  - Category: `deep` - Reason: This task fixes repo-wide control flow and prevents downstream ambiguity.
  - Skills: `[]` - No extra skill is required beyond repo analysis.
  - Omitted: `playwright` - No browser interaction is needed.

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 | Blocked By: none

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `README.md:13` - README explicitly marks `Train_ConvLSTM.py` as the recommended training entrypoint.
  - Pattern: `README.md:33` - Current documented training command is ConvLSTM-first.
  - Pattern: `Train_ConvLSTM.py:36` - Existing ConvLSTM training entrypoint.
  - Pattern: `Train_Ai.py:216` - Existing RandomForest training path to keep as baseline-only.
  - Pattern: `api_server.py:97` - Runtime resource loading depends on checkpoint metadata.

  **Acceptance Criteria** (agent-executable only):
  - [ ] `python -c "import Train_ConvLSTM, Train_Ai; print(hasattr(Train_ConvLSTM,'train'), hasattr(Train_Ai,'train'))"` confirms both paths still exist.
  - [ ] The canonical trainer exposes a single config source for split mode, variable set, threshold percentile, persistence length, and calibration toggle.
  - [ ] Checkpoint metadata written by the canonical path includes model type, channel contract, split description, threshold definition, and calibration metadata placeholders.

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```
  Scenario: Canonical path detection
    Tool: Bash
    Steps: Run `python -c "import torch, Train_ConvLSTM; print('ok')"` and inspect canonical config export points.
    Expected: ConvLSTM path imports cleanly and exposes the agreed config contract.
    Evidence: .sisyphus/evidence/task-1-canonical-contract.txt

  Scenario: Baseline path preserved
    Tool: Bash
    Steps: Run `python -c "import Train_Ai; print('baseline-ok')"`.
    Expected: RandomForest path still imports and remains callable for baseline evaluation only.
    Evidence: .sisyphus/evidence/task-1-canonical-contract-baseline.txt
  ```

  **Commit**: NO | Message: `chore(train): define canonical experiment contract` | Files: `Train_ConvLSTM.py`, `Train_Ai.py`, `api_server.py`

- [ ] 2. Expand ERA5 acquisition to the fixed Phase-1 variable set

  **What to do**: Update `download_era5.py` so Phase-1 downloads fetch the fixed variable list: existing `2m_temperature`, `volumetric_soil_water_layer_1`, and `geopotential@500hPa`, plus `2m_dewpoint_temperature`, `10m_u_component_of_wind`, `10m_v_component_of_wind`, `surface_solar_radiation_downwards`, `surface_thermal_radiation_downwards`, and `mean_sea_level_pressure`. Keep the same Thailand bounding box and hourly selection strategy already used in the repo.
  **Must NOT do**: Do not add external data sources, do not widen the geographic crop, and do not introduce additional pressure levels in this phase.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` - Reason: The work is straightforward but must be exact because it affects data availability and memory cost.
  - Skills: `[]` - No special skill is needed.
  - Omitted: `playwright` - No browser interaction is needed.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 6 | Blocked By: 1

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `download_era5.py:73` - Existing downloader entrypoint.
  - Pattern: `download_era5.py:101` - Current single-level request block to extend.
  - Pattern: `download_era5.py:136` - Current pressure-level request block to preserve for `z500`.
  - External: `https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation` - Authoritative ERA5 variable/product guidance.

  **Acceptance Criteria** (agent-executable only):
  - [ ] The downloader code clearly enumerates the fixed Phase-1 single-level variables and preserves the existing bounding box `[21, 97, 5, 106]`.
  - [ ] The pressure-level request still downloads `geopotential` at 500 hPa and does not expand to extra levels in this phase.
  - [ ] Dry-run inspection of request payloads shows no unsupported variable names.

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```
  Scenario: Single-level payload inspection
    Tool: Bash
    Steps: Run `python -c "import download_era5, inspect; print(inspect.getsource(download_era5.download_era5_data))"` and capture the variable lists.
    Expected: All fixed Phase-1 variables appear exactly once in the retrieval logic.
    Evidence: .sisyphus/evidence/task-2-era5-payload.txt

  Scenario: Bounding box preserved
    Tool: Bash
    Steps: Run `python -c "import download_era5, inspect; src=inspect.getsource(download_era5.download_era5_data); print('[21, 97, 5, 106]' in src)"`.
    Expected: Output is `True` and no alternate area is introduced.
    Evidence: .sisyphus/evidence/task-2-era5-area.txt
  ```

  **Commit**: NO | Message: `feat(data): expand era5 phase-1 variables` | Files: `download_era5.py`

- [ ] 3. Implement leakage-safe year-block splits and train-only climatology

  **What to do**: Replace ratio-based headline evaluation with year-block splitting built from file years or decoded timestamps. Compute day-of-year climatology and percentile thresholds from training years only, using a +/-7 day smoothing window around each day-of-year for stability. Persist these artifacts in checkpoint metadata so evaluation and inference use the same definition.
  **Must NOT do**: Do not compute climatology from validation/test years, do not use random shuffling for headline evaluation, and do not drop the existing temporal split helper until the new path fully replaces it.

  **Recommended Agent Profile**:
  - Category: `deep` - Reason: This task changes the correctness model for every reported metric.
  - Skills: `[]` - No special skill is needed.
  - Omitted: `playwright` - No browser interaction is needed.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 4, 5, 8, 9, 11 | Blocked By: 1

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `Train_Ai.py:99` - Existing temporal split helper to replace as the headline logic.
  - Pattern: `Train_Ai.py:307` - Current threshold uses a simple train percentile; this must become year-block + day-of-year-aware.
  - Pattern: `data_loader.py:95` - File naming/year scanning already begins here.
  - External: `https://weatherbench2.readthedocs.io/en/latest/` - Benchmark philosophy for leakage-safe weather evaluation.
  - External: `https://www.wcrp-climate.org/etccdi` - ETCCDI percentile-style climate extremes definitions.

  **Acceptance Criteria** (agent-executable only):
  - [ ] A helper returns deterministic train/validation/test year partitions with zero overlap.
  - [ ] Climatology and percentile thresholds are computed from training years only and saved in checkpoint/report metadata.
  - [ ] A command-line or importable check can print the year partitions and threshold baseline without reading validation/test years into the baseline calculation.

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```
  Scenario: Year-block split integrity
    Tool: Bash
    Steps: Run `python -c "from Train_ConvLSTM import build_year_block_splits; print(build_year_block_splits('era5_data'))"`.
    Expected: Output lists non-overlapping year groups and is deterministic across repeated runs.
    Evidence: .sisyphus/evidence/task-3-year-splits.txt

  Scenario: Climatology leakage guard
    Tool: Bash
    Steps: Run a helper command that prints the years used for climatology generation.
    Expected: Only training years appear in the climatology/percentile artifact report.
    Evidence: .sisyphus/evidence/task-3-climatology-years.txt
  ```

  **Commit**: NO | Message: `feat(eval): add year-block climatology splits` | Files: `Train_ConvLSTM.py`, `Train_Ai.py`, `data_loader.py`

- [ ] 4. Define anomaly targets, exceedance labels, and heatwave metrics

  **What to do**: Add deterministic helpers that (a) convert `t2m` into anomaly targets using the train-year day-of-year climatology, (b) derive per-grid exceedance labels against the local 95th-percentile threshold, and (c) derive regional alert labels for reporting by marking an alert when at least 10% of Thailand grid cells exceed threshold for at least 2 consecutive forecast steps. Extend evaluation to report RMSE/MAE for continuous fields plus Brier score, precision, recall, F1, hit rate, false alarm rate, and ETS for exceedance/alert outputs.
  **Must NOT do**: Do not keep the current global scalar threshold as the main event definition, and do not replace continuous metrics with classification-only reporting.

  **Recommended Agent Profile**:
  - Category: `deep` - Reason: This task sets the exact modeling target and acceptance logic.
  - Skills: `[]` - No extra skill is required.
  - Omitted: `playwright` - No browser interaction is needed.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 5, 8, 9, 10, 11 | Blocked By: 1, 3

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `Train_Ai.py:113` - Existing event label logic based on regional max temperature; replace this with the new local-threshold + regional-alert scheme.
  - Pattern: `Train_Ai.py:132` - Existing classification metric helper to extend.
  - Pattern: `Train_Ai.py:172` - Existing evaluation function to expand.
  - Pattern: `api_server.py:67` - Current risk mapping is absolute-threshold-based; future API logic must consume the new alert outputs additively.
  - External: `https://ghhin.org/resources/heat-wave-trends-in-southeast-asia-during-1979-2018-the-impact-of-humidity/` - Regional evidence that humidity-aware heatwave framing matters in SEA.
  - External: `https://www.nature.com/articles/s41612-024-00797-w` - 2023 SEA heatwave study motivating humidity/soil-moisture-aware event design.

  **Acceptance Criteria** (agent-executable only):
  - [ ] The repo has one authoritative helper for anomaly target generation and one authoritative helper for exceedance/alert labels.
  - [ ] The training/evaluation report includes RMSE, MAE, Brier, precision, recall, F1, hit rate, false alarm rate, and ETS.
  - [ ] A toy sequence test demonstrates that the 10%-grid / 2-step persistence rule labels alerts correctly.

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```
  Scenario: Toy label correctness
    Tool: Bash
    Steps: Run `python -c "from Train_ConvLSTM import make_heatwave_labels; import numpy as np; x=np.zeros((1,2,10,10)); x[:, :, :2, :5]=1; print(make_heatwave_labels(x, persistence=2, min_fraction=0.1))"`.
    Expected: The toy case returns a positive alert because exactly 10% of cells exceed threshold for 2 steps.
    Evidence: .sisyphus/evidence/task-4-toy-labels.txt

  Scenario: Metrics report coverage
    Tool: Bash
    Steps: Run a lightweight evaluation helper and print the metric keys.
    Expected: Output contains `rmse`, `mae`, `brier`, `precision`, `recall`, `f1`, `hit_rate`, `false_alarm_rate`, and `ets`.
    Evidence: .sisyphus/evidence/task-4-metric-keys.txt
  ```

  **Commit**: NO | Message: `feat(targets): add anomaly and heatwave event metrics` | Files: `Train_ConvLSTM.py`, `Train_Ai.py`

- [ ] 5. Align the RandomForest baseline with the new split and metric contract

  **What to do**: Keep `Train_Ai.py` as a comparison-only baseline by wiring it into the year-block split, anomaly target, exceedance label, and expanded metric/reporting contract. Ensure saved baseline reports are directly comparable to ConvLSTM outputs and are clearly marked as non-deployment artifacts.
  **Must NOT do**: Do not allow RandomForest checkpoints to become the default model loaded by `api_server.py`, and do not maintain a divergent threshold definition.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` - Reason: It is mostly integration work but must stay tightly aligned with the new evaluation contract.
  - Skills: `[]` - No special skill is needed.
  - Omitted: `playwright` - No browser interaction is needed.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 11 | Blocked By: 1, 3, 4

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `Train_Ai.py:192` - Existing baseline comparison helper logic.
  - Pattern: `Train_Ai.py:216` - Existing RandomForest training workflow to adapt.
  - Pattern: `api_server.py:118` - Runtime still supports random_forest checkpoints; preserve compatibility but prevent default deployment.

  **Acceptance Criteria** (agent-executable only):
  - [ ] `Train_Ai.py` runs under the same year-block split and label logic as the ConvLSTM path.
  - [ ] Baseline reports include the same continuous and event metrics as the canonical path.
  - [ ] Saved baseline metadata explicitly marks model role as `baseline_only` or equivalent.

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```
  Scenario: Baseline evaluation run
    Tool: Bash
    Steps: Run `python Train_Ai.py` with a short configuration on the available dataset.
    Expected: The run completes and prints/saves continuous + event metrics under the new contract.
    Evidence: .sisyphus/evidence/task-5-rf-baseline.txt

  Scenario: Baseline metadata role
    Tool: Bash
    Steps: Inspect the saved checkpoint/report metadata after the baseline run.
    Expected: Metadata explicitly marks the RandomForest artifact as a baseline rather than the deployment default.
    Evidence: .sisyphus/evidence/task-5-rf-metadata.txt
  ```

  **Commit**: NO | Message: `chore(baseline): align random forest evaluation contract` | Files: `Train_Ai.py`, `api_server.py`

## Final Verification Wave (4 parallel agents, ALL must APPROVE)
- [ ] F1. Plan Compliance Audit - oracle
- [ ] F2. Code Quality Review - unspecified-high
- [ ] F3. Real Manual QA - unspecified-high (+ playwright if UI added later)
- [ ] F4. Scope Fidelity Check - deep

## Commit Strategy
- Use one integration commit per completed wave after its QA passes.
- Keep baseline-preservation changes and API contract changes in separate commits if wave boundaries blur.
- Do not commit downloaded ERA5 binaries, checkpoints, or generated evidence artifacts.

## Success Criteria
- ConvLSTM remains the deployed model path and beats or matches the retained RandomForest baseline on event-focused metrics for held-out years.
- Heatwave metrics are leakage-safe and reproducible from saved reports.
- API consumers keep receiving current summary fields while gaining calibrated heat-risk context.
- The repo documents the new variable contract and threshold logic through checkpoint metadata and runtime responses, not hidden assumptions.
