# Prediction Accuracy Overhaul — Fix RF Inference + Implement ConvLSTM

## TL;DR

> **Quick Summary**: Fix the completely fake RF inference pipeline (probability-based temp boost hack), implement the missing ConvLSTM architecture for real temperature regression, fix normalization data leakage, remove all hardcoded dummy values, and create a dual-model inference system.
> 
> **Deliverables**:
> - Fixed RF inference that properly uses classifier probability output
> - Working ConvLSTM model implementation (ConvLSTMCell, HeatwaveConvLSTM, PhysicsInformedLoss)
> - Updated ConvLSTM training script with proper data pipeline
> - Dual-model inference in api_server.py (RF for events, ConvLSTM for temperatures)
> - Clean API responses with real data instead of hardcoded dummies
> - Fixed normalization (train-split-only stats, no leakage)
> 
> **Estimated Effort**: Large
> **Parallel Execution**: YES — 4 waves
> **Critical Path**: Task 1 → Task 3 → Task 5 → Task 8 → Task 10 → Task 11 → F1-F4

---

## Context

### Original Request
"ต้องการให้การPredict มันแม่นยำที่สุดและไม่พลาดแก้พวกข้อมุลจากโค้ดเดิม"
(Want prediction to be as accurate as possible and fix all data issues from existing code)

### Interview Summary
**Key Discussions**:
- User chose comprehensive fix: fix RF inference + add ConvLSTM + fix all data issues
- Frontend only uses `day_name`, `date`, `T2M_MAX`, `risk_level` from forecast — dummy weather fields are sent but never displayed
- User accepts that retraining will be required after fixes

**Research Findings**:
- RF model is a BalancedRandomForest CLASSIFIER but inference pretends it's a regressor (adds `prob × 1.5°C` to last temp)
- ConvLSTM code in `src/models/convlstm.py` is just 3 lines of `None` stubs — completely unimplemented
- `Train_ConvLSTM.py` would crash immediately because it imports `None` classes
- `ModelManager` in `src/models/manager.py` already has conditional loading for sklearn vs PyTorch
- Data pipeline produces 8 channels (z, t2m, swvl1, tp, humidity + elevation, lat, lon)
- Training uses 70/15/15 temporal split but inference uses 80/20 — mismatch
- Normalization stats are computed on ALL data before split — data leakage

### Metis Review
**Identified Gaps** (addressed):
- Missing Phase 0 for regression guards before changes → Added Task 1
- `ModelManager.predict_event()` only handles sklearn → Will update alongside api_server.py (Task 10)
- `Train_ConvLSTM.py` has own 80/20 split (not 70/15/15) → Fixed in Task 8
- `create_sequences` parameter name mismatch (`pred_len` vs `future_seq`) → Fixed in Task 8
- Checkpoint naming divergence (RF vs ConvLSTM patterns) → Handled in Task 10
- Two API servers exist (Flask + FastAPI) → Focus on Flask `api_server.py` as production path; FastAPI `src/api/main.py` has no prediction endpoints

---

## Work Objectives

### Core Objective
Make the heatwave prediction system produce accurate, real predictions by fixing the fake inference pipeline, implementing the missing ConvLSTM architecture, and removing all hardcoded/dummy data from API responses.

### Concrete Deliverables
- `src/models/convlstm.py` — Full ConvLSTM implementation (ConvLSTMCell, HeatwaveConvLSTM, PhysicsInformedLoss)
- `src/data/loader.py` — Fixed normalization (train-split-only stats)
- `api_server.py` — Fixed inference pipeline with dual-model support, real data in responses
- `Train_ConvLSTM.py` — Updated training script with proper data pipeline
- `src/models/manager.py` — Updated to support ConvLSTM inference

### Definition of Done
- [ ] `python -c "from src.models.convlstm import ConvLSTMCell, HeatwaveConvLSTM, PhysicsInformedLoss; print('OK')"` → prints OK
- [ ] `python Train_ConvLSTM.py` runs without import errors (may not complete training without GPU)
- [ ] API `/api/predict` returns real dates (not "2000-XX-XX") and model-derived probability
- [ ] API `/api/forecast` returns no hardcoded weather values (RH2M≠60, WS10M≠2.5)
- [ ] Inference split matches training split (70/15/15)
- [ ] Normalization stats computed on training split only

### Must Have
- ConvLSTM implementation matching the API contract expected by `Train_ConvLSTM.py`
- RF inference using actual `predict_proba` output (not as temp modifier)
- Normalization leakage fix
- Split consistency between training and inference
- Clipping during inference to match training
- Real dates from data time index
- Backward-compatible checkpoint loading (old checkpoints still work)

### Must NOT Have (Guardrails)
- Do NOT change `create_sequences` function signature — 3 callers depend on it
- Do NOT change API response JSON key names — frontend depends on them
- Do NOT remove weather fields from API response (send `null` instead of hardcoded values)
- Do NOT modify `src/data/loader.py`'s `DataLoader` class beyond normalization fix
- Do NOT change `Train_Ai.py` (RF training) — it's well-built, leave it alone
- Do NOT add unnecessary abstractions — keep code direct and readable
- Do NOT mix Thai and English in new code comments — use English consistently
- Do NOT introduce new dependencies — use only existing packages (torch, numpy, flask, sklearn)
- Do NOT change the `(Batch, Time, Channels, H, W)` tensor shape contract
- Do NOT consolidate Flask/FastAPI servers — out of scope

---

## Verification Strategy (MANDATORY)

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: NO (no test framework configured)
- **Automated tests**: None (no test infrastructure to leverage)
- **Framework**: None
- **QA Method**: Agent-Executed QA Scenarios — each task has concrete verification commands

### QA Policy
Every task MUST include agent-executed QA scenarios.
Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

- **Model/Module**: Use Bash (python -c) — Import, instantiate, verify shapes/outputs
- **API**: Use Bash (curl + python) — Send requests, assert response fields
- **Training**: Use Bash (python) — Run training for 1 epoch, verify checkpoint format
- **Data Pipeline**: Use Bash (python -c) — Load data, verify shapes, normalization ranges

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately — foundation):
├── Task 1: Capture current behavior baselines [quick]
├── Task 2: Fix normalization data leakage in loader.py [quick]
└── Task 3: Implement ConvLSTM architecture in src/models/convlstm.py [deep]

Wave 2 (After Wave 1 — RF fixes + training):
├── Task 4: Add data clipping in inference path [quick]
├── Task 5: Fix RF inference — remove fake temp boost [deep]
├── Task 6: Fix risk/probability to use model output [quick]
├── Task 7: Remove hardcoded dummy values from API responses [quick]
└── Task 8: Update ConvLSTM training script [unspecified-high]

Wave 3 (After Wave 2 — integration):
├── Task 9: Update ModelManager for ConvLSTM inference [quick]
└── Task 10: Implement dual-model inference in api_server.py [deep]

Wave 4 (After Wave 3 — cleanup + verify):
└── Task 11: End-to-end API verification [unspecified-high]

Wave FINAL (After ALL tasks — independent review, 4 parallel):
├── Task F1: Plan compliance audit (oracle)
├── Task F2: Code quality review (unspecified-high)
├── Task F3: Real manual QA (unspecified-high)
└── Task F4: Scope fidelity check (deep)

Critical Path: Task 1 → Task 3 → Task 5 → Task 8 → Task 10 → Task 11 → F1-F4
Parallel Speedup: ~55% faster than sequential
Max Concurrent: 5 (Wave 2)
```

### Dependency Matrix

| Task | Depends On | Blocks | Wave |
|------|-----------|--------|------|
| 1 | — | 4, 5, 6, 7 | 1 |
| 2 | — | 5, 8 | 1 |
| 3 | — | 8, 9, 10 | 1 |
| 4 | 1 | 10 | 2 |
| 5 | 1, 2 | 10, 11 | 2 |
| 6 | 1 | 10, 11 | 2 |
| 7 | 1 | 11 | 2 |
| 8 | 2, 3 | 10 | 2 |
| 9 | 3 | 10 | 3 |
| 10 | 4, 5, 6, 8, 9 | 11 | 3 |
| 11 | 5, 6, 7, 10 | F1-F4 | 4 |

### Agent Dispatch Summary

- **Wave 1**: **3 tasks** — T1 → `quick`, T2 → `quick`, T3 → `deep`
- **Wave 2**: **5 tasks** — T4 → `quick`, T5 → `deep`, T6 → `quick`, T7 → `quick`, T8 → `unspecified-high`
- **Wave 3**: **2 tasks** — T9 → `quick`, T10 → `deep`
- **Wave 4**: **1 task** — T11 → `unspecified-high`
- **FINAL**: **4 tasks** — F1 → `oracle`, F2 → `unspecified-high`, F3 → `unspecified-high`, F4 → `deep`

---

## TODOs

> Implementation + Verification = ONE Task. Never separate.
> EVERY task MUST have: Recommended Agent Profile + Parallelization info + QA Scenarios.

- [x] 1. Capture Current Behavior Baselines

  **What to do**:
  - Run the existing API server and capture current response shapes/values from `/api/predict`, `/api/forecast`, `/api/map`, `/api/health`
  - Document the current data pipeline output shapes: what `load_era5_data()` returns, what `create_sequences()` returns, tensor dimensions
  - Record the current checkpoint format/keys
  - Save all baseline evidence to `.sisyphus/evidence/`

  **Must NOT do**:
  - Do NOT change any code — this is observation only
  - Do NOT run training

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple data capture, no complex logic
  - **Skills**: []
  - **Skills Evaluated but Omitted**:
    - `playwright`: Not needed — API endpoints only, no browser UI

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2, 3)
  - **Blocks**: Tasks 4, 5, 6, 7 (need baselines to verify changes)
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References**:
  - `api_server.py:858-899` — `/api/predict` endpoint response shape
  - `api_server.py:902-957` — `/api/forecast` endpoint response shape
  - `api_server.py:960-1023` — `/api/map` endpoint GeoJSON response shape
  - `api_server.py:1113-1116` — `/api/health` endpoint

  **API/Type References**:
  - `src/data/loader.py:261-275` — `create_sequences(data, seq_len, pred_len)` returns (X, Y) numpy arrays
  - `src/data/loader.py:322-348` — `load_era5_data()` returns (data, lats, lons, mean, std)

  **WHY Each Reference Matters**:
  - These are the exact functions and endpoints we'll modify later — we need their current behavior captured before any changes

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Capture data pipeline shapes
    Tool: Bash (python -c)
    Preconditions: ERA5 data exists in era5_data/
    Steps:
      1. Run: python -c "from data_loader import load_era5_data, create_sequences; d,la,lo,m,s = load_era5_data('era5_data'); print(f'data={d.shape} lats={la.shape} lons={lo.shape} mean={m.shape} std={s.shape}'); X,Y = create_sequences(d, seq_len=7, pred_len=2); print(f'X={X.shape} Y={Y.shape}')"
      2. Save output to .sisyphus/evidence/task-1-data-shapes.txt
    Expected Result: Shapes printed without errors. Data shape is (Time, Channels, H, W).
    Failure Indicators: ImportError, FileNotFoundError, shape mismatch
    Evidence: .sisyphus/evidence/task-1-data-shapes.txt

  Scenario: Capture checkpoint format
    Tool: Bash (python -c)
    Preconditions: Model checkpoint exists in models/
    Steps:
      1. Run: python -c "import torch, glob; f=sorted(glob.glob('models/heatwave_model_checkpoint_v*.pth'))[-1]; ckpt=torch.load(f, map_location='cpu', weights_only=False); print('Keys:', list(ckpt.keys())); md=ckpt.get('metadata',{}); print('Metadata keys:', list(md.keys())); print('Model type:', ckpt.get('model_type','unknown'))"
      2. Save output to .sisyphus/evidence/task-1-checkpoint-format.txt
    Expected Result: Checkpoint keys and metadata keys printed
    Failure Indicators: No checkpoint files found, load error
    Evidence: .sisyphus/evidence/task-1-checkpoint-format.txt

  Scenario: Capture current API responses (if server starts)
    Tool: Bash (curl + python)
    Preconditions: Try starting api_server.py (may fail if no model)
    Steps:
      1. Start server in background: python api_server.py &
      2. Wait 10 seconds for startup
      3. curl -s http://localhost:5000/api/health > .sisyphus/evidence/task-1-health.json
      4. curl -s http://localhost:5000/api/predict > .sisyphus/evidence/task-1-predict.json
      5. curl -s http://localhost:5000/api/forecast > .sisyphus/evidence/task-1-forecast.json
      6. Kill server
    Expected Result: JSON responses saved. If server can't start (no model), document that as evidence too.
    Failure Indicators: Server crash, connection refused (acceptable — document it)
    Evidence: .sisyphus/evidence/task-1-health.json, task-1-predict.json, task-1-forecast.json
  ```

  **Evidence to Capture:**
  - [ ] task-1-data-shapes.txt — Current data pipeline dimensions
  - [ ] task-1-checkpoint-format.txt — Checkpoint key structure
  - [ ] task-1-health.json — Health endpoint response (or error doc)
  - [ ] task-1-predict.json — Predict endpoint response (or error doc)
  - [ ] task-1-forecast.json — Forecast endpoint response (or error doc)

  **Commit**: YES
  - Message: `chore: capture baseline behavior for regression checks`
  - Files: `.sisyphus/evidence/task-1-*`

---

- [x] 2. Fix Normalization Data Leakage in Data Loader

  **What to do**:
  - Modify `src/data/loader.py` `prepare_training_data()` method to NOT compute normalization internally
  - Instead, return the raw (un-normalized) stacked array along with the stats dict (still include lats, lons, time_index, etc.)
  - Add a new method `compute_train_normalization_stats(data, train_end_idx)` that computes mean/std only on `data[:train_end_idx]`
  - Update the backward-compatible `load_era5_data()` function to support an optional `normalize=False` mode (it already has this flag at line 343)
  - Ensure the `compute_normalization_stats()` standalone function (line 285) is NOT changed — it's used by `Train_ConvLSTM.py`
  - Add `clean_data()` call capability: the standalone `clean_data()` function already exists and works, just ensure it's accessible

  **Must NOT do**:
  - Do NOT change `create_sequences()` signature
  - Do NOT change `DataLoader.load_era5()` — it's solid
  - Do NOT break backward compatibility with `load_era5_data()` wrapper — callers still pass `normalize=True` as default

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Small, targeted change to a single function in one file
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3)
  - **Blocks**: Tasks 5, 8 (need proper normalization before fixing inference/training)
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References**:
  - `src/data/loader.py:190-259` — `prepare_training_data()` method — THIS is what changes
  - `src/data/loader.py:242-246` — The leakage: `mean = data_array.mean(axis=(0,2,3), keepdims=True)` on ALL data
  - `src/data/loader.py:285-289` — `compute_normalization_stats()` standalone function — keep unchanged
  - `src/data/loader.py:322-348` — `load_era5_data()` backward-compatible wrapper — minor update

  **API/Type References**:
  - `Train_Ai.py:505-548` — How training calls `loader.prepare_training_data(full_ds)` and then `temporal_split_data()`
  - `Train_Ai.py:513` — Training saves stats as `train_mean`, `train_std` — misleading name, actually global stats
  - `Train_Ai.py:890-891` — Checkpoint saves `normalization_mean`, `normalization_std` in metadata

  **WHY Each Reference Matters**:
  - `prepare_training_data` is where the leakage happens — need to return raw data so callers can normalize after splitting
  - Training pipeline shows how stats are used and saved — our fix must remain compatible
  - The standalone functions are used by `Train_ConvLSTM.py` and must not change signature

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: prepare_training_data returns un-normalized data when updated
    Tool: Bash (python -c)
    Preconditions: ERA5 data in era5_data/
    Steps:
      1. Run: python -c "
         from src.data.loader import DataLoader
         loader = DataLoader()
         ds = loader.load_era5()
         data, stats = loader.prepare_training_data(ds)
         import numpy as np
         # Check data is raw (not z-score normalized — mean should NOT be ~0)
         ch1_mean = np.mean(data[:, 1, :, :])  # t2m channel
         print(f't2m channel mean: {ch1_mean:.2f}')
         assert abs(ch1_mean) > 1.0, f'Data appears already normalized (mean={ch1_mean})'
         print('PASS: data is raw (un-normalized)')
         print(f'Stats keys: {list(stats.keys())}')
         "
      2. Save output to .sisyphus/evidence/task-2-normalization-fix.txt
    Expected Result: t2m channel mean is a large value (temperature in Kelvin ~290-310), confirming data is NOT normalized
    Failure Indicators: Mean is close to 0 (still normalized), ImportError
    Evidence: .sisyphus/evidence/task-2-normalization-fix.txt

  Scenario: Backward-compatible load_era5_data still works with normalize=True
    Tool: Bash (python -c)
    Preconditions: ERA5 data in era5_data/
    Steps:
      1. Run: python -c "
         from data_loader import load_era5_data
         d, la, lo, m, s = load_era5_data('era5_data', normalize=True)
         import numpy as np
         ch1_mean = np.abs(np.mean(d[:, 1, :, :]))
         print(f't2m normalized mean: {ch1_mean:.4f}')
         assert ch1_mean < 1.0, f'Data not normalized (mean={ch1_mean})'
         print('PASS: backward compat with normalize=True works')
         "
    Expected Result: t2m mean is close to 0 (z-score normalized)
    Failure Indicators: Mean is large (not normalized), ImportError
    Evidence: .sisyphus/evidence/task-2-backward-compat.txt
  ```

  **Evidence to Capture:**
  - [ ] task-2-normalization-fix.txt — Proof data is returned un-normalized
  - [ ] task-2-backward-compat.txt — Proof backward compat still works

  **Commit**: YES
  - Message: `fix(data): compute normalization stats on training split only`
  - Files: `src/data/loader.py`

---

- [x] 3. Implement ConvLSTM Architecture

  **What to do**:
  - Implement the full ConvLSTM architecture in `src/models/convlstm.py` (currently just 3 lines of `None` stubs)
  - Implement 3 classes:
    1. **`ConvLSTMCell`**: Single ConvLSTM cell with gates (input, forget, output, cell)
       - Constructor: `(input_dim, hidden_dim, kernel_size, bias=True)`
       - Forward: `(input_tensor, cur_state) → (h_next, c_next)`
       - `init_hidden(batch_size, image_size) → (h, c)` with zeros
    2. **`HeatwaveConvLSTM`**: Multi-layer ConvLSTM encoder-decoder for spatiotemporal forecasting
       - Constructor: `(input_dim, hidden_dim, kernel_size, num_layers)` — MUST match `Train_ConvLSTM.py:64-69`
       - Forward: `(x, future_seq) → predictions` — MUST match `Train_ConvLSTM.py:85`
       - Input shape: `(batch, seq_len, channels, H, W)`
       - Output shape: `(batch, future_seq, channels, H, W)`
       - Architecture: encode input sequence through stacked ConvLSTM layers, then decode by running `future_seq` forward steps, using Conv2d to project hidden state back to input_dim channels
    3. **`PhysicsInformedLoss`**: Combined MSE + physics-consistency loss
       - Constructor: `(lambda_phy=0.1)` — MUST match `Train_ConvLSTM.py:71`
       - Forward: `(prediction, target) → (total_loss, mse_loss, physics_loss)` — MUST match `Train_ConvLSTM.py:86`
       - Physics component: penalize spatially unrealistic temperature gradients (adiabatic lapse rate constraint)

  **Must NOT do**:
  - Do NOT add extra constructor parameters not expected by `Train_ConvLSTM.py`
  - Do NOT change the expected forward() return types
  - Do NOT import external model libraries — implement from scratch using torch.nn

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Complex neural network architecture implementation requiring careful spatial reasoning and physics understanding
  - **Skills**: []
  - **Skills Evaluated but Omitted**:
    - `playwright`: Not relevant — no browser interaction

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2)
  - **Blocks**: Tasks 8 (ConvLSTM training), 9 (ModelManager), 10 (dual inference)
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References**:
  - `src/models/convlstm.py:1-3` — Current stubs to REPLACE: `ConvLSTMCell = None; HeatwaveConvLSTM = None; PhysicsInformedLoss = None`
  - `heatwave_model.py:8-12` — Re-export wrapper that imports from `src.models.convlstm` — implementation must export these 3 names

  **API/Type References (CRITICAL — these define the contract):**
  - `Train_ConvLSTM.py:64-69` — Constructor call: `HeatwaveConvLSTM(input_dim=CHANNELS, hidden_dim=HIDDEN_DIM, kernel_size=KERNEL_SIZE, num_layers=NUM_LAYERS)`
  - `Train_ConvLSTM.py:71` — Loss constructor: `PhysicsInformedLoss(lambda_phy=0.1)`
  - `Train_ConvLSTM.py:85` — Forward call: `output = model(batch_x, future_seq=FUTURE_SEQ)`
  - `Train_ConvLSTM.py:86` — Loss call: `loss, mse, phy = criterion(output, batch_y)`
  - `Train_ConvLSTM.py:14-22` — Config constants: `CHANNELS=6, HIDDEN_DIM=[32,32], KERNEL_SIZE=[(3,3),(3,3)], NUM_LAYERS=2`
  - `src/models/manager.py:52-57` — ModelManager instantiation: `HeatwaveConvLSTM(input_dim=X, hidden_dim=[16,16], kernel_size=[(3,3),(3,3)], num_layers=2)`

  **External References:**
  - ConvLSTM paper: Shi et al. 2015 "Convolutional LSTM Network" — standard ConvLSTM cell architecture
  - PyTorch Conv2d docs for implementing gates with convolutions

  **WHY Each Reference Matters:**
  - `Train_ConvLSTM.py` lines define the EXACT API contract the implementation must satisfy — any mismatch = crash
  - `ModelManager` shows the model will also be loaded with potentially different hyperparams — must handle variable hidden_dim/kernel_size/num_layers
  - Config constants show default values to test with

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: ConvLSTM classes are importable and not None
    Tool: Bash (python -c)
    Preconditions: None
    Steps:
      1. Run: python -c "
         from src.models.convlstm import ConvLSTMCell, HeatwaveConvLSTM, PhysicsInformedLoss
         assert ConvLSTMCell is not None, 'ConvLSTMCell is None'
         assert HeatwaveConvLSTM is not None, 'HeatwaveConvLSTM is None'
         assert PhysicsInformedLoss is not None, 'PhysicsInformedLoss is None'
         print('PASS: All 3 classes are importable and not None')
         "
    Expected Result: All 3 classes importable
    Failure Indicators: ImportError, assertion failure
    Evidence: .sisyphus/evidence/task-3-import.txt

  Scenario: HeatwaveConvLSTM matches Train_ConvLSTM.py API contract
    Tool: Bash (python -c)
    Preconditions: None (CPU-only test)
    Steps:
      1. Run: python -c "
         import torch
         from src.models.convlstm import HeatwaveConvLSTM, PhysicsInformedLoss
         # Match Train_ConvLSTM.py:64-69 exactly
         model = HeatwaveConvLSTM(input_dim=6, hidden_dim=[32,32], kernel_size=[(3,3),(3,3)], num_layers=2)
         # Match Train_ConvLSTM.py:85 — forward with future_seq kwarg
         x = torch.randn(2, 5, 6, 16, 16)  # (batch=2, seq=5, ch=6, H=16, W=16)
         output = model(x, future_seq=2)
         assert output.shape == (2, 2, 6, 16, 16), f'Wrong shape: {output.shape}'
         print(f'PASS: output shape = {output.shape}')
         # Match Train_ConvLSTM.py:71,86 — loss
         criterion = PhysicsInformedLoss(lambda_phy=0.1)
         target = torch.randn(2, 2, 6, 16, 16)
         loss, mse, phy = criterion(output, target)
         assert loss.requires_grad, 'Loss not differentiable'
         print(f'PASS: loss={loss.item():.4f}, mse={mse.item():.4f}, phy={phy.item():.4f}')
         "
    Expected Result: Output shape (2,2,6,16,16), loss is differentiable
    Failure Indicators: Shape mismatch, TypeError on constructor args, loss not differentiable
    Evidence: .sisyphus/evidence/task-3-api-contract.txt

  Scenario: Works with 8 channels (for future dual-channel support)
    Tool: Bash (python -c)
    Preconditions: None
    Steps:
      1. Run: python -c "
         import torch
         from src.models.convlstm import HeatwaveConvLSTM
         model = HeatwaveConvLSTM(input_dim=8, hidden_dim=[32,32], kernel_size=[(3,3),(3,3)], num_layers=2)
         x = torch.randn(1, 7, 8, 16, 16)
         out = model(x, future_seq=2)
         assert out.shape == (1, 2, 8, 16, 16), f'Wrong: {out.shape}'
         print(f'PASS: 8-channel model works, shape={out.shape}')
         "
    Expected Result: 8-channel model works correctly
    Failure Indicators: Channel mismatch error
    Evidence: .sisyphus/evidence/task-3-8channel.txt

  Scenario: heatwave_model.py re-export still works
    Tool: Bash (python -c)
    Preconditions: None
    Steps:
      1. Run: python -c "
         from heatwave_model import ConvLSTMCell, HeatwaveConvLSTM, PhysicsInformedLoss
         assert ConvLSTMCell is not None
         assert HeatwaveConvLSTM is not None
         assert PhysicsInformedLoss is not None
         print('PASS: Re-export from heatwave_model.py works')
         "
    Expected Result: All 3 classes accessible via legacy import path
    Failure Indicators: ImportError, None values
    Evidence: .sisyphus/evidence/task-3-reexport.txt
  ```

  **Evidence to Capture:**
  - [ ] task-3-import.txt — Classes importable
  - [ ] task-3-api-contract.txt — API contract matches Train_ConvLSTM.py
  - [ ] task-3-8channel.txt — 8-channel support works
  - [ ] task-3-reexport.txt — Legacy import path works

  **Commit**: YES
  - Message: `feat(model): implement ConvLSTM architecture (ConvLSTMCell, HeatwaveConvLSTM, PhysicsInformedLoss)`
  - Files: `src/models/convlstm.py`

---

---

- [x] 4. Add Data Clipping in Inference Path

  **What to do**:
  - In `api_server.py`'s `load_resources()` function, add a call to `clean_data()` on the raw data BEFORE normalization
  - Import `clean_data` from `data_loader` (already available via `src/data/loader.py` re-export)
  - If checkpoint metadata contains `clip_bounds`, use those; otherwise compute fresh percentile bounds
  - This ensures inference data undergoes the same preprocessing as training data

  **Must NOT do**:
  - Do NOT change `clean_data()` function itself
  - Do NOT apply clipping AFTER normalization

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single function call addition in one location
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 5, 6, 7, 8)
  - **Blocks**: Task 10 (dual inference needs clean data)
  - **Blocked By**: Task 1 (need baseline to verify change)

  **References**:

  **Pattern References**:
  - `api_server.py:732` — Current inference: `data = normalize_data(data_raw, mean, std)` — NO clipping before this
  - `Train_Ai.py:73-74` — Training config: `CLIP_LOW_PERCENTILE = 0.5`, `CLIP_HIGH_PERCENTILE = 99.5`
  - `Train_Ai.py:510-512` — Training calls: `data_clean, clip_bounds = loader.clean_data(data_array)` before normalization
  - `src/data/loader.py:297-319` — `clean_data()` function: replaces inf, fills NaN, clips outliers

  **WHY Each Reference Matters**:
  - `api_server.py:732` is where the fix goes — add `clean_data()` call before `normalize_data()`
  - Training code shows the exact clipping parameters used — inference must match
  - `clean_data()` returns `(cleaned_data, (lower, upper))` — save bounds for consistency

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Verify clean_data is called during inference
    Tool: Bash (python -c)
    Preconditions: None
    Steps:
      1. Read api_server.py and verify that `clean_data` is imported and called in `load_resources()`
      2. Run: python -c "
         import ast
         with open('api_server.py') as f:
             source = f.read()
         assert 'clean_data' in source, 'clean_data not found in api_server.py'
         # Check it's in the imports
         assert 'clean_data' in source.split('from data_loader import')[1].split('\n')[0], 'clean_data not imported'
         print('PASS: clean_data is imported and present in api_server.py')
         "
    Expected Result: clean_data is imported and called
    Failure Indicators: clean_data not found in source
    Evidence: .sisyphus/evidence/task-4-clipping.txt
  ```

  **Commit**: NO (groups with Tasks 5, 6, 7)

---

- [x] 5. Fix RF Inference — Remove Fake Temperature Boost

  **What to do**:
  - Completely rewrite `get_prediction_sequence()` in `api_server.py` (lines 820-855)
  - REMOVE the fake temp boost: `pred_step[1] = pred_step[1] + (prob * runtime_inference_temp_boost_c)` (line 843)
  - REMOVE the `runtime_inference_temp_boost_c` variable (line 78)
  - For RF model inference, the function should:
    1. Flatten the input sequence: `x_flat = x_seq.reshape(1, -1)`
    2. Get RF probability: `prob = model.predict_proba(x_flat)[0, 1]`
    3. Use the LAST KNOWN temperature grid from input as the prediction (persistence baseline)
    4. Return the probability ALONGSIDE the temperature grid (don't modify the grid with probability)
  - For the autoregressive loop: since RF is a classifier (not a regressor), it CANNOT predict future temperatures. The autoregressive approach is fundamentally wrong for a classifier. Instead:
    - Day 1: Use last input temp grid + RF probability
    - Days 2-7: Acknowledge these are extrapolations with increasing uncertainty
    - Store RF probability per day (it may change if we shift input window)
  - Fix the split: change `int(len(X) * 0.8)` to `int(len(X) * 0.85)` to use last 15% as test (matching training's 70/15/15 where test = last 15%)
  - Use normalization stats from checkpoint metadata if available (already partially done at lines 707-731, just ensure consistency)

  **Must NOT do**:
  - Do NOT try to make RF predict temperatures — it's a classifier, accept that
  - Do NOT remove the autoregressive loop structure — it will be used properly by ConvLSTM in Task 10
  - Do NOT change the function signature of `get_prediction_sequence()`
  - Do NOT modify any training code

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Core inference logic rewrite requires understanding of the full prediction pipeline
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 4, 6, 7, 8)
  - **Blocks**: Tasks 10, 11 (inference pipeline must be correct)
  - **Blocked By**: Tasks 1 (baseline), 2 (normalization fix)

  **References**:

  **Pattern References**:
  - `api_server.py:820-855` — `get_prediction_sequence()` — THE function to rewrite
  - `api_server.py:837-843` — The fake temp boost hack to REMOVE
  - `api_server.py:78` — `runtime_inference_temp_boost_c = 1.5` — REMOVE this variable
  - `api_server.py:741-746` — Split index and temp scalar extraction
  - `api_server.py:810-817` — `get_prediction_data()` wrapper that calls `get_prediction_sequence`

  **API/Type References**:
  - `api_server.py:858-899` — `/api/predict` endpoint — calls `get_prediction_data()`, uses returned temp grid
  - `api_server.py:902-957` — `/api/forecast` endpoint — calls `get_prediction_sequence(days=7)`

  **WHY Each Reference Matters**:
  - Lines 820-855 are the exact code to rewrite — the fake prediction logic
  - Lines 858-957 are the callers — must understand what they expect from the function
  - The split fix at line 741 ensures test data matches what training evaluated on

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Fake temp boost code is removed
    Tool: Bash (python -c)
    Preconditions: None
    Steps:
      1. Run: python -c "
         with open('api_server.py') as f:
             source = f.read()
         assert 'runtime_inference_temp_boost_c' not in source, 'Temp boost variable still exists'
         assert 'prob * runtime_inference_temp_boost_c' not in source, 'Temp boost formula still exists'
         print('PASS: Fake temp boost code removed')
         "
    Expected Result: No references to temp boost
    Failure Indicators: String still found in source
    Evidence: .sisyphus/evidence/task-5-no-boost.txt

  Scenario: Split matches training (85% boundary)
    Tool: Bash (python -c)
    Preconditions: None
    Steps:
      1. Run: python -c "
         with open('api_server.py') as f:
             source = f.read()
         assert '0.85' in source or 'train_ratio' in source, 'Split not updated from 0.8'
         assert 'int(len(X) * 0.8)' not in source, 'Old 80/20 split still present'
         print('PASS: Split updated')
         "
    Expected Result: Old 0.8 split removed, 0.85 or configurable split present
    Failure Indicators: Old split pattern still in code
    Evidence: .sisyphus/evidence/task-5-split.txt
  ```

  **Commit**: NO (groups with Tasks 4, 6, 7)

---

- [x] 6. Fix Risk/Probability to Use Model Output

  **What to do**:
  - Rewrite `get_risk_and_probability()` in `api_server.py` (lines 240-255) to accept both temperature AND model probability as inputs
  - New signature: `get_risk_and_probability(max_temp, model_probability=None)`
  - When `model_probability` is provided (from RF `predict_proba`), use it directly as the `probability` field
  - Keep temperature-based risk levels (they're reasonable thresholds for Thailand):
    - LOW: < 35°C
    - MEDIUM: 35-38°C
    - HIGH: 38-41°C
    - CRITICAL: >= 41°C
  - BUT replace the static probability lookup table with actual model probability
  - Update callers in `/api/predict` (line 870) and `/api/forecast` (line 920) to pass model probability

  **Must NOT do**:
  - Do NOT remove temperature-based risk levels (they're valid domain thresholds)
  - Do NOT change the risk level names (frontend depends on them)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple function signature update + caller updates
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 4, 5, 7, 8)
  - **Blocks**: Tasks 10, 11
  - **Blocked By**: Task 1 (baseline)

  **References**:

  **Pattern References**:
  - `api_server.py:240-255` — `get_risk_and_probability()` to rewrite
  - `api_server.py:870` — Caller in predict_summary: `risk_level, probability = get_risk_and_probability(max_temp)`
  - `api_server.py:920` — Caller in forecast_summary: same pattern

  **WHY Each Reference Matters**:
  - The function itself and its 2 callers are the complete scope of this change
  - Risk level thresholds should be kept but probability should come from model

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Probability map is removed
    Tool: Bash (python -c)
    Preconditions: None
    Steps:
      1. Run: python -c "
         with open('api_server.py') as f:
             source = f.read()
         assert 'probability_map' not in source, 'Static probability map still exists'
         assert '0.95' not in source.split('get_risk_and_probability')[1].split('def ')[0], 'Hardcoded 0.95 still in risk function'
         print('PASS: Static probability map removed')
         "
    Expected Result: No static probability lookup
    Failure Indicators: probability_map dict still in code
    Evidence: .sisyphus/evidence/task-6-probability.txt

  Scenario: Function accepts model_probability parameter
    Tool: Bash (python -c)
    Preconditions: None
    Steps:
      1. Run: python -c "
         import ast
         with open('api_server.py') as f:
             tree = ast.parse(f.read())
         for node in ast.walk(tree):
             if isinstance(node, ast.FunctionDef) and node.name == 'get_risk_and_probability':
                 args = [a.arg for a in node.args.args]
                 print(f'Args: {args}')
                 assert 'model_probability' in args or len(args) >= 2, f'Missing model_probability param: {args}'
                 print('PASS: function accepts model probability')
                 break
         "
    Expected Result: Function has model_probability parameter
    Failure Indicators: Function still has only max_temp parameter
    Evidence: .sisyphus/evidence/task-6-signature.txt
  ```

  **Commit**: NO (groups with Tasks 4, 5, 7)

---

- [x] 7. Remove Hardcoded Dummy Values from API Responses

  **What to do**:
  - Fix `predict_summary()` endpoint (lines 858-899):
    - Replace `"date": "2000-XX-XX"` (line 875) with actual date from data time index
    - The time index should be extracted during `load_resources()` from the data or checkpoint metadata
    - Replace `"RH2M": 60` and `"WS10M": 2.5` (lines 885-886) with `None`
  - Fix `forecast_summary()` endpoint (lines 902-957):
    - Replace `"PRECTOTCORR": 0` (line 937) with `None`
    - Replace `"WS10M": 2` (line 938) with `None`
    - Replace `"RH2M": 60` (line 939) with `None`
    - Replace `"NDVI": 0.5` (line 940) with `None`
  - For the date: during `load_resources()`, store the time index from the loaded data. Use the actual date of the test sample being predicted.

  **Must NOT do**:
  - Do NOT remove the weather dict keys (RH2M, WS10M, etc.) — just set values to `None`
  - Do NOT change the JSON response key names
  - Do NOT fabricate fake data to replace dummy data

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple value replacements across known locations
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 4, 5, 6, 8)
  - **Blocks**: Task 11
  - **Blocked By**: Task 1 (baseline)

  **References**:

  **Pattern References**:
  - `api_server.py:875` — `"date": "2000-XX-XX"` — REPLACE with real date
  - `api_server.py:885-886` — `"RH2M": 60, "WS10M": 2.5` — REPLACE with None
  - `api_server.py:937-940` — Forecast dummy values — REPLACE with None
  - `src/data/loader.py:253` — `time_index` is saved in stats dict — use this for real dates

  **WHY Each Reference Matters**:
  - These are the exact lines with dummy values to fix
  - time_index from data loader gives us real dates to use

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: No dummy date string in code
    Tool: Bash (python -c)
    Preconditions: None
    Steps:
      1. Run: python -c "
         with open('api_server.py') as f:
             source = f.read()
         assert '2000-XX-XX' not in source, 'Dummy date still in code'
         assert 'NDVI' not in source or 'None' in source.split('NDVI')[1][:20], 'NDVI still hardcoded'
         print('PASS: No dummy values found')
         "
    Expected Result: No hardcoded dummy strings
    Failure Indicators: Dummy patterns still in source
    Evidence: .sisyphus/evidence/task-7-no-dummies.txt
  ```

  **Commit**: YES (groups Tasks 4, 5, 6, 7 together)
  - Message: `fix(api): remove fake inference, use real model output, clean dummy data`
  - Files: `api_server.py`

---
---

- [x] 8. Update ConvLSTM Training Script

  **What to do**:
  - Update `Train_ConvLSTM.py` to fix multiple issues:
    1. **Fix split**: Change from 80/20 (line 53) to 70/15/15 temporal split matching `Train_Ai.py`
    2. **Fix normalization**: Compute stats on TRAINING split only (not entire dataset)
       - After splitting, compute mean/std on `train_data` only
       - Apply those stats to normalize train, val, and test data
    3. **Add clipping**: Call `clean_data()` before normalization (import from data_loader)
    4. **Fix parameter naming**: Line 50 passes `future_seq=FUTURE_SEQ` but `create_sequences` parameter is `pred_len` — use positional arg or fix to `pred_len=`
    5. **Update channel count**: Change `CHANNELS = 6` to `CHANNELS = 8` to match current data pipeline (z, t2m, swvl1, tp, humidity + elev, lat, lon)
    6. **Save proper checkpoint metadata**: Include `normalization_mean`, `normalization_std`, `input_dim`, `hidden_dim`, `kernel_size`, `num_layers`, `clip_bounds` in checkpoint dict
    7. **Add validation loop**: Current training only saves on best val loss but doesn't report validation metrics. Add proper validation reporting.
  - The training script should now produce checkpoints compatible with `ModelManager.load_model()`

  **Must NOT do**:
  - Do NOT change `create_sequences()` function signature
  - Do NOT modify the model architecture parameters (those come from Task 3)
  - Do NOT change the checkpoint filename pattern `heatwave_convlstm_v{N}.pth`

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Multi-aspect training pipeline update requiring data pipeline understanding
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 4, 5, 6, 7)
  - **Blocks**: Task 10 (dual inference needs ConvLSTM checkpoints)
  - **Blocked By**: Tasks 2 (normalization fix), 3 (ConvLSTM implementation)

  **References**:

  **Pattern References**:
  - `Train_ConvLSTM.py:1-124` — ENTIRE file to update
  - `Train_ConvLSTM.py:9` — Imports from `data_loader` — add `clean_data` to imports
  - `Train_ConvLSTM.py:19` — `CHANNELS = 6` — change to 8
  - `Train_ConvLSTM.py:41-47` — Data loading + cleaning section
  - `Train_ConvLSTM.py:52-55` — 80/20 split to replace with 70/15/15
  - `Train_ConvLSTM.py:110-117` — Checkpoint save dict — add metadata fields

  **API/Type References**:
  - `Train_Ai.py:504-566` — How RF training handles data pipeline (model to follow)
  - `Train_Ai.py:121-133` — `temporal_split_data()` function — consider reusing or copying pattern
  - `Train_Ai.py:886-920` — How RF saves checkpoint with full metadata
  - `src/models/manager.py:50-60` — How ModelManager loads PyTorch checkpoints — checkpoint must have `model_state_dict` key

  **WHY Each Reference Matters**:
  - `Train_ConvLSTM.py` is the file to modify — understanding every section is critical
  - `Train_Ai.py` shows the "correct" patterns for data handling, splitting, and checkpoint saving
  - `ModelManager` defines what checkpoint format is loadable

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Train_ConvLSTM.py imports work (no crash on import)
    Tool: Bash (python -c)
    Preconditions: Task 3 completed (ConvLSTM classes implemented)
    Steps:
      1. Run: python -c "
         from Train_ConvLSTM import train
         print('PASS: Train_ConvLSTM imports successfully')
         "
    Expected Result: Import succeeds (no TypeError from None classes)
    Failure Indicators: ImportError, TypeError
    Evidence: .sisyphus/evidence/task-8-import.txt

  Scenario: Channel count updated to 8
    Tool: Bash (python -c)
    Preconditions: None
    Steps:
      1. Run: python -c "
         with open('Train_ConvLSTM.py') as f:
             source = f.read()
         assert 'CHANNELS = 8' in source, f'CHANNELS not set to 8'
         print('PASS: CHANNELS = 8')
         "
    Expected Result: CHANNELS constant is 8
    Failure Indicators: Still 6
    Evidence: .sisyphus/evidence/task-8-channels.txt

  Scenario: 70/15/15 split implemented
    Tool: Bash (python -c)
    Preconditions: None
    Steps:
      1. Run: python -c "
         with open('Train_ConvLSTM.py') as f:
             source = f.read()
         assert '0.7' in source or 'train_ratio' in source, 'No 70% split found'
         assert 'int(0.8' not in source, 'Old 80/20 split still present'
         print('PASS: Split updated from 80/20')
         "
    Expected Result: 70/15/15 split or similar proper temporal split
    Failure Indicators: Old 0.8 split still present
    Evidence: .sisyphus/evidence/task-8-split.txt

  Scenario: Checkpoint saves required metadata
    Tool: Bash (python -c)
    Preconditions: None
    Steps:
      1. Run: python -c "
         with open('Train_ConvLSTM.py') as f:
             source = f.read()
         required_keys = ['normalization_mean', 'normalization_std', 'input_dim']
         for key in required_keys:
             assert key in source, f'Missing checkpoint key: {key}'
         print('PASS: Checkpoint metadata includes required keys')
         "
    Expected Result: All required metadata keys present in checkpoint save
    Failure Indicators: Missing keys
    Evidence: .sisyphus/evidence/task-8-checkpoint-meta.txt
  ```

  **Commit**: YES
  - Message: `fix(train): align ConvLSTM training with proper data pipeline`
  - Files: `Train_ConvLSTM.py`

---

- [x] 9. Update ModelManager for ConvLSTM Inference

  **What to do**:
  - Update `src/models/manager.py` to support ConvLSTM inference, not just loading:
    1. Add `predict_temperature(input_sequence)` method for ConvLSTM forward pass
       - Takes numpy array of shape `(1, seq_len, channels, H, W)` or torch tensor
       - Converts to tensor, runs `model.forward(x, future_seq=N)`, returns numpy
    2. Update `predict_event()` to handle ConvLSTM-based event detection
       - If model is ConvLSTM: run forward pass, check if max predicted temp exceeds threshold, return probability based on temperature
    3. Store normalization stats (mean, std) as instance attributes from checkpoint metadata
    4. Add `denormalize_temperature(normalized_grid, channel_idx=1)` helper
    5. Update `get_latest_checkpoint()` to also search for ConvLSTM checkpoints (`heatwave_convlstm_v*.pth`)
       - Prefer the most recent checkpoint regardless of type, OR prefer ConvLSTM if both exist

  **Must NOT do**:
  - Do NOT change the sklearn predict_proba path — it must still work for RF models
  - Do NOT remove backward compatibility with existing checkpoint formats

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Small additions to an existing manager class
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Task 10)
  - **Blocks**: Task 10 (api_server needs ModelManager capabilities)
  - **Blocked By**: Task 3 (ConvLSTM must be implemented first)

  **References**:

  **Pattern References**:
  - `src/models/manager.py:1-87` — ENTIRE file to update
  - `src/models/manager.py:20-26` — `get_latest_checkpoint()` — add ConvLSTM pattern
  - `src/models/manager.py:28-76` — `load_model()` — already handles both sklearn and PyTorch
  - `src/models/manager.py:77-85` — `predict_event()` — only handles sklearn, need ConvLSTM path

  **API/Type References**:
  - `src/models/convlstm.py` — The `HeatwaveConvLSTM` class API (from Task 3)
  - `Train_ConvLSTM.py:110-117` — ConvLSTM checkpoint format: `model_state_dict`, `mean`, `std`, `lats`, `lons`, `channels`

  **WHY Each Reference Matters**:
  - `ModelManager` is the abstraction layer between checkpoints and inference
  - Understanding both checkpoint formats is critical for loading and inference

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: ModelManager finds both RF and ConvLSTM checkpoints
    Tool: Bash (python -c)
    Preconditions: At least one RF checkpoint exists in models/
    Steps:
      1. Run: python -c "
         from src.models.manager import ModelManager
         mm = ModelManager()
         ckpt = mm.get_latest_checkpoint()
         print(f'Latest checkpoint: {ckpt}')
         assert ckpt is not None, 'No checkpoint found'
         print('PASS: Checkpoint discovery works')
         "
    Expected Result: Returns a path to a checkpoint
    Failure Indicators: Returns None when checkpoints exist
    Evidence: .sisyphus/evidence/task-9-discovery.txt

  Scenario: predict_temperature method exists
    Tool: Bash (python -c)
    Preconditions: None
    Steps:
      1. Run: python -c "
         from src.models.manager import ModelManager
         mm = ModelManager()
         assert hasattr(mm, 'predict_temperature'), 'No predict_temperature method'
         print('PASS: predict_temperature method exists')
         "
    Expected Result: Method exists
    Failure Indicators: AttributeError
    Evidence: .sisyphus/evidence/task-9-predict-temp.txt
  ```

  **Commit**: NO (groups with Task 10)

---

- [x] 10. Implement Dual-Model Inference in api_server.py

  **What to do**:
  - Update `load_resources()` to detect model type and configure inference accordingly:
    - If loaded model has `predict_proba` (sklearn) → RF inference path
    - If loaded model is `HeatwaveConvLSTM` → ConvLSTM inference path
    - Store `runtime_model_type` appropriately
  - Rewrite `get_prediction_sequence()` to branch on model type:
    - **RF path** (from Task 5): Use classifier probability + persistence baseline
    - **ConvLSTM path** (NEW):
      1. Convert input sequence to torch tensor: `x = torch.from_numpy(x_seq).unsqueeze(0).float().to(device)`
      2. Run forward pass: `with torch.no_grad(): output = model(x, future_seq=days)`
      3. Output shape: `(1, days, channels, H, W)`
      4. Extract temperature channel (index 1) and denormalize: `temp = output[0, :, 1, :, :] * std + mean`
      5. Convert Kelvin to Celsius if needed (check if mean > 200)
      6. Return list of temperature grids (one per day)
  - Update `load_resources()` to try loading ConvLSTM checkpoint first (from `heatwave_convlstm_v*.pth`), fall back to RF checkpoint
  - Ensure both prediction endpoints work correctly with either model type

  **Must NOT do**:
  - Do NOT run both models simultaneously — use whichever is loaded
  - Do NOT change API response format — same JSON structure regardless of model type
  - Do NOT modify training code

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Complex integration of two inference paths with proper tensor handling
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (sequential within wave — depends on Task 9)
  - **Blocks**: Task 11
  - **Blocked By**: Tasks 4, 5, 6, 8, 9 (all prior fixes and ConvLSTM support)

  **References**:

  **Pattern References**:
  - `api_server.py:820-855` — `get_prediction_sequence()` — rewritten in Task 5, extend for ConvLSTM
  - `api_server.py:620-807` — `load_resources()` — resource loading to update
  - `api_server.py:94-107` — `get_latest_model()` — add ConvLSTM checkpoint discovery
  - `api_server.py:64-77` — Global variables (model, runtime_model_type, etc.)

  **API/Type References**:
  - `src/models/manager.py` — `ModelManager` updated in Task 9 — can be used or logic duplicated in api_server.py
  - `src/models/convlstm.py` — `HeatwaveConvLSTM` forward API: `model(x, future_seq=N) → (batch, N, channels, H, W)`
  - `Train_ConvLSTM.py:110-117` — ConvLSTM checkpoint format for loading

  **WHY Each Reference Matters**:
  - `get_prediction_sequence` is the core function that needs dual-path logic
  - `load_resources` needs to handle both checkpoint types
  - Understanding ConvLSTM forward API is critical for correct tensor operations

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: api_server.py handles RF model loading (existing behavior)
    Tool: Bash (python -c)
    Preconditions: RF checkpoint exists in models/
    Steps:
      1. Run: python -c "
         with open('api_server.py') as f:
             source = f.read()
         assert 'predict_proba' in source, 'RF predict_proba call missing'
         assert 'heatwave_convlstm' in source.lower() or 'convlstm' in source.lower(), 'ConvLSTM path missing'
         print('PASS: Both model type paths present in code')
         "
    Expected Result: Both RF and ConvLSTM code paths exist
    Failure Indicators: Missing either path
    Evidence: .sisyphus/evidence/task-10-dual-path.txt

  Scenario: ConvLSTM inference path exists
    Tool: Bash (python -c)
    Preconditions: None
    Steps:
      1. Run: python -c "
         with open('api_server.py') as f:
             source = f.read()
         # Check for torch inference pattern
         assert 'torch.no_grad' in source, 'No torch.no_grad found (ConvLSTM inference needs this)'
         assert 'future_seq' in source, 'No future_seq parameter in inference'
         print('PASS: ConvLSTM inference patterns present')
         "
    Expected Result: Torch inference patterns found
    Failure Indicators: Missing torch.no_grad or future_seq
    Evidence: .sisyphus/evidence/task-10-convlstm-inference.txt
  ```

  **Commit**: YES (groups Tasks 9, 10)
  - Message: `feat(api): dual-model inference for RF and ConvLSTM`
  - Files: `api_server.py`, `src/models/manager.py`

---

- [x] 11. End-to-End API Verification

  **What to do**:
  - Start the Flask API server with the existing RF checkpoint
  - Hit every prediction endpoint and verify:
    - `/api/predict` — returns real date, model-derived probability, no dummy weather values
    - `/api/forecast` — returns 7 days with real dates, no hardcoded weather values, model probability
    - `/api/map` — returns valid GeoJSON with temperature values in reasonable range
    - `/api/health` — returns model_loaded=true and correct model_type
  - Verify temperature values are in a reasonable range for Thailand (20-50°C)
  - Verify no dummy patterns remain in responses
  - Save all responses as evidence

  **Must NOT do**:
  - Do NOT modify any code — this is verification only
  - Do NOT run training

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Integration testing requiring server management and multiple endpoint checks
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 4 (solo)
  - **Blocks**: F1-F4 (final verification)
  - **Blocked By**: Tasks 5, 6, 7, 10 (all API changes must be complete)

  **References**:

  **Pattern References**:
  - `.sisyphus/evidence/task-1-*.json` — Baseline responses from Task 1 to compare against
  - `api_server.py:858-899` — `/api/predict` endpoint
  - `api_server.py:902-957` — `/api/forecast` endpoint
  - `api_server.py:960-1023` — `/api/map` endpoint
  - `api_server.py:1113-1116` — `/api/health` endpoint

  **WHY Each Reference Matters**:
  - Baselines from Task 1 let us diff before/after to prove changes worked
  - Endpoint code tells us what to expect in responses

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: All prediction endpoints return valid data
    Tool: Bash (curl + python)
    Preconditions: Model checkpoint exists, server can start
    Steps:
      1. Start server: python api_server.py &
      2. Wait 15 seconds for startup and data loading
      3. Test health: curl -s http://localhost:5000/api/health | python -c "
         import sys,json; d=json.load(sys.stdin)
         assert d['status']=='ok', f'Health not ok: {d}'
         assert d['model_loaded']==True, 'Model not loaded'
         print(f'Health OK, model_type={d.get(\"model_type\", \"unknown\")}')
         "
      4. Test predict: curl -s http://localhost:5000/api/predict | python -c "
         import sys,json; d=json.load(sys.stdin)
         assert '2000' not in d.get('date',''), f'Still dummy date: {d[\"date\"]}'
         assert d['weather']['T2M_MAX'] > 20 and d['weather']['T2M_MAX'] < 60, f'Temp out of range: {d[\"weather\"][\"T2M_MAX\"]}'
         prob = d.get('probability', -1)
         assert prob not in [0.1, 0.5, 0.8, 0.95], f'Static probability: {prob}'
         print(f'Predict OK: date={d[\"date\"]}, T2M_MAX={d[\"weather\"][\"T2M_MAX\"]:.1f}, prob={prob:.4f}')
         " 
      5. Test forecast: curl -s http://localhost:5000/api/forecast | python -c "
         import sys,json; d=json.load(sys.stdin)
         assert len(d['forecasts'])==7, f'Expected 7 days, got {len(d[\"forecasts\"])}'
         f0 = d['forecasts'][0]
         assert f0['weather'].get('RH2M') is None, f'RH2M not None: {f0[\"weather\"].get(\"RH2M\")}'
         assert f0['weather'].get('WS10M') is None, f'WS10M not None: {f0[\"weather\"].get(\"WS10M\")}'
         print(f'Forecast OK: 7 days, no dummy weather')
         "
      6. Test map: curl -s http://localhost:5000/api/map | python -c "
         import sys,json; d=json.load(sys.stdin)
         assert d['type']=='FeatureCollection'
         assert len(d['features']) > 0, 'No features'
         temps = [f['properties']['temperature'] for f in d['features']]
         assert min(temps) > -10 and max(temps) < 70, f'Temp range unreasonable: {min(temps)}-{max(temps)}'
         print(f'Map OK: {len(d[\"features\"])} features, temp range {min(temps):.1f}-{max(temps):.1f}')
         "
      7. Kill server
    Expected Result: All 4 endpoints return valid, non-dummy data
    Failure Indicators: Dummy date, static probability, hardcoded weather values, out-of-range temperatures
    Evidence: .sisyphus/evidence/task-11-predict.json, task-11-forecast.json, task-11-map.json, task-11-health.json

  Scenario: Compare with baselines from Task 1
    Tool: Bash (python -c)
    Preconditions: Both baseline and new evidence files exist
    Steps:
      1. Compare .sisyphus/evidence/task-1-predict.json with task-11-predict.json
      2. Verify dummy values are gone, real values present
    Expected Result: New responses have real data where baselines had dummies
    Evidence: .sisyphus/evidence/task-11-comparison.txt
  ```

  **Evidence to Capture:**
  - [ ] task-11-health.json
  - [ ] task-11-predict.json
  - [ ] task-11-forecast.json
  - [ ] task-11-map.json
  - [ ] task-11-comparison.txt

  **Commit**: YES
  - Message: `test: end-to-end API verification after accuracy overhaul`
  - Files: `.sisyphus/evidence/task-11-*`

---

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

> 4 review agents run in PARALLEL. ALL must APPROVE. Rejection → fix → re-run.

- [x] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists (read file, curl endpoint, run command). For each "Must NOT Have": search codebase for forbidden patterns — reject with file:line if found. Check evidence files exist in .sisyphus/evidence/. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [x] F2. **Code Quality Review** — `unspecified-high`
  Run linter if available. Review all changed files for: `as any`/type ignoring, empty try/except, print statements in production code, commented-out code, unused imports. Check for AI slop: excessive comments, over-abstraction, generic variable names. Verify no hardcoded dummy values remain (search for `"2000-XX-XX"`, `RH2M.*60`, `WS10M.*2.5`, `NDVI.*0.5`).
  Output: `Lint [PASS/FAIL] | Files [N clean/N issues] | Dummy Values [CLEAN/N found] | VERDICT`

- [x] F3. **Real Manual QA** — `unspecified-high`
  Start Flask server. Hit every prediction endpoint: `/api/predict`, `/api/forecast`, `/api/map`, `/api/health`. Verify: no dummy dates, no static probabilities, temperature values in reasonable range (20-50°C for Thailand), GeoJSON features have valid coordinates, forecast has 7 days. Test error cases: what happens if model isn't loaded?
  Output: `Endpoints [N/N pass] | Data Quality [N/N] | Error Handling [N tested] | VERDICT`

- [x] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", check actual changes. Verify 1:1 — everything in spec was built (no missing), nothing beyond spec was built (no creep). Check "Must NOT do" compliance. Specifically verify: `create_sequences` signature unchanged, API response JSON keys unchanged, `Train_Ai.py` unmodified, no new dependencies added. Flag unaccounted changes.
  Output: `Tasks [N/N compliant] | Scope [CLEAN/N issues] | VERDICT`

---

## Commit Strategy

| After Task(s) | Commit Message | Key Files |
|---|---|---|
| 1 | `chore: capture baseline behavior for regression checks` | `.sisyphus/evidence/` |
| 2 | `fix(data): compute normalization stats on training split only` | `src/data/loader.py` |
| 3 | `feat(model): implement ConvLSTM architecture` | `src/models/convlstm.py` |
| 4, 5, 6, 7 | `fix(api): remove fake inference, use real model output, clean dummy data` | `api_server.py` |
| 8 | `fix(train): align ConvLSTM training with proper data pipeline` | `Train_ConvLSTM.py` |
| 9, 10 | `feat(api): dual-model inference for RF and ConvLSTM` | `api_server.py`, `src/models/manager.py` |
| 11 | `test: end-to-end API verification` | `.sisyphus/evidence/` |

---

## Success Criteria

### Verification Commands
```bash
# ConvLSTM classes importable and functional
python -c "from src.models.convlstm import HeatwaveConvLSTM, PhysicsInformedLoss; import torch; m=HeatwaveConvLSTM(8,[32,32],[(3,3),(3,3)],2); x=torch.randn(1,5,8,16,16); print(m(x,future_seq=2).shape)"
# Expected: torch.Size([1, 2, 8, 16, 16])

# No dummy values in predict endpoint
curl -s http://localhost:5000/api/predict | python -c "import sys,json; d=json.load(sys.stdin); assert '2000' not in d.get('date',''), 'Dummy date'; print('OK')"

# No hardcoded weather in forecast
curl -s http://localhost:5000/api/forecast | python -c "import sys,json; d=json.load(sys.stdin); w=d['forecasts'][0]['weather']; assert w.get('RH2M') is None, f'RH2M={w.get(\"RH2M\")}'; print('OK')"

# Training script imports work
python -c "from Train_ConvLSTM import train; print('Import OK')"
```

### Final Checklist
- [x] All "Must Have" present
- [x] All "Must NOT Have" absent
- [x] No dummy/hardcoded values in API responses
- [x] ConvLSTM architecture implemented and importable
- [x] Normalization computed on training split only
- [x] Inference split matches training split (70/15/15)
- [x] Backward-compatible checkpoint loading works
