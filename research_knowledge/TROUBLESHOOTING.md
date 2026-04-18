# AI TROUBLESHOOTING GUIDE: GPU, CLI & ERRORS

**Objective:** Ensure the system is robust, handles hardware limitations gracefully, and provides clear diagnostics.

## 🔌 1. GPU Detection & Bypass Strategy
- **Issue:** `torch.cuda.is_available()` returns `False` despite having a GPU.
- **Root Cause:** Incompatible CUDA toolkit versions or missing drivers in the environment.
- **AI Directive (Bypass):**
  - Use a global device selector that checks for an environment variable:
    ```python
    import os
    import torch
    
    # Bypass logic
    force_cpu = os.environ.get("HW_FORCE_CPU", "0") == "1"
    if force_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ```
  - **Action:** Ensure all training scripts (`Train_ConvLSTM.py`) and API servers use this centralized logic.

## 🖥️ 2. CLI Enhancements (Look & Stability)
- **Issue:** CLI feels "buggy" or crashes on network errors.
- **Improvement Strategy:**
  - **Visuals:** Use `rich.table.Table` for checkpoints and `rich.live.Live` for training progress.
  - **Error Handling:** Wrap all HTTP calls (`urllib`) in `try-except` blocks with meaningful user messages.
  - **Non-Blocking:** For long operations (like download), use status indicators (`console.status("[bold green]Downloading...")`).

## ⚠️ 3. Handling Log Errors (CDS API 500)
- **Issue:** `Internal Server Error 500` during ERA5 download.
- **Context:** This is a server-side issue at Copernicus (Europe), common during high load.
- **AI Directive:**
  - Do not treat 500 errors as code bugs.
  - Implement "Checkpointing" in `download_era5.py`: Save metadata after each successful year so the script can resume exactly where it left off.
  - Use `sleep` with exponential backoff (e.g., wait 2m, then 4m, then 8m).

## 🌳 4. Balanced Random Forest Integration
- **Concept:** When using `RandomForestRegressor`, the model struggles with rare heatwaves.
- **AI Directive:** 
  - Switch to `BalancedRandomForestClassifier` from `imblearn.ensemble`.
  - Use `sampling_strategy='auto'` to ensure minority class (heatwaves) is balanced in every tree.
  - Set `class_weight='balanced_subsample'`.
