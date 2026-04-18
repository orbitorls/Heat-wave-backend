# SKILL: System Diagnostics & GPU Verification

**Description:** Procedures to verify hardware readiness and software environment stability for AI operations.

## 📝 Procedural Guidance

### GPU/CUDA Verification
- **Primary Tool:** `nvidia-smi` (if on NVIDIA).
- **AI Verification (PyTorch):**
  - `import torch; print(torch.cuda.is_available())`.
  - Check `torch.version.cuda` to match installed toolkit.
- **Bypass Procedure:** If `cuda.is_available()` fails, enforce CPU training to allow development to continue.

### CLI Health Checks
- Verify if `rich` and `prompt_toolkit` are installed.
- Ensure all API endpoints (`api_server.py`) return valid JSON.
- Test connection to the Flask API (`http://127.0.0.1:5000/api/health`) before starting CLI diagnostics.

### Networking Stability
- If downloading from CDS (ERA5), ensure `.cdsapirc` exists and contains valid credentials.
- Use `urllib` timeouts to prevent CLI from hanging during long server waits.

## 🛠️ Resources
- **Reference Code:** `heatwave_cli.py` (Diagnostics section).
- **Target Script:** `system` command in CLI.
- **Library:** `torch`, `rich`.
