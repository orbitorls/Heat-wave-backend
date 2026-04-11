import torch
import shutil
import subprocess
import numpy as np
from typing import Dict, Tuple

def detect_gpu_capability() -> Dict:
    """Detect if GPU is available via torch or nvidia-smi."""
    torch_cuda = bool(torch.cuda.is_available())
    nvidia_smi = False
    
    candidates = [
        "nvidia-smi",
        r"C:\Windows\System32\nvidia-smi.exe",
    ]
    
    found = shutil.which("nvidia-smi")
    if found:
        candidates.insert(0, found)

    for candidate in candidates:
        try:
            proc = subprocess.run(
                [candidate, "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                nvidia_smi = True
                break
        except Exception:
            continue

    return {
        "gpu_detected": bool(torch_cuda or nvidia_smi),
        "torch_cuda": torch_cuda,
        "nvidia_smi": nvidia_smi,
    }

def get_risk_level(max_temp: float) -> Tuple[str, float]:
    """Calculate risk level and probability based on max temperature."""
    if max_temp >= 41:
        return "CRITICAL", 0.95
    if max_temp >= 38:
        return "HIGH", 0.8
    if max_temp >= 35:
        return "MEDIUM", 0.5
    return "LOW", 0.1

def to_jsonable(value):
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
