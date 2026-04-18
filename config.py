import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

_config_path = Path("config/config.yaml")
_yaml = {}
if _config_path.exists():
    with open(_config_path, encoding="utf-8") as f:
        _yaml = yaml.safe_load(f) or {}

DATA_DIR: str = os.getenv("DATA_DIR", _yaml.get("data_dir", "era5_data"))
MODELS_DIR: str = os.getenv("MODELS_DIR", _yaml.get("models_dir", "models"))
PORT: int = int(os.getenv("PORT", _yaml.get("server", {}).get("port", 5000)))
DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
API_KEY: str = os.getenv("API_KEY", "")
CORS_ORIGINS: str = os.getenv("CORS_ORIGINS", "*")
