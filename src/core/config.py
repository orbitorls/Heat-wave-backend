from pathlib import Path
from typing import Any, Dict, Tuple

from pydantic_settings import BaseSettings, SettingsConfigDict


def _load_yaml_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml
    except Exception:
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return raw if isinstance(raw, dict) else {}


class Settings(BaseSettings):
    # Project Paths
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "era5_data"
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    OUTPUT_DIR: Path = PROJECT_ROOT / "output"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"
    CONFIG_PATH: Path = PROJECT_ROOT / "config" / "config.yaml"

    # Data Settings (Thailand Bounds)
    LAT_BOUNDS: Tuple[float, float] = (5.0, 21.0)
    LON_BOUNDS: Tuple[float, float] = (97.0, 106.0)

    # Model Settings
    DEFAULT_MODEL_VERSION: str = "latest"
    SEQUENCE_LENGTH: int = 7
    PREDICTION_HORIZON: int = 2

    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 5000
    DEBUG: bool = False

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    def model_post_init(self, __context):
        cfg = _load_yaml_config(self.CONFIG_PATH)

        api_cfg = cfg.get("api", {}) if isinstance(cfg.get("api"), dict) else {}
        if "host" in api_cfg:
            self.API_HOST = str(api_cfg["host"])
        if "port" in api_cfg:
            try:
                self.API_PORT = int(api_cfg["port"])
            except Exception:
                pass
        if "debug" in api_cfg:
            self.DEBUG = bool(api_cfg["debug"])

        model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}
        if "seq_len" in model_cfg:
            try:
                self.SEQUENCE_LENGTH = int(model_cfg["seq_len"])
            except Exception:
                pass
        if "future_seq" in model_cfg:
            try:
                self.PREDICTION_HORIZON = int(model_cfg["future_seq"])
            except Exception:
                pass


settings = Settings()
project_config: Dict[str, Any] = _load_yaml_config(settings.CONFIG_PATH)
