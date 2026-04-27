import os
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

    # Training Hyperparameters
    BATCH_SIZE: int = 4
    SEQ_LEN: int = 5
    FUTURE_SEQ: int = 2
    LEARNING_RATE: float = 0.001
    EPOCHS: int = 10
    RANDOM_SEED: int = 42

    # XGBoost/RandomForest Hyperparameters
    RF_N_ESTIMATORS: int = 260
    RF_MAX_DEPTH: int = 25
    RF_MIN_SAMPLES_LEAF: int = 2
    MIN_TRAIN_POSITIVE_RATE: float = 0.08
    MAX_TRAIN_POSITIVE_RATE: float = 0.40

    # Heatwave Detection
    HEATWAVE_TEMP_THRESHOLD: float = 38.0
    HEATWAVE_ANOMALY_THRESHOLD: float = 6.0
    HEATWAVE_MIN_DURATION: int = 3
    EVENT_MIN_DURATION_DAYS: int = 3
    EVENT_MIN_HOT_FRACTION: float = 0.10

    # Data Split Ratios
    TRAIN_RATIO: float = 0.75
    VAL_RATIO: float = 0.10
    TEST_RATIO: float = 0.15

    # Training Options
    USE_XGBOOST: bool = True
    USE_LIGHTGBM: bool = True
    FEATURE_ENGINEERING_ENABLED: bool = False
    WALK_FORWARD_ENABLED: bool = False

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
