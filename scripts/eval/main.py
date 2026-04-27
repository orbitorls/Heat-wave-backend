import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

# Structured logger for the entrypoint
logger = logging.getLogger(__name__)


def _load_structured_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    if cfg_path.suffix.lower() == ".json":
        return json.loads(cfg_path.read_text(encoding="utf-8"))

    if cfg_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except Exception as exc:
            raise RuntimeError(
                "PyYAML is required for YAML config loading. Install `pyyaml`."
            ) from exc
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError("Top-level config must be a mapping/dict")
        return data

    raise ValueError(f"Unsupported config extension: {cfg_path.suffix}")


def _extract_training_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map PROJECT_CONTEXT-style config.yaml into Train_Ai.py runtime config keys.
    Unknown keys are ignored safely.
    """
    if not cfg:
        return {}

    out: Dict[str, Any] = {}
    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data"), dict) else {}
    training_cfg = cfg.get("training", {}) if isinstance(cfg.get("training"), dict) else {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}

    mapping = {
        "batch_size": "batch_size",
        "seq_len": "seq_len",
        "future_seq": "future_seq",
        "epochs": "epochs",
        "learning_rate": "learning_rate",
        "train_ratio": "train_ratio",
        "val_ratio": "val_ratio",
        "heatwave_percentile": "heatwave_percentile",
        "clip_low_percentile": "clip_low_percentile",
        "clip_high_percentile": "clip_high_percentile",
        "event_min_duration_days": "event_min_duration_days",
        "event_min_hot_fraction": "event_min_hot_fraction",
        "allow_sample_mean_fallback": "allow_sample_mean_fallback",
        "require_dynamic_features": "require_dynamic_features",
        "min_train_positive_rate": "min_train_positive_rate",
        "max_train_positive_rate": "max_train_positive_rate",
        "min_eval_positive_count": "min_eval_positive_count",
        "rf_n_estimators": "rf_n_estimators",
        "rf_max_depth": "rf_max_depth",
        "rf_min_samples_leaf": "rf_min_samples_leaf",
        "rf_sampling_strategy": "rf_sampling_strategy",
        "rf_replacement": "rf_replacement",
    }

    for src_key, dst_key in mapping.items():
        if src_key in training_cfg:
            out[dst_key] = training_cfg[src_key]
        elif src_key in model_cfg:
            out[dst_key] = model_cfg[src_key]
        elif src_key in data_cfg:
            out[dst_key] = data_cfg[src_key]

    # Keep label mode as metadata-compatible passthrough for future pipeline integration.
    if "labeling_method" in data_cfg:
        out["labeling_method"] = data_cfg["labeling_method"]
    if "heatwave_heat_index_threshold" in data_cfg:
        out["heatwave_heat_index_threshold"] = data_cfg["heatwave_heat_index_threshold"]
    if "heatwave_temperature_threshold" in data_cfg:
        out["heatwave_temperature_threshold"] = data_cfg["heatwave_temperature_threshold"]

    return out


def run_train(config_path: Optional[str]) -> None:
    from Train_Ai import train

    project_cfg = _load_structured_config(config_path)
    training_cfg = _extract_training_config(project_cfg)
    logger.info("Starting training (PROJECT_CONTEXT aligned mode)...")
    result = train(config=training_cfg or None)
    if not result:
        raise RuntimeError("Training failed or returned empty result.")
    logger.info(json.dumps(result, indent=2, ensure_ascii=False, default=str))


def run_dashboard(host: str, port: int, debug: bool) -> None:
    from api_server import app, load_resources

    if not load_resources():
        logger.warning("Warning: model/data not fully loaded; dashboard still starts.")
    logger.info(f"Starting dashboard at http://{host}:{port}/trainer")
    app.run(host=host, port=port, debug=debug)


def run_predict(input_path: str, proba: bool) -> None:
    import numpy as np
    import pandas as pd
    from src.models.manager import model_manager

    if not model_manager.load_model():
        raise RuntimeError("No loadable checkpoint found in models directory.")

    inp = Path(input_path)
    if not inp.exists():
        raise FileNotFoundError(f"Input file not found: {inp}")

    if inp.suffix.lower() in {".csv", ".txt"}:
        frame = pd.read_csv(inp)
    else:
        raise ValueError("Predict mode currently supports CSV input only.")

    x = frame.select_dtypes(include=["number"]).to_numpy(dtype=np.float32)
    if x.size == 0:
        raise ValueError("Input CSV must contain numeric feature columns.")

    if hasattr(model_manager.model, "predict_proba"):
        probs = model_manager.model.predict_proba(x)[:, 1]
        if proba:
            print(json.dumps({"probabilities": probs.tolist()}, ensure_ascii=False))
        else:
            labels = (probs >= 0.5).astype(int)
            print(json.dumps({"labels": labels.tolist()}, ensure_ascii=False))
        return

    if hasattr(model_manager.model, "predict"):
        pred = model_manager.model.predict(x)
        print(json.dumps({"predictions": np.asarray(pred).tolist()}, ensure_ascii=False))
        return

    raise RuntimeError("Loaded model does not support predict/predict_proba.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="HEATWAVE-AI unified entrypoint (PROJECT_CONTEXT aligned)"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "dashboard", "predict", "cli", "web"],
        default="web",
        help="Execution mode (web is recommended)",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to JSON/YAML config file used in train mode",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Dashboard host")
    parser.add_argument("--port", type=int, default=5000, help="Dashboard port")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    parser.add_argument("--input", help="Input CSV path for predict mode")
    parser.add_argument(
        "--proba",
        action="store_true",
        help="Output probabilities in predict mode",
    )
    parser.add_argument(
        "--model",
        default="balanced_rf",
        help="Model selector placeholder for PROJECT_CONTEXT compatibility",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "train":
        run_train(args.config)
    elif args.mode in {"dashboard", "web"}:
        run_dashboard(args.host, args.port, args.debug)
    elif args.mode == "predict":
        if not args.input:
            raise SystemExit("--input is required when --mode predict")
        run_predict(args.input, args.proba)
    else:
        from src.cli.main import cli

        cli()
