import torch
import glob
import os
import pickle
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Union
from src.core.config import settings
from src.core.logger import logger
from src.models.convlstm import HeatwaveConvLSTM

class ModelManager:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[Any] = None
        self.metadata: Dict[str, Any] = {}
        self.current_path: Optional[Path] = None
        self.model_type: str = "unknown"
        self.normalization_mean: Optional[np.ndarray] = None
        self.normalization_std: Optional[np.ndarray] = None

    def get_latest_checkpoint(self) -> Optional[Path]:
        rf_files = list(settings.MODELS_DIR.glob("heatwave_model_checkpoint_v*.pth"))
        convlstm_files = list(settings.MODELS_DIR.glob("heatwave_convlstm_v*.pth"))
        all_files = rf_files + convlstm_files
        if not all_files:
            return None
        def get_version(f: Path):
            try:
                stem = f.stem.split("_v")[-1]
                return int(stem)
            except: return 0
        return max(all_files, key=get_version)

    def load_model(self, checkpoint_path: Optional[Path] = None) -> bool:
        if checkpoint_path is None:
            checkpoint_path = self.get_latest_checkpoint()

        if not checkpoint_path or not checkpoint_path.exists():
            logger.warning("Target checkpoint not found.")
            return False

        try:
            logger.info(f"Attempting to load model from {checkpoint_path}...")
            # Use weights_only=False to allow loading of Sklearn models (Pickle)
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            self.metadata = checkpoint.get("metadata", {})

            norm_mean = self.metadata.get("normalization_mean") or checkpoint.get("normalization_mean")
            norm_std = self.metadata.get("normalization_std") or checkpoint.get("normalization_std")
            if norm_mean is not None:
                self.normalization_mean = np.asarray(norm_mean)
            else:
                self.normalization_mean = None
            if norm_std is not None:
                self.normalization_std = np.asarray(norm_std)
            else:
                self.normalization_std = None

            self.model_type = checkpoint.get("model_type", "unknown")

            # Case 1: Sklearn-based Model (RF / XGBoost)
            if "sklearn_model" in checkpoint:
                self.model = checkpoint["sklearn_model"]
                logger.info(f"Successfully loaded Sklearn model type: {self.model_type}")
            
            # Case 2: PyTorch-based Model (ConvLSTM)
            elif "model_state_dict" in checkpoint:
                input_dim = self.metadata.get("input_dim", 8)
                self.model = HeatwaveConvLSTM(
                    input_dim=input_dim,
                    hidden_dim=self.metadata.get("hidden_dim", [16, 16]),
                    kernel_size=self.metadata.get("kernel_size", [(3, 3), (3, 3)]),
                    num_layers=self.metadata.get("num_layers", 2)
                ).to(self.device)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.model.eval()
                logger.info("Successfully loaded PyTorch ConvLSTM model")
            
            # Case 3: Fallback for older checkpoints that might just be the model itself
            elif hasattr(checkpoint, "predict_proba"):
                self.model = checkpoint
                logger.info("Loaded model via direct fallback")
            
            else:
                logger.error(f"Checkpoint keys: {list(checkpoint.keys())}")
                raise ValueError("Checkpoint format is not recognized (no sklearn_model or model_state_dict)")

            self.current_path = checkpoint_path
            return True
        except Exception as e:
            logger.error(f"Critical Failure in Model Loading: {e}")
            return False

    def denormalize_temperature(self, normalized_grid: np.ndarray, channel_idx: int = 1) -> np.ndarray:
        """Denormalize a temperature grid using stored normalization stats.

        Args:
            normalized_grid: Array of shape (H, W) or (N, H, W)
            channel_idx: Which channel index holds temperature (default: 1 for t2m)

        Returns:
            Denormalized array in the same shape.
        """
        if self.normalization_mean is None or self.normalization_std is None:
            return normalized_grid

        mean_val = float(np.take(self.normalization_mean.squeeze(), channel_idx))
        std_val = float(np.take(self.normalization_std.squeeze(), channel_idx))

        result = normalized_grid * std_val + mean_val
        if np.nanmean(result) > 200:
            result = result - 273.15
        return result

    def predict_temperature(self, input_sequence: np.ndarray, future_seq: int = 1) -> np.ndarray:
        """Run ConvLSTM forward pass to predict temperature grids.

        Args:
            input_sequence: numpy array of shape (1, seq_len, channels, H, W)
            future_seq: number of future time steps to predict

        Returns:
            numpy array of shape (future_seq, H, W) with temperature in Celsius

        Raises:
            RuntimeError: if model is not a ConvLSTM or is not loaded
        """
        if self.model is None:
            raise RuntimeError("No model loaded")
        if not isinstance(self.model, HeatwaveConvLSTM):
            raise RuntimeError("predict_temperature requires a ConvLSTM model")

        x = torch.from_numpy(input_sequence).float().to(self.device)
        with torch.no_grad():
            output = self.model(x, future_seq=future_seq)
        temp_channel = output[0, :, 1, :, :].cpu().numpy()
        return self.denormalize_temperature(temp_channel, channel_idx=1)

    def predict_event(self, features_flat: np.ndarray) -> Dict:
        if self.model is None: return {"error": "Model not loaded"}
        try:
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(features_flat)[:, 1]
                return {"probabilities": probs.tolist()}
            if isinstance(self.model, HeatwaveConvLSTM):
                temp_grids = self.predict_temperature(features_flat, future_seq=1)
                max_temp = float(np.nanmax(temp_grids))
                prob = min(1.0, max(0.0, (max_temp - 30.0) / 15.0))
                return {"probabilities": [prob]}
            return {"error": "Model does not support event prediction"}
        except Exception as e:
            return {"error": str(e)}

model_manager = ModelManager()
