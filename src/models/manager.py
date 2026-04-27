import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from src.core.config import settings
from src.core.logger import logger

# Lazy import ConvLSTM to speed up initial load
HeatwaveConvLSTM = None


class CheckpointLoadError(Exception):
    """Base exception for checkpoint loading failures."""
    pass


class CheckpointNotFoundError(CheckpointLoadError):
    """Checkpoint file does not exist."""
    pass


class CheckpointFormatError(CheckpointLoadError):
    """Checkpoint has unrecognized format."""
    pass


class CheckpointCorruptedError(CheckpointLoadError):
    """Checkpoint file is corrupted or invalid."""
    pass

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
        patterns = [
            "heatwave_model_checkpoint_v*.pth",
            "heatwave_convlstm_v*.pth",
            "heatwave_daily_xgboost_v*.pth",
        ]
        all_files = []
        for pattern in patterns:
            all_files.extend(settings.MODELS_DIR.glob(pattern))
        if not all_files:
            return None
        def get_version(f: Path):
            try:
                stem = f.stem.split("_v")[-1]
                return int(stem)
            except (ValueError, IndexError):
                return 0
        return max(all_files, key=get_version)

    def load_model(self, checkpoint_path: Optional[Path] = None) -> bool:
        if checkpoint_path is None:
            checkpoint_path = self.get_latest_checkpoint()

        if not checkpoint_path:
            logger.warning("No checkpoint available to load.")
            return False
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint file not found: {checkpoint_path}")
            return False

        try:
            logger.info(f"Attempting to load model from {checkpoint_path}...")
            # Use weights_only=False to allow loading of Sklearn models (Pickle)
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            if not isinstance(checkpoint, dict):
                raise CheckpointFormatError("Checkpoint is not a dictionary")
            
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
                if not hasattr(self.model, "predict_proba") and not hasattr(self.model, "predict"):
                    raise CheckpointFormatError("Loaded sklearn_model has no predict methods")
                logger.info(f"Successfully loaded Sklearn model type: {self.model_type}")
            
            # Case 2: PyTorch-based Model (ConvLSTM)
            elif "model_state_dict" in checkpoint:
                # Lazy import ConvLSTM only when needed
                global HeatwaveConvLSTM
                if HeatwaveConvLSTM is None:
                    from src.models.convlstm import HeatwaveConvLSTM as _ConvLSTM
                    HeatwaveConvLSTM = _ConvLSTM
                
                input_dim = self.metadata.get("input_dim", 8)
                if not isinstance(input_dim, int) or input_dim <= 0:
                    raise CheckpointFormatError(f"Invalid input_dim: {input_dim}")
                
                self.model = HeatwaveConvLSTM(
                    input_dim=input_dim,
                    hidden_dim=self.metadata.get("hidden_dim", [16, 16]),
                    kernel_size=self.metadata.get("kernel_size", [(3, 3), (3, 3)]),
                    num_layers=self.metadata.get("num_layers", 2)
                ).to(self.device)
                
                try:
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                    self.model.eval()
                except Exception as e:
                    raise CheckpointCorruptedError(f"Failed to load state_dict: {e}")
                
                logger.info("Successfully loaded PyTorch ConvLSTM model")
            
            # Case 3: Fallback for older checkpoints that might just be the model itself
            elif hasattr(checkpoint, "predict_proba") or hasattr(checkpoint, "predict"):
                self.model = checkpoint
                logger.info("Loaded model via direct fallback")
            
            else:
                available_keys = list(checkpoint.keys())
                raise CheckpointFormatError(
                    f"Checkpoint format not recognized. Available keys: {available_keys}. "
                    f"Expected 'sklearn_model', 'model_state_dict', or a model object with predict methods."
                )

            self.current_path = checkpoint_path
            return True
        except CheckpointLoadError as e:
            logger.error(f"Checkpoint load error: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error loading model: {type(e).__name__}: {e}")
            return False

    def denormalize_temperature(self, normalized_grid: np.ndarray, channel_idx: int = 1) -> np.ndarray:
        """Denormalize a temperature grid using stored normalization stats.

        Args:
            normalized_grid: Array of shape (H, W) or (N, H, W)
            channel_idx: Which channel index holds temperature (default: 1 for t2m)

        Returns:
            Denormalized array in the same shape.

        Raises:
            ValueError: if channel_idx is out of bounds or normalization stats are invalid.
        """
        if self.normalization_mean is None or self.normalization_std is None:
            return normalized_grid

        # Validate normalization stats shape
        norm_shape = self.normalization_mean.shape
        if len(norm_shape) == 0:
            raise ValueError("normalization_mean is a scalar, expected array with shape (C,) or (1, C)")
        if norm_shape[-1] == 0:
            raise ValueError(f"normalization_mean has invalid shape {norm_shape}, expected at least 1 channel")

        num_channels = norm_shape[-1] if len(norm_shape) > 0 else 1

        # Bounds check for channel_idx
        if channel_idx < 0:
            raise ValueError(f"channel_idx must be non-negative, got {channel_idx}")
        if channel_idx >= num_channels:
            raise ValueError(f"channel_idx {channel_idx} exceeds available channels {num_channels}")

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
        
        # Lazy import check
        global HeatwaveConvLSTM
        if HeatwaveConvLSTM is None:
            from src.models.convlstm import HeatwaveConvLSTM as _ConvLSTM
            HeatwaveConvLSTM = _ConvLSTM
        
        if not isinstance(self.model, HeatwaveConvLSTM):
            raise RuntimeError("predict_temperature requires a ConvLSTM model")

        x = torch.from_numpy(input_sequence).float().to(self.device)
        with torch.no_grad():
            output = self.model(x, future_seq=future_seq)
        temp_channel = output[0, :, 1, :, :].cpu().numpy()
        return self.denormalize_temperature(temp_channel, channel_idx=1)

    def predict_event(self, features_flat: np.ndarray) -> Dict[str, Any]:
        if self.model is None: return {"error": "Model not loaded"}
        try:
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(features_flat)[:, 1]
                return {"probabilities": probs.tolist()}
            
            # Lazy import check
            global HeatwaveConvLSTM
            if HeatwaveConvLSTM is None:
                from src.models.convlstm import HeatwaveConvLSTM as _ConvLSTM
                HeatwaveConvLSTM = _ConvLSTM
            
            if isinstance(self.model, HeatwaveConvLSTM):
                temp_grids = self.predict_temperature(features_flat, future_seq=1)
                max_temp = float(np.nanmax(temp_grids))
                prob = min(1.0, max(0.0, (max_temp - 30.0) / 15.0))
                return {"probabilities": [prob]}
            return {"error": "Model does not support event prediction"}
        except Exception as e:
            return {"error": str(e)}

model_manager = ModelManager()
