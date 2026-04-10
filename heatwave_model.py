"""
DEPRECATED — do not use in new code.
All logic lives in src/models/convlstm.py. Use: from src.models.convlstm import ...
This file exists only for backward compatibility with legacy callers.
"""
import warnings
warnings.warn(
    "heatwave_model.py at repo root is deprecated. Use 'from src.models.convlstm import ...' instead.",
    DeprecationWarning,
    stacklevel=2,
)
import torch

from src.models.convlstm import (  # noqa: F401
    ConvLSTMCell,
    HeatwaveConvLSTM,
    PhysicsInformedLoss,
)

# Legacy module-level device variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
