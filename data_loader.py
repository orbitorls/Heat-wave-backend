"""
DEPRECATED — do not use in new code.
All logic lives in src/data/loader.py. Use: from src.data.loader import ...
This file exists only for backward compatibility with legacy callers.
"""
import warnings
warnings.warn(
    "data_loader.py at repo root is deprecated. Use 'from src.data.loader import ...' instead.",
    DeprecationWarning,
    stacklevel=2,
)
from src.data.loader import (  # noqa: F401
    DataLoader,
    create_sequences,
    clean_data,
    compute_normalization_stats,
    fill_nan_along_time as _fill_nan_along_time,
    load_era5_data,
    normalize_data,
    EPSILON,
    DYNAMIC_VARIABLE_SPECS,
)

# Legacy alias
_fill_nan_along_time = _fill_nan_along_time
