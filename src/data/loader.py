import xarray as xr
import numpy as np
import os
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from src.core.config import settings
from src.core.logger import logger

EPSILON = 1e-6

# ERA5 variable specifications
DYNAMIC_VARIABLE_SPECS = [
    ("z", ["z", "geopotential"]),
    ("t2m", ["t2m", "2m_temperature"]),
    ("swvl1", ["swvl1", "volumetric_soil_water_layer_1"]),
    ("tp", ["tp", "total_precipitation"]),
    ("humidity", ["rh2m", "rh", "r", "q", "d2m", "2m_dewpoint_temperature"]),
]

# NASA POWER variable specifications (updated with more variables)
NASA_POWER_VARIABLE_SPECS = [
    ("z", ["z", "geopotential"]),  # Not available in NASA POWER, will be zeros
    ("t2m", ["t2m", "T2M"]),
    ("swvl1", ["swvl1"]),  # Not available in NASA POWER
    ("tp", ["tp", "PRECTOT", "PRECTOTCORR"]),
    ("humidity", ["humidity", "RH2M", "d2m", "T2MDEW"]),
    ("wind", ["wind", "WS10M", "WS2M"]),  # NEW: Wind speed
    ("radiation", ["radiation", "ALLSKY_SFC_SW_DWN", "GHI"]),  # NEW: Solar radiation
    ("pressure", ["pressure", "PS", "surface_pressure"]),  # NEW: Surface pressure
]

PRESSURE_DIM_CANDIDATES = ("pressure_level", "isobaricInhPa", "level", "plev")
TIME_DIM_CANDIDATES = ("time", "valid_time")
LAT_DIM_CANDIDATES = ("latitude", "lat")
LON_DIM_CANDIDATES = ("longitude", "lon")

def fill_nan_along_time(field: np.ndarray) -> np.ndarray:
    """
    Fill NaN values in a (Time, H, W) field using linear interpolation along time.
    """
    time_steps = field.shape[0]
    flat = field.reshape(time_steps, -1)
    time_idx = np.arange(time_steps)

    for col in range(flat.shape[1]):
        series = flat[:, col]
        if not np.isnan(series).any():
            continue

        valid = ~np.isnan(series)
        if not np.any(valid):
            flat[:, col] = 0.0
            continue

        if np.sum(valid) == 1:
            flat[:, col] = series[valid][0]
            continue

        interpolated = np.interp(time_idx, time_idx[valid], series[valid])
        flat[:, col] = interpolated

    return flat.reshape(field.shape)

class DataLoader:
    def __init__(self):
        self.lat_min, self.lat_max = settings.LAT_BOUNDS
        self.lon_min, self.lon_max = settings.LON_BOUNDS
        self.data_dir = settings.DATA_DIR

    def _rename_standard_dims(self, ds: xr.Dataset) -> xr.Dataset:
        """Normalize common coordinate and time names."""
        rename_map = {}
        for candidate in TIME_DIM_CANDIDATES:
            if candidate in ds.dims and candidate != "time":
                rename_map[candidate] = "time"
                break
        for candidate in LAT_DIM_CANDIDATES:
            if candidate in ds.dims and candidate != "latitude":
                rename_map[candidate] = "latitude"
                break
        for candidate in LON_DIM_CANDIDATES:
            if candidate in ds.dims and candidate != "longitude":
                rename_map[candidate] = "longitude"
                break
        if rename_map:
            ds = ds.rename(rename_map)
        return ds

    def _resolve_pressure_dim(self, ds: xr.Dataset) -> Optional[str]:
        for dim_name in PRESSURE_DIM_CANDIDATES:
            if dim_name in ds.dims:
                return dim_name
        return None

    def _select_nc_files(self, year: Optional[int] = None) -> List[Path]:
        pattern = f"*{year}*.nc" if year else "*.nc"
        files = list(self.data_dir.glob(pattern))

        def _priority(path: Path) -> Tuple[int, str]:
            name = path.name.lower()
            if "surface" in name:
                return (0, name)
            if "upper" in name:
                return (1, name)
            return (2, name)

        files.sort(key=_priority)
        return files

    def _preprocess_ds(self, ds: xr.Dataset) -> xr.Dataset:
        """Preprocess individual dataset before concatenation."""
        ds = self._rename_standard_dims(ds)
        required_dims = {"latitude", "longitude"}
        if not required_dims.issubset(set(ds.dims)):
            return xr.Dataset()

        # 1. Drop unnecessary variables
        vars_to_keep = []
        for _, candidates in DYNAMIC_VARIABLE_SPECS:
            for candidate in candidates:
                if candidate in ds and candidate not in vars_to_keep:
                    vars_to_keep.append(candidate)
        
        if not vars_to_keep:
            return xr.Dataset()
            
        ds = ds[vars_to_keep]

        # 2. Handle 'expver'
        if "expver" in ds.dims:
            ds = ds.reduce(np.nanmax, dim="expver")
        elif "expver" in ds.coords:
            ds = ds.drop_vars("expver")

        # 3. Standardize time dimension
        if "time" not in ds.dims:
            return xr.Dataset()

        # 4. Handle pressure levels if any
        pressure_dim = self._resolve_pressure_dim(ds)
        if pressure_dim is not None:
            if int(ds.sizes.get(pressure_dim, 0)) > 1:
                ds = ds.isel({pressure_dim: 0})
            if pressure_dim in ds.dims:
                ds = ds.squeeze(pressure_dim, drop=True)

        # 5. Crop Spatially
        lat_slice = slice(self.lat_max, self.lat_min) if ds.latitude[0] > ds.latitude[-1] else slice(self.lat_min, self.lat_max)
        ds = ds.sel(latitude=lat_slice, longitude=slice(self.lon_min, self.lon_max))

        if ds.sizes.get("latitude", 0) == 0 or ds.sizes.get("longitude", 0) == 0:
            return xr.Dataset()

        return ds

    def load_era5(self, year: Optional[int] = None) -> xr.Dataset:
        """Load and merge ERA5 datasets lazily."""
        files = self._select_nc_files(year=year)
        
        if not files:
            raise FileNotFoundError(f"No NetCDF files found in {self.data_dir}")

        logger.info(f"Loading {len(files)} files from {self.data_dir}...")
        
        datasets = []
        skipped = 0
        for f in sorted(files):
            try:
                # Attempt to open with chunks (lazy), fallback to eager if dask fails
                try:
                    ds = xr.open_dataset(f, chunks={"time": 24})
                except (ImportError, Exception):
                    ds = xr.open_dataset(f)
                
                processed = self._preprocess_ds(ds)
                if processed.dims:
                    datasets.append(processed)
                else:
                    skipped += 1
            except Exception as e:
                logger.error(f"Error processing {f.name}: {e}")
                skipped += 1

        if not datasets:
            raise ValueError("No valid data could be processed.")
        if skipped > 0:
            logger.warning(f"Skipped {skipped} file(s) due to incompatible schema or empty crop.")

        # Merge all datasets
        full_ds = xr.concat(
            datasets,
            dim="time",
            data_vars="minimal",
            coords="minimal",
            compat="override",
            join="outer",
        ).sortby("time")
        
        # Remove duplicates
        _, index = np.unique(full_ds.time, return_index=True)
        full_ds = full_ds.isel(time=np.sort(index))
        
        return full_ds

    def load_nasa_power(self, year: Optional[int] = None) -> xr.Dataset:
        """Load NASA POWER datasets lazily."""
        pattern = f"nasa_power*{year}*.nc" if year else "nasa_power*.nc"
        files = list(self.data_dir.glob(pattern))
        
        if not files:
            logger.warning(f"No NASA POWER files found matching {pattern}")
            return xr.Dataset()
        
        logger.info(f"Loading {len(files)} NASA POWER files...")
        
        datasets = []
        for f in sorted(files):
            try:
                # Attempt to open with chunks (lazy), fallback to eager if dask fails
                try:
                    ds = xr.open_dataset(f, chunks={"time": 24})
                except (ImportError, Exception):
                    ds = xr.open_dataset(f)
                
                ds = self._rename_standard_dims(ds)
                
                # Crop spatially
                lat_slice = slice(self.lat_max, self.lat_min) if ds.latitude[0] > ds.latitude[-1] else slice(self.lat_min, self.lat_max)
                ds = ds.sel(latitude=lat_slice, longitude=slice(self.lon_min, self.lon_max))
                
                if ds.sizes.get("latitude", 0) > 0 and ds.sizes.get("longitude", 0) > 0:
                    datasets.append(ds)
            except Exception as e:
                logger.error(f"Error loading {f.name}: {e}")
        
        if not datasets:
            return xr.Dataset()
        
        # Merge by concatenating along time
        full_ds = xr.concat(
            datasets,
            dim="time",
            data_vars="minimal",
            coords="minimal",
            compat="override",
            join="outer",
        ).sortby("time")
        
        # Remove duplicates
        _, index = np.unique(full_ds.time, return_index=True)
        full_ds = full_ds.isel(time=np.sort(index))
        
        return full_ds
    
    def load_combined(self, year: Optional[int] = None) -> xr.Dataset:
        """Load and combine ERA5 + NASA POWER data."""
        # Load ERA5
        era5_ds = self.load_era5(year=year)
        
        # Load NASA POWER
        power_ds = self.load_nasa_power(year=year)
        
        if not power_ds.dims:
            logger.info("No NASA POWER data found, using ERA5 only")
            return era5_ds
        
        if not era5_ds.dims:
            logger.info("No ERA5 data found, using NASA POWER only")
            return power_ds
        
        # Merge ERA5 and NASA POWER
        logger.info("Merging ERA5 and NASA POWER datasets...")
        
        # Find common variables ( ERA5 and NASA POWER both have temperature)
        # Rename NASA POWER variables to match ERA5 naming convention
        rename_map = {}
        
        # Map NASA POWER variables to ERA5 variable names
        # Note: ERA5 already has t2m, so we use different approach
        # IMPORTANT: NASA POWER T2M is in Celsius, ERA5 t2m is in Kelvin
        # We need to convert NASA POWER temperature to Kelvin for consistency
        if 'T2M' in power_ds:
            # NASA POWER T2M is in Celsius, convert to Kelvin
            # ERA5 t2m is in Kelvin (273.15 + Celsius)
            power_ds['t2m_nasa'] = power_ds['T2M'] + 273.15
            power_ds['t2m_nasa'].attrs = power_ds['T2M'].attrs.copy()
            power_ds['t2m_nasa'].attrs['units'] = 'K'
            power_ds['t2m_nasa'].attrs['long_name'] = '2m Temperature (converted from NASA POWER Celsius)'
            logger.info("Converted NASA POWER T2M from Celsius to Kelvin (added 273.15)")
        if 'RH2M' in power_ds:
            rename_map['RH2M'] = 'humidity'
        if 'PRECTOTCORR' in power_ds:
            rename_map['PRECTOTCORR'] = 'tp'
        if 'T2MDEW' in power_ds:
            # T2MDEW is also in Celsius, convert to Kelvin
            power_ds['d2m_nasa'] = power_ds['T2MDEW'] + 273.15
            power_ds['d2m_nasa'].attrs = power_ds['T2MDEW'].attrs.copy()
            power_ds['d2m_nasa'].attrs['units'] = 'K'
            logger.info("Converted NASA POWER T2MDEW from Celsius to Kelvin")
        
        if rename_map:
            power_ds = power_ds.rename(rename_map)
            logger.info(f"Renamed NASA POWER variables: {rename_map}")
        
        # Combine - use outer join to keep all variables
        combined = xr.merge([era5_ds, power_ds], join='outer', compat='override')
        
        logger.info(f"Combined dataset variables: {list(combined.data_vars)}")
        
        return combined
        """Load and merge ERA5 datasets lazily."""
        files = self._select_nc_files(year=year)
        
        if not files:
            raise FileNotFoundError(f"No NetCDF files found in {self.data_dir}")

        logger.info(f"Loading {len(files)} files from {self.data_dir}...")
        
        datasets = []
        skipped = 0
        for f in sorted(files):
            try:
                # Attempt to open with chunks (lazy), fallback to eager if dask fails
                try:
                    ds = xr.open_dataset(f, chunks={"time": 24})
                except (ImportError, Exception):
                    ds = xr.open_dataset(f)
                
                processed = self._preprocess_ds(ds)
                if processed.dims:
                    datasets.append(processed)
                else:
                    skipped += 1
            except Exception as e:
                logger.error(f"Error processing {f.name}: {e}")
                skipped += 1

        if not datasets:
            raise ValueError("No valid data could be processed.")
        if skipped > 0:
            logger.warning(f"Skipped {skipped} file(s) due to incompatible schema or empty crop.")

        # Merge all datasets
        full_ds = xr.concat(
            datasets,
            dim="time",
            data_vars="minimal",
            coords="minimal",
            compat="override",
            join="outer",
        ).sortby("time")
        
        # Remove duplicates
        _, index = np.unique(full_ds.time, return_index=True)
        full_ds = full_ds.isel(time=np.sort(index))
        
        return full_ds

    def prepare_training_data(self, ds: xr.Dataset, fill_nan: bool = True, 
                               variable_specs: Optional[List[Tuple[str, List[str]]]] = None) -> Tuple[np.ndarray, Dict]:
        """Convert xarray dataset to raw numpy array for training (NOT normalized).
        
        Normalization is deferred to compute_train_normalization_stats() to avoid
        data leakage from test/val splits during training.
        
        Args:
            ds: xarray Dataset with ERA5 data
            fill_nan: If False, skip NaN filling (do it after temporal split to prevent leakage)
            variable_specs: Override variable specifications (default: ERA5 specs)
        
        Returns:
            data_array: Numpy array of shape (Time, Channels, H, W)
            stats: Dictionary with metadata including time_index for proper splitting
        """
        logger.info("Preparing training data (dynamic + static channels)...")
        
        # Auto-detect variable specs based on available variables
        if variable_specs is None:
            # Check if this is NASA POWER data (has T2M, RH2M, etc.)
            if 'T2M' in ds or 'RH2M' in ds or 'PRECTOTCORR' in ds:
                variable_specs = NASA_POWER_VARIABLE_SPECS
                logger.info("Detected NASA POWER data format")
            else:
                variable_specs = DYNAMIC_VARIABLE_SPECS
        
        # 1. Extract dynamic variables
        channels = []
        reference = None
        dynamic_available = []
        dynamic_missing = []
        dynamic_sources: Dict[str, Optional[str]] = {}
        for name, candidates in variable_specs:
            source = next((c for c in candidates if c in ds), None)
            if source:
                data = ds[source].values
                if fill_nan:
                    data = fill_nan_along_time(data)
                channels.append(data)
                dynamic_available.append(name)
                dynamic_sources[name] = source
                if reference is None:
                    reference = data
            else:
                logger.warning(f"Variable {name} not found, filling with zeros.")
                dynamic_missing.append(name)
                dynamic_sources[name] = None
                if reference is None:
                    ref_source = next((k for k in ds.data_vars if "time" in ds[k].dims), None)
                    if ref_source is None:
                        raise ValueError("No time-dependent variables available for channel fallback.")
                    reference = ds[ref_source].values
                channels.append(np.zeros_like(reference))

        # 2. Add static channels (Elevation, Lat, Lon)
        lats, lons = ds.latitude.values, ds.longitude.values
        lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
        
        # Use 'z' (geopotential) mean as elevation proxy
        z_source = next((c for c in DYNAMIC_VARIABLE_SPECS[0][1] if c in ds), None)
        if z_source is None:
            logger.warning("No geopotential channel found; using zero elevation proxy.")
            elev = np.zeros_like(lat_grid, dtype=np.float32)
        else:
            elev = ds[z_source].mean(dim='time').values

        t_len = len(ds.time)
        channels.append(np.repeat(elev[np.newaxis, ...], t_len, axis=0))
        channels.append(np.repeat(lat_grid[np.newaxis, ...], t_len, axis=0))
        channels.append(np.repeat(lon_grid[np.newaxis, ...], t_len, axis=0))

        # Stack to (Time, Channels, H, W)
        data_array = np.stack(channels, axis=1).astype(np.float32)
        np.nan_to_num(data_array, copy=False)
        
        # 3. Return raw (un-normalized) data with metadata
        stats = {
            "mean": None,
            "std": None,
            "lats": lats,
            "lons": lons,
            "time_index": ds.time.values.astype("datetime64[ns]").astype(str).tolist(),
            "dynamic_available": dynamic_available,
            "dynamic_missing": dynamic_missing,
            "dynamic_sources": dynamic_sources,
        }
        
        return data_array, stats


    def compute_train_normalization_stats(self, data: np.ndarray, train_end_idx: int) -> dict:
        """
        Compute normalization statistics from training data only.
        
        Args:
            data: Raw data array of shape (Time, Channels, H, W)
            train_end_idx: Index (exclusive) marking the end of training data
            
        Returns:
            dict with 'mean' and 'std' arrays of shape (1, Channels, 1, 1)
        """
        train_data = data[:train_end_idx]
        mean = train_data.mean(axis=(0, 2, 3), keepdims=True)
        std = train_data.std(axis=(0, 2, 3), keepdims=True)
        std = np.where(std < 1e-8, 1e-8, std)  # prevent division by zero
        return {'mean': mean, 'std': std}

def create_sequences(data: np.ndarray, seq_len: int = 10, pred_len: int = 5):
    """Create input/target sequences using sliding window."""
    total_window = seq_len + pred_len
    num_samples = len(data) - total_window + 1
    
    if num_samples <= 0:
        return np.array([]), np.array([])

    windows = np.lib.stride_tricks.sliding_window_view(data, window_shape=total_window, axis=0)
    windows = np.moveaxis(windows, -1, 1)
    
    X = windows[:, :seq_len, ...].copy()
    Y = windows[:, seq_len:, ...].copy()
    
    return X, Y


# ---------------------------------------------------------------------------
# Backward-compatible standalone functions
# These match the signatures from the legacy root data_loader.py so that
# api_server.py and other callers can migrate without breakage.
# ---------------------------------------------------------------------------


def compute_normalization_stats(data: np.ndarray):
    """Compute per-channel mean/std from data of shape (Time, Channels, H, W)."""
    mean = data.mean(axis=(0, 2, 3), keepdims=True)
    std = data.std(axis=(0, 2, 3), keepdims=True)
    return mean, std


def normalize_data(data: np.ndarray, mean: np.ndarray, std: np.ndarray):
    """Normalize data of shape (Time, Channels, H, W) with provided stats."""
    return (data - mean) / (std + EPSILON)


def clean_data(data, clip_percentiles=(0.5, 99.5), clip_bounds=None):
    """
    Clean data of shape (Time, Channels, H, W) before normalization/training.
    Steps: Replace inf with NaN, fill NaNs, clip outliers.
    """
    cleaned = np.array(data, dtype=np.float32, copy=True)
    cleaned[~np.isfinite(cleaned)] = np.nan

    for channel_idx in range(cleaned.shape[1]):
        cleaned[:, channel_idx, :, :] = fill_nan_along_time(
            cleaned[:, channel_idx, :, :]
        )

    if clip_bounds is None:
        low_q, high_q = clip_percentiles
        lower = np.percentile(cleaned, low_q, axis=(0, 2, 3), keepdims=True)
        upper = np.percentile(cleaned, high_q, axis=(0, 2, 3), keepdims=True)
    else:
        lower, upper = clip_bounds

    cleaned = np.clip(cleaned, lower, upper)
    np.nan_to_num(cleaned, copy=False)
    return cleaned, (lower, upper)


def load_era5_data(data_dir, year=None, normalize=True, stats=None):
    """
    Backward-compatible wrapper around the DataLoader class.

    Returns:
        data_out, lats, lons, mean, std
    """
    loader = DataLoader()
    loader.data_dir = Path(data_dir)

    ds = loader.load_era5(year=year)
    data_array, data_stats = loader.prepare_training_data(ds)

    lats = data_stats["lats"]
    lons = data_stats["lons"]
    
    # Since prepare_training_data() now returns raw data (no leakage),
    # compute normalization stats on the full returned array for backward compat
    if normalize:
        mean, std = compute_normalization_stats(data_array)
        data_out = normalize_data(data_array, mean, std)
    else:
        # Return zeros for mean/std when not normalizing
        data_out = data_array
        mean = np.zeros((1, data_array.shape[1], 1, 1), dtype=np.float32)
        std = np.zeros((1, data_array.shape[1], 1, 1), dtype=np.float32)
    
    # Allow override via stats parameter
    if stats is not None:
        mean, std = stats
        data_out = normalize_data(data_array, mean, std)

    return data_out, lats, lons, mean, std
