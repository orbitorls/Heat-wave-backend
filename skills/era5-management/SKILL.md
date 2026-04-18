# SKILL: ERA5 Data Management for Thailand

**Description:** Procedures for fetching, cropping, and pre-processing ERA5 NetCDF files for Thailand-specific meteorological analysis.

## 📝 Procedural Guidance

### Data Fetching
- Use the **CDS API** via `download_era5.py`.
- **Target Coordinates:** `[21, 97, 5, 106]` (North, West, South, East).
- **Target Years:** 2000 to present.
- **Surface Variables:** `2m_temperature`, `soil_temperature_level_1`, `total_precipitation`, `surface_pressure`, `relative_humidity`.
- **Upper-Air Variables:** `geopotential` (at 500 hPa level).

### Data Pre-processing
- **Spatial Cropping:** Use `xarray.Dataset.sel` to crop exactly to Thailand's boundaries (to save RAM).
- **Temporal Alignment:** Match hourly data to daily maximums (Tmax) if doing heatwave analysis.
- **Normalization:** Standardize weather variables (z, t2m, swvl1) while keeping spatial features (Lat, Lon, Elevation) consistent.

### Error Handling
- **HTTP 500 (CDS API):** This is common when the queue is full. Use an exponential backoff retry strategy.
- **Memory Management:** Use `chunks={}` in `xr.open_dataset` to enable Dask (Lazy loading) for large years.
- **Coordinate Mismatch:** Always verify that the grid resolution matches across different years (ERA5 usually 0.25°).

## 🛠️ Resources
- **Reference Code:** `download_era5.py`, `data_loader.py`.
- **Library:** `xarray`, `dask`, `netCDF4`.
