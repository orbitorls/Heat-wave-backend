# Heatwave Prediction - Dataset Download Instructions

## Quick Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Weather Data

The dataset is ~260MB and needs to be downloaded separately:

```bash
# Download NASA POWER data (1981-2015) - ~35 years of daily data
python download_nasa_power.py

# Optional: Download ERA5 data (requires CDS API credentials)
# python download_era5.py
```

### 3. Train the Model

```bash
# Train XGBoost model (recommended)
python train_daily_xgboost.py

# Or train ConvLSTM model
python Train_Ai.py
```

### 4. Run the API
```bash
# Start API server
run_agni.bat
# Or
python api_server.py
```

## Dataset Details

| Dataset | Years | Size | Variables |
|---------|-------|------|-----------|
| NASAPOWER | 1981-2015 | ~5MB/year | T2M, RH2M, PRECTOT, WS10M, PS |
| ERA5 | 2000-2015 | ~3MB/year | T2M, Z, SWVL1, TP |

### Variables Used
- **T2M**: Temperature at 2 meters (Kelvin → Celsius)
- **RH2M**: Relative humidity (%)
- **PRECTOT**: Precipitation (mm)
- **WS10M**: Wind speed at 10m (m/s)
- **PS**: Surface pressure (hPa)
- **Z**: Geopotential height
- **SWVL1**: Soil moisture layer 1

## License

Datasatasets:
- NASA POWER: Public domain
- ERA5: Copernicus license (free for research)