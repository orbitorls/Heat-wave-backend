# AGENTS.md - Heat-wave-backend Project Guide

> Compact instruction file for future OpenCode sessions working on this Thai heatwave prediction system.

## Project Overview

Thailand Heatwave Forecasting Backend using Deep Learning (ConvLSTM) and ML (XGBoost/RandomForest) models with ERA5 meteorological data.

### Key Facts
- **Models**: 
  - ConvLSTM for sequence-based spatial forecasting
  - XGBoost/RandomForest for daily classification
- **Data**: ERA5 + NASA POWER meteorological data
- **Task**: Binary classification (heatwave: yes/no) + temperature prediction
- **Threshold**: 35-38°C depending on model version

## Development Commands

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Key packages: torch, xgboost, scikit-learn, xarray, netCDF4, pandas, numpy
```

### Data Preparation
```bash
# Download ERA5 data (requires CDS API credentials)
python download_era5.py

# Or use NASA POWER data
python download_nasa_power.py
```

### Training Models

#### XGBoost Daily Model (Recommended for quick training)
```bash
python train_daily_xgboost.py
```
- Faster than ConvLSTM (~minutes vs hours)
- Uses daily features instead of sequences
- Outputs: `models/heatwave_daily_xgboost_v{N}.pth`
- Generates: `output/xgboost_daily_report_v{N}.png`

#### ConvLSTM Model (For sequence-based forecasting)
```bash
python Train_Ai.py
```
- Uses 7-day sequences to predict 2-day horizon
- Outputs: `models/heatwave_model_checkpoint_v{N}.pth`
- Generates: `output/training_report_*.png`

### Evaluation
```bash
# Run tests
pytest

# Check model accuracy
python check_model_accuracy.py
```

### Running the TUI (Text User Interface)
```bash
# Launch TUI (recommended - full-featured terminal interface)
python -m src.tui.app

# Or via main.py
python main.py --mode tui

# Or via Start.bat (option 1)
Start.bat
```

### Running the API (Deprecated)
```bash
# Flask API server (production - deprecated, use TUI instead)
python _deprecated_api/api_server.py.deprecated

# FastAPI (development - deprecated)
python _deprecated_api/src_api/main.py
```

## Model Accuracy Check

**Current Status**: XGBoost v6 model exists but couldn't load metrics due to missing xgboost in environment.

**To retrain if needed**:
```bash
pip install xgboost scikit-learn imbalanced-learn
python train_daily_xgboost.py
```

**Target Metrics**:
- F1 Score >= 0.70 (Good)
- F1 Score 0.50-0.70 (Moderate)
- F1 Score < 0.50 (Needs Improvement)

**If accuracy is low**, adjust:
1. `HEATWAVE_TEMP_THRESHOLD` (default: 38.0C for XGBoost, 35.0C for ConvLSTM)
2. `MIN_TRAIN_POSITIVE_RATE` (target 8-40% positive samples)
3. `RF_N_ESTIMATORS`, `RF_MAX_DEPTH` (hyperparameters)
4. Feature engineering in `add_engineered_features()`

## Architecture Notes

### TUI Features

The Textual TUI provides a full-featured terminal interface with the following screens:

- **Dashboard**: System health check, GPU status, model loading status, quick action buttons
- **Predict**: Load model, select input data, run inference, view results
- **Map**: Thailand heatmap visualization using Unicode block characters with color gradients
- **Train**: Configure hyperparameters, start/stop training, view live logs
- **Data**: Download ERA5 data, audit files, organize data directory
- **Eval**: Run model evaluation, check accuracy, view metrics (F1, Precision, Recall, RMSE)
- **Checkpoints**: List available model checkpoints, load/delete models
- **Logs**: View training logs, system logs, API logs with real-time updates

**TUI Navigation**:
- Keyboard shortcuts: `d` (Dashboard), `p` (Predict), `m` (Map), `t` (Train), `a` (Data), `e` (Eval), `c` (Checkpoints), `l` (Logs), `q` (Quit)
- Or use button navigation with mouse/touch
- Arrow keys for scrolling and map navigation

### Directory Structure
```
src/
├── tui/           # Text User Interface (Textual-based)
│   ├── app.py     # Main TUI application
│   ├── screens/   # Dashboard, Predict, Map, Train, Data, Eval, Checkpoints, Logs
│   ├── widgets/   # MapCanvas, MetricPanel, LogPanel
│   └── map_renderer.py # Terminal heatmap rendering
├── cli/           # Command-line interface
├── core/          # Config, logger, utils
├── data/loader.py # ERA5 data loading & preprocessing
└── models/
    ├── convlstm.py    # HeatwaveConvLSTM + PhysicsInformedLoss
    └── manager.py     # ModelManager (load/predict)
models/            # Model checkpoints (.pth)
output/            # Training reports
era5_data/         # ERA5 NetCDF inputs (download separately)
_deprecated_api/   # Legacy API/Web interface (moved here)
```

### Model Configuration

**XGBoost Daily** (`train_daily_xgboost.py`):
```python
HEATWAVE_TEMP_THRESHOLD = 38.0  # Celsius (Thai summer)
HEATWAVE_MIN_DURATION = 3  # Consecutive days
TRAIN_RATIO = 0.75
VAL_RATIO = 0.10
TEST_RATIO = 0.15
```

**ConvLSTM** (`Train_Ai.py`):
```python
SEQ_LEN = 5  # Input sequence length
FUTURE_SEQ = 2  # Prediction horizon
HEATWAVE_TEMPERATURE_THRESHOLD = 35.0  # Celsius
HEATWAVE_ANOMALY_THRESHOLD = 6.0  # Above climatological mean
```

### Key Classes

1. **HeatwaveConvLSTM** (`src/models/convlstm.py`): Multi-layer ConvLSTM for sequence-to-sequence forecasting

2. **PhysicsInformedLoss**: MSE + spatial gradient regularization

3. **DataLoader** (`src/data/loader.py`): Loads ERA5 NetCDF files with coordinate encoding

### Data Encoding

**Input Channels (6)**:
- Channel 0: Geopotential Height (Z500)
- Channel 1: Temperature at 2m (T2m)
- Channel 2: Soil Moisture (SWVL1)
- Channel 3: Elevation (static)
- Channel 4: Latitude (normalized)
- Channel 5: Longitude (normalized)

## Common Issues

### 1. Missing Dependencies
```bash
# All required packages
pip install torch xgboost scikit-learn imbalanced-learn xarray netCDF4 pandas numpy matplotlib seaborn
```

### 2. Model Load Errors
```
ModuleNotFoundError: No module named 'xgboost'
```
**Fix**: Install xgboost before loading .pth files
```bash
pip install xgboost
```

### 3. Data Not Found
```
FileNotFoundError: era5_data/
```
**Fix**: Download ERA5 data
```bash
python download_era5.py  # Requires CDS API key in ~/.cdsapirc
python download_nasa_power.py  # Alternative (no auth needed)
```

### 4. Low Accuracy
If F1 < 0.50:
1. Check positive rate (aim for 5-30%)
2. Lower temperature threshold
3. Try anomaly-based labeling instead of absolute threshold
4. Enable feature engineering: `FEATURE_ENGINEERING_ENABLED=True`
5. Try walk-forward validation: `WALK_FORWARD_ENABLED=True`

## Evaluation Metrics

**Classification Metrics**:
- Precision: TP / (TP + FP) - How many predicted heatwaves were real
- Recall: TP / (TP + FN) - How many real heatwaves were caught
- F1: Harmonic mean of Precision and Recall
- Brier Score: Probability calibration
- PR-AUC: Area under Precision-Recall curve

**Regression Metrics** (ConvLSTM):
- MAE (°C): Mean Absolute Error
- RMSE (°C): Root Mean Square Error
- R²: Coefficient of determination

**Baseline Comparisons**:
- Persistence: Tomorrow = Today's weather
- Climatology: Mean historical temperature

## Testing

```bash
# Unit tests
pytest tests/

# Specific test
pytest tests/test_model.py -k "PhysicsInformedLoss" -v
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/predict` | POST | Forecast temperature |
| `/api/map` | GET | Get GeoJSON for map visualization |
| `/api/health` | GET | System health check |

## Environment Variables

Override defaults via environment variables (see `Train_Ai.py`):
```bash
export HW_BATCH_SIZE=4
export HW_SEQ_LEN=5
export HW_FUTURE_SEQ=2
export HW_HEATWAVE_TEMPERATURE_THRESHOLD=35.0
export HW_TRAIN_RATIO=0.75
export HW_USE_XGBOOST=True
export HW_USE_LIGHTGBM=True
```

## Output Files

| File | Description |
|------|-------------|
| `models/heatwave_daily_xgboost_v{N}.pth` | XGBoost checkpoint |
| `models/heatwave_model_checkpoint_v{N}.pth` | ConvLSTM checkpoint |
| `output/xgboost_daily_report_v{N}.png` | Training visualization |
| `output/training_report_*.png` | ConvLSTM training report |
| `output/forecast_sample_*.png` | Prediction visualization |

## Important Notes

1. **Data Leakage Prevention**: Temporal split (train < val < test), no shuffling
2. **Class Imbalance**: Uses BalancedRandomForest or scale_pos_weight
3. **NaN Handling**: `fill_nan_along_time()` interpolates missing values
4. **Coordinate Encoding**: Lat/Lon/Elevation added as input channels for spatial awareness
5. **Physics-Informed**: Spatial gradient loss prevents unrealistic temperature jumps

## Quick Start for New Session

```bash
# 1. Check accuracy of existing models
pip install torch xgboost scikit-learn
python check_model_accuracy.py

# 2. Retrain if needed
python train_daily_xgboost.py

# 3. Run API
python api_server.py
```

## Retraining Decision Tree

1. Load existing model → Check F1 score
2. If F1 < 0.50:
   - Lower threshold (38→36→35°C)
   - Enable feature engineering
   - Increase positive samples
3. If F1 0.50-0.70:
   - Hyperparameter tuning
   - Try different algorithms
4. If F1 >= 0.70:
   - Model is good, no retrain needed

## Related Commands

```bash
# Check GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Inspect data shape
python -c "from src.data.loader import DataLoader; ds = DataLoader().load_combined(); print(ds.shape)"

# View model architecture
python -c "from src.models.convlstm import HeatwaveConvLSTM; print(HeatwaveConvLSTM(6, [64,64], [(3,3),(3,3)], 2))"
```

---

**Last Updated**: 2026-04-11
**Status**: Active development, models trained but need dependency check for evaluation