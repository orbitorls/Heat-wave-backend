@echo off
setlocal
cd /d "%~dp0"

echo ============================================
echo   HEATWAVE-AI Launcher
echo ============================================
echo.
echo   [1] Launch GUI (PyQt6 Desktop Application)
echo   [2] Launch TUI (Text Interface)
echo   [3] Train XGBoost Model
echo   [4] Train ConvLSTM/RF Model
echo   [5] Check Model Accuracy
echo   [6] Download ERA5 Data
echo   [0] Exit
echo.
set /p CHOICE=Select option: 

if "%CHOICE%"=="1" (
  echo.
  echo ========================================
  echo   Starting PyQt6 GUI...
  echo   Full-featured desktop interface
  echo ========================================
  python -m src.gui
  goto :eof
)
if "%CHOICE%"=="2" (
  echo.
  echo ========================================
  echo   Starting Textual TUI...
  echo   Full-featured terminal interface
  echo ========================================
  python -m src.tui.app
  pause
  goto :eof
)
if "%CHOICE%"=="3" (
  echo.
  echo ========================================
  echo   Training XGBoost Daily Model...
  echo   Uses single-day weather features
  echo ========================================
  python scripts/training/train_daily_xgboost.py
  echo.
  echo Training complete! Check output/ for report.
  pause
  goto :eof
)
if "%CHOICE%"=="4" (
  echo.
  echo ========================================
  echo   Training ConvLSTM/RF Model...
  echo   Sequence-based spatial forecasting
  echo ========================================
  python scripts/training/Train_Ai.py
  pause
  goto :eof
)
if "%CHOICE%"=="5" (
  echo.
  echo ========================================
  echo   Checking Model Accuracy...
  echo ========================================
  python scripts/eval/check_model_accuracy.py
  pause
  goto :eof
)
if "%CHOICE%"=="6" (
  echo.
  echo ========================================
  echo   Downloading ERA5 Data...
  echo ========================================
  python scripts/data/download_era5.py
  pause
  goto :eof
)
if "%CHOICE%"=="0" goto :eof

echo Invalid choice.
pause