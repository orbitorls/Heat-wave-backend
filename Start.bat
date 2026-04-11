@echo off
setlocal
cd /d "%~dp0"

echo ============================================
echo   HEATWAVE-AI Launcher (XGBoost Model)
echo ============================================
echo.
echo   [1] Train XGBoost Model (RECOMMENDED)
echo   [2] Train ConvLSTM/RF Model (Legacy)
echo   [3] Run Dashboard (API + Frontend)
echo   [4] Health Check
echo   [0] Exit
echo.
set /p CHOICE=Select option: 

if "%CHOICE%"=="1" (
  echo.
  echo ========================================
  echo   Training XGBoost Daily Model...
  echo   Uses single-day weather features
  echo ========================================
  python train_daily_xgboost.py
  echo.
  echo Training complete! Check output/ for report.
  pause
  goto :eof
)
if "%CHOICE%"=="2" (
  python Train_Ai.py
  goto :eof
)
if "%CHOICE%"=="3" (
  echo.
  echo Starting Dashboard (XGBoost model)...
  start "Agni API" cmd /k "cd /d D:\Heat-wave-backend && python api_server.py"
  timeout /t 2 /nobreak >nul
  start "Agni Frontend" cmd /k "cd /d D:\Heat-wave-backend\agni-web && npm run dev"
  timeout /t 3 /nobreak >nul
  start http://localhost:5173
  goto :eof
)
if "%CHOICE%"=="4" (
  echo Checking API health...
  curl http://localhost:5000/api/health
  echo.
  curl http://localhost:5000/api/daily/health
  pause
  goto :eof
)
if "%CHOICE%"=="0" goto :eof

echo Invalid choice.
pause