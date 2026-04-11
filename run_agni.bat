@echo off
echo ==========================================
echo   Agni - Thailand Heatwave Forecast
echo   Using XGBoost Daily Model
echo ==========================================
echo.

echo [1/2] Starting API Server (XGBoost)...
start "Agni API" cmd /k "cd /d D:\Heat-wave-backend && python api_server.py"

echo [2/2] Starting Frontend...
start "Agni Frontend" cmd /k "cd /d D:\Heat-wave-backend\agni-web && npm run dev"

echo.
echo ==========================================
echo   API Endpoints:
echo.
echo   Main (XGBoost):
echo     GET  /api/predict_xgb        - Dashboard format
echo     POST /api/daily/predict      - JSON input
echo     GET  /api/daily/model_info   - Model info
echo.
echo   Legacy (ConvLSTM):
echo     GET  /api/predict            - Old prediction
echo     GET  /api/forecast          - 7-day forecast
echo.
echo   Frontend: http://localhost:5173
echo   API Health: http://localhost:5000/api/health
echo ==========================================

timeout /t 3 /nobreak >nul
start http://localhost:5173