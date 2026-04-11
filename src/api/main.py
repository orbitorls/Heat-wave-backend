from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse
from pydantic import BaseModel
import asyncio
import json
from typing import Dict, List, Optional, Any

from src.core.config import settings
from src.core.logger import logger
from src.core.utils import detect_gpu_capability
from src.models.manager import model_manager
from src.api.services.training import training_service

app = FastAPI(title="Thailand Heatwave Prediction API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    model_manager.load_model()

@app.get("/")
async def root():
    return RedirectResponse(url="/trainer")

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "gpu": detect_gpu_capability(),
        "model_loaded": model_manager.model is not None,
        "model_type": model_manager.model_type,
    }

# --- Training Endpoints ---

@app.get("/api/training/status")
async def get_training_status():
    return training_service.get_status()

@app.post("/api/training/start")
async def start_training(config: Dict):
    try:
        training_service.start_training(config)
        return {"message": "Training started."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/training/stop")
async def stop_training():
    training_service.stop_training()
    return {"message": "Stop command sent."}

# WebSocket for live updates
@app.websocket("/ws/training")
async def websocket_training(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Send status every 1 second
            status = training_service.get_status()
            await websocket.send_json(status)
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")

# --- UI Route ---

@app.get("/trainer", response_class=HTMLResponse)
async def trainer_ui():
    with open("src/api/templates/trainer.html", "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)
