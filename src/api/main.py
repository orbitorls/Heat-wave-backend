"""FastAPI backend for Heatwave Prediction Web App"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.manager import ModelManager
from src.core.logger import logger

# Import routes
from .routes import models, predict, train, eval as eval_routes, data, map as map_routes, system

# Global model manager instance
model_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager - load ModelManager on startup"""
    global model_manager
    logger.info("Starting FastAPI backend...")
    try:
        model_manager = ModelManager()
        logger.info("ModelManager initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize ModelManager: {e}")
        model_manager = None
    yield
    logger.info("Shutting down FastAPI backend...")


app = FastAPI(
    title="Heatwave Prediction API",
    description="Backend API for Thailand Heatwave Prediction System",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Heatwave Prediction API", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_manager_loaded": model_manager is not None
    }


# Register routes
app.include_router(models.router, prefix="/api/models", tags=["models"])
app.include_router(predict.router, prefix="/api/predict", tags=["predict"])
app.include_router(train.router, prefix="/api/train", tags=["train"])
app.include_router(eval_routes.router, prefix="/api/eval", tags=["eval"])
app.include_router(data.router, prefix="/api/data", tags=["data"])
app.include_router(map_routes.router, prefix="/api/map", tags=["map"])
app.include_router(system.router, prefix="/api/system", tags=["system"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
