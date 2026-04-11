"""
API routes for daily XGBoost heatwave prediction.

This provides endpoints for predicting heatwaves from single-day weather data
without requiring historical sequences like the ConvLSTM model.
"""

import os
import glob
import logging
from typing import Optional, Dict, Any
from datetime import datetime

import numpy as np
import torch
from flask import Blueprint, request, jsonify

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from src.data.loader import DataLoader, fill_nan_along_time
from pathlib import Path

# Blueprint for daily prediction routes
daily_bp = Blueprint('daily_prediction', __name__)
LOGGER = logging.getLogger(__name__)

# Global model and normalization stats
_daily_model = None
_daily_feature_names = None
_daily_temp_mean = None
_daily_temp_std = None
_daily_threshold = None
_daily_normalization_mean = None
_daily_normalization_std = None


def get_latest_xgboost_model(models_dir: str = "models"):
    """Find the latest XGBoost daily model checkpoint."""
    if not os.path.exists(models_dir):
        return None
    
    files = glob.glob(os.path.join(models_dir, "heatwave_daily_xgboost_v*.pth"))
    if not files:
        return None
    
    def get_version(f):
        try:
            return int(f.split("_v")[-1].split(".")[0])
        except (ValueError, IndexError):
            return 0
    
    return max(files, key=get_version)


def load_daily_model(checkpoint_path: Optional[str] = None):
    """Load the XGBoost daily prediction model."""
    global _daily_model, _daily_feature_names, _daily_temp_mean, _daily_temp_std
    global _daily_threshold, _daily_normalization_mean, _daily_normalization_std
    
    if checkpoint_path is None:
        checkpoint_path = get_latest_xgboost_model()
    
    if checkpoint_path is None:
        LOGGER.warning("No XGBoost daily model found")
        return False
    
    LOGGER.info(f"Loading daily XGBoost model from {checkpoint_path}")
    
    try:
        # PyTorch 2.6+ needs weights_only=False for XGBoost models
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except Exception as e:
        LOGGER.error(f"Failed to load checkpoint: {e}")
        return False
    
    if "sklearn_model" not in checkpoint:
        LOGGER.error("Checkpoint missing sklearn_model")
        return False
    
    _daily_model = checkpoint["sklearn_model"]
    _daily_feature_names = checkpoint.get("feature_names", [])
    
    metadata = checkpoint.get("metadata", {})
    _daily_threshold = metadata.get("heatwave_temp_threshold", 28.43)
    
    # Temperature stats for inference
    _daily_temp_mean = metadata.get("temp_mean_celsius", 26.8)
    _daily_temp_std = metadata.get("temp_std", 5.0)
    
    # Normalization stats
    norm_mean = metadata.get("normalization_mean")
    norm_std = metadata.get("normalization_std")
    
    if norm_mean is not None and norm_std is not None:
        _daily_normalization_mean = np.array(norm_mean)
        _daily_normalization_std = np.array(norm_std)
    
    LOGGER.info(f"Daily model loaded: threshold={_daily_threshold}°C")
    LOGGER.info(f"Features: {_daily_feature_names}")
    
    return True


def daily_model_ready():
    """Check if daily model is loaded."""
    return _daily_model is not None


def extract_features_from_weather(weather_data: Dict[str, Any]) -> np.ndarray:
    """
    Extract features from daily weather data.
    
    Args:
        weather_data: Dict with keys like 'temp_mean', 'temp_max', 'humidity', etc.
        
    Returns:
        Feature array of shape (1, n_features)
    """
    # Default feature order from training
    feature_names = [
        "temp_mean", "temp_max", "temp_min", "temp_std", "temp_range",
        "hot_fraction", "z_mean", "z_std", "swvl1_mean", "tp_mean", "humidity_mean"
    ]
    
    features = []
    
    for name in feature_names:
        value = weather_data.get(name, 0.0)
        if value is None:
            value = 0.0
        features.append(float(value))
    
    return np.array(features, dtype=np.float32).reshape(1, -1)


def predict_from_daily_weather(weather_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict heatwave occurrence from daily weather data.
    
    Args:
        weather_data: Dict with daily weather features
            - temp_mean: Mean temperature in Celsius
            - temp_max: Max temperature in Celsius
            - temp_min: Min temperature in Celsius
            - temp_std: Temperature standard deviation
            - humidity: Relative humidity percentage
            - pressure: Surface pressure (optional)
            - etc.
            
    Returns:
        Dict with:
            - heatwave_probability: float (0-1)
            - heatwave_predicted: bool
            - risk_level: str (LOW/MEDIUM/HIGH/CRITICAL)
    """
    if not daily_model_ready():
        load_daily_model()
    
    if not daily_model_ready():
        return {
            "error": "Daily prediction model not loaded",
            "heatwave_probability": None,
            "heatwave_predicted": False,
            "risk_level": "UNKNOWN"
        }
    
    # Extract features
    features = extract_features_from_weather(weather_data)
    
    # Predict
    probability = _daily_model.predict_proba(features)[0, 1]
    prediction = probability >= 0.5
    
    # Map to risk level
    if probability >= 0.8:
        risk_level = "CRITICAL"
    elif probability >= 0.6:
        risk_level = "HIGH"
    elif probability >= 0.4:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"
    
    return {
        "heatwave_probability": float(probability),
        "heatwave_predicted": bool(prediction),
        "risk_level": risk_level,
        "threshold_used": _daily_threshold,
        "model_type": "xgboost_daily"
    }


# API Routes

@daily_bp.route('/api/daily/predict', methods=['POST'])
def daily_predict():
    """
    Predict heatwave from daily weather data.
    
    Request body:
    {
        "temp_mean": 28.5,      // Mean temperature (Celsius)
        "temp_max": 35.2,       // Max temperature (Celsius)
        "temp_min": 22.1,       // Min temperature (Celsius)
        "temp_std": 3.5,        // Temperature std dev
        "humidity": 75.0,       // Relative humidity (%)
        "pressure": 1013.25,    // Surface pressure (hPa)
        "wind_speed": 5.2,      // Wind speed (m/s)
        "precipitation": 0.0,   // Precipitation (mm)
        "date": "2024-01-15"    // Optional date
    }
    
    Response:
    {
        "heatwave_probability": 0.75,
        "heatwave_predicted": true,
        "risk_level": "HIGH",
        "threshold_used": 28.43
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Validate required fields
        required_fields = ["temp_mean"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Compute derived features if not provided
        weather_data = {}
        
        # Temperature features
        weather_data["temp_mean"] = float(data.get("temp_mean", 0))
        weather_data["temp_max"] = float(data.get("temp_max", data.get("temp_mean", 0) + 5))
        weather_data["temp_min"] = float(data.get("temp_min", data.get("temp_mean", 0) - 5))
        
        if "temp_std" in data:
            weather_data["temp_std"] = float(data["temp_std"])
        else:
            # Estimate std from range
            temp_range = weather_data["temp_max"] - weather_data["temp_min"]
            weather_data["temp_std"] = temp_range / 4.0  # Approximate std
        
        weather_data["temp_range"] = weather_data["temp_max"] - weather_data["temp_min"]
        
        # Hot fraction: what fraction of grid is above threshold
        threshold = 35.0 if weather_data["temp_max"] > 35 else 30.0
        weather_data["hot_fraction"] = max(0, (weather_data["temp_max"] - threshold) / 10.0)
        
        # Other features
        weather_data["z_mean"] = float(data.get("pressure", 1013.25) * 10)  # Approximate geopotential
        weather_data["z_std"] = float(data.get("z_std", 100))
        weather_data["swvl1_mean"] = float(data.get("soil_moisture", 0.3))
        weather_data["tp_mean"] = float(data.get("precipitation", 0))
        weather_data["humidity_mean"] = float(data.get("humidity", 70))
        
        # Make prediction
        result = predict_from_daily_weather(weather_data)
        
        # Add metadata
        result["input_features"] = weather_data
        result["timestamp"] = datetime.now().isoformat()
        
        return jsonify(result)
        
    except Exception as e:
        LOGGER.exception(f"Error in daily_predict: {e}")
        return jsonify({"error": str(e)}), 500


@daily_bp.route('/api/daily/model_info', methods=['GET'])
def daily_model_info():
    """Get information about the loaded daily prediction model."""
    if not daily_model_ready():
        if not load_daily_model():
            return jsonify({"error": "No daily model loaded"}), 404
    
    checkpoint_path = get_latest_xgboost_model()
    
    return jsonify({
        "model_type": "xgboost_daily",
        "model_path": checkpoint_path,
        "threshold_celsius": _daily_threshold,
        "feature_names": _daily_feature_names,
        "temp_mean_celsius": _daily_temp_mean,
        "temp_std": _daily_temp_std,
    })


@daily_bp.route('/api/daily/health', methods=['GET'])
def daily_health():
    """Health check for daily prediction endpoint."""
    model_loaded = daily_model_ready()
    
    return jsonify({
        "status": "ready" if model_loaded else "not_loaded",
        "model_available": os.path.exists(get_latest_xgboost_model() or ""),
        "xgboost_available": XGBOOST_AVAILABLE
    })


@daily_bp.route('/api/predict_xgb', methods=['GET', 'POST'])
def predict_xgb_dashboard():
    """
    Dashboard-compatible endpoint using XGBoost model.
    
    Returns prediction in the same format as /api/predict but uses XGBoost.
    Uses the latest test data from the dataset + XGBoost prediction.
    """
    if request.method == 'POST':
        data = request.get_json() or {}
    else:
        data = {}
    
    # Load model if not ready
    if not daily_model_ready():
        if not load_daily_model():
            return jsonify({"error": "XGBoost model not loaded"}), 500
    
    # Get prediction from XGBoost
    # Use median values from Thailand climate data if not provided
    weather_data = {
        'temp_mean': data.get('temp_mean', 30.0),
        'temp_max': data.get('temp_max', 35.0),
        'temp_min': data.get('temp_min', 25.0),
        'temp_std': data.get('temp_std', 4.0),
        'humidity': data.get('humidity', 70.0),
        'precipitation': data.get('precipitation', 0.0),
    }
    
    try:
        result = predict_from_daily_weather(weather_data)
        
        prob = result['heatwave_probability']
        risk_level = result['risk_level']
        
        # Build dashboard-compatible response
        risk_map = {
            'LOW': {'code': 'LOW', 'label': 'Low Risk', 'index': 0},
            'MEDIUM': {'code': 'MEDIUM', 'label': 'Medium Risk', 'index': 1},
            'HIGH': {'code': 'HIGH', 'label': 'High Risk', 'index': 2},
            'CRITICAL': {'code': 'CRITICAL', 'label': 'Critical Risk', 'index': 3},
        }
        
        risk_info = risk_map.get(risk_level, risk_map['LOW'])
        
        return jsonify({
            "status": "ok",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "generated_at": datetime.now().isoformat(),
            "risk_level": risk_info['code'],
            "risk_code": risk_info['code'],
            "risk_label": risk_info['label'],
            "risk_index": risk_info['index'],
            "probability": round(prob, 4),
            "heatwave_probability": round(prob, 4),
            "advice": f"Temperature {weather_data['temp_max']:.1f}C with {prob*100:.1f}% heatwave probability. {risk_info['label']}.",
            "model_type": "xgboost_daily",
            "input_features": weather_data,
            "threshold_used": _daily_threshold,
        })
        
    except Exception as e:
        LOGGER.exception(f"XGBoost prediction failed: {e}")
        return jsonify({"error": str(e)}), 500


def init_daily_routes(app):
    """Register daily prediction routes with the Flask app."""
    app.register_blueprint(daily_bp)
    
    # Try to load model on startup
    load_daily_model()
    
    return True