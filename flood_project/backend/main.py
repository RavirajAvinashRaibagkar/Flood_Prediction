"""
FloodSentinel AI — FastAPI Backend
Main application entry point with CORS, routers, real-time WebSockets,
and background tasks using real data from Open-Meteo, Overpass, and OSRM.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
import json
import logging
import os
from datetime import datetime

from api.prediction_routes import router as prediction_router
from api.map_routes import router as map_router
from api.sensor_routes import router as sensor_router
from api.weather_routes import router as weather_router
from api.signals_routes import router as signals_router
from api.shelter_routes import router as shelter_router
from services.websocket_manager import manager
from services.weather_service import fetch_open_meteo
from services.hydro_service import fetch_river_levels
from services.alert_service import evaluate_alerts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default region (overridden by frontend per-request)
DEFAULT_LAT = 28.6139
DEFAULT_LNG = 77.2090


# ─── Background: Real Alert Engine ──────────────────────────
async def alert_engine():
    """Every 60 seconds: fetch real weather + hydro data, run alert rules, broadcast."""
    while True:
        await asyncio.sleep(60)
        try:
            weather = await fetch_open_meteo(DEFAULT_LAT, DEFAULT_LNG)
            hydro = await fetch_river_levels(DEFAULT_LAT, DEFAULT_LNG)

            region_data = {
                "lat": DEFAULT_LAT,
                "lng": DEFAULT_LNG,
                "rainfall_1h": weather.get("current", {}).get("rainfall_1h", 0),
                "soil_moisture": weather.get("current", {}).get("soil_moisture"),
                "precipitation_probability": weather.get("current", {}).get("precipitation_probability"),
                "current_discharge": hydro.get("current_discharge", 0),
                "avg_7day": hydro.get("avg_7day", 0),
                "surge_ratio": hydro.get("surge_ratio", 1.0),
                "danger_threshold": hydro.get("danger_threshold", 100),
                "trend": hydro.get("trend", "stable"),
            }

            new_alerts = evaluate_alerts(region_data)
            for alert in new_alerts:
                await manager.broadcast(alert, channel="alerts")
                logger.info(f"Alert broadcasted: [{alert['severity']}] {alert['message']}")

        except Exception as e:
            logger.error(f"Alert engine error: {e}")


# ─── Background: Real Sensor Broadcaster ────────────────────
async def sensor_broadcaster():
    """Every 30 seconds: fetch real river discharge and broadcast to connected clients."""
    while True:
        await asyncio.sleep(30)
        try:
            hydro = await fetch_river_levels(DEFAULT_LAT, DEFAULT_LNG)
            update = {
                "sensor_id": "LIVE-DISCHARGE",
                "water_level": hydro.get("current_discharge", 0),
                "trend": hydro.get("trend", "stable"),
                "surge_ratio": hydro.get("surge_ratio", 1.0),
                "timestamp": datetime.utcnow().isoformat(),
                "status": "active",
            }
            await manager.broadcast(update, channel="sensors")
        except Exception as e:
            logger.error(f"Sensor broadcast error: {e}")


# ─── Background: Weather Refresher ──────────────────────────
async def weather_refresher():
    """Pre-warm the weather cache every 10 minutes."""
    while True:
        await asyncio.sleep(600)
        try:
            await fetch_open_meteo(DEFAULT_LAT, DEFAULT_LNG)
            logger.info("Weather cache refreshed")
        except Exception as e:
            logger.error(f"Weather refresh error: {e}")


# ─── App Lifecycle ──────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: launch background tasks
    tasks = [
        asyncio.create_task(alert_engine()),
        asyncio.create_task(sensor_broadcaster()),
        asyncio.create_task(weather_refresher()),
    ]
    logger.info("Background tasks started: alert_engine, sensor_broadcaster, weather_refresher")
    yield
    # Shutdown
    for t in tasks:
        t.cancel()


# ─── FastAPI Application ────────────────────────────────────
app = FastAPI(
    title="FloodSentinel AI",
    description="Research-grade flood prediction & evacuation routing platform — real data from Open-Meteo, OSM, OSRM",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv(
        "CORS_ORIGINS", "http://localhost:5173,http://localhost:5174,http://localhost:3000"
    ).split(",") if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Mount Routers ──────────────────────────────────────────
app.include_router(prediction_router, prefix="/api/predict", tags=["Prediction"])
app.include_router(map_router, prefix="/api/map", tags=["Map"])
app.include_router(sensor_router, prefix="/api/sensors", tags=["Sensors"])
app.include_router(weather_router, prefix="/api/weather", tags=["Weather"])
app.include_router(signals_router, prefix="/api/signals", tags=["Emergency Signals"])
app.include_router(shelter_router, prefix="/api/shelters", tags=["Shelters"])


# ─── WebSocket Endpoints ────────────────────────────────────
@app.websocket("/ws/{channel}")
async def websocket_endpoint(websocket: WebSocket, channel: str):
    if channel not in ["alerts", "sensors", "predictions"]:
        await websocket.close(code=1008)
        return

    await manager.connect(websocket, channel=channel)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, channel=channel)


# ─── Health Check ───────────────────────────────────────────
@app.get("/health")
async def health_check():
    import hashlib
    from pathlib import Path
    
    # Feature spec from models/lstm_water_prediction
    try:
        from models.lstm_water_prediction import FEATURE_NAMES, FEATURES
        spec_str = ",".join(FEATURE_NAMES)
        spec_hash = hashlib.sha256(spec_str.encode("utf-8")).hexdigest()[:16]
    except Exception:
        FEATURE_NAMES = []
        FEATURES = 0
        spec_hash = "unknown"

    meta_file = Path(__file__).resolve().parent / ".." / "ml_training" / "data" / "processed_v3" / "feature_metadata.json"
    training_date = "2026-10-19"
    if meta_file.exists():
        try:
            with open(meta_file) as f:
                meta = json.load(f)
                training_date = meta.get("timestamp", "2026-10-19")
        except Exception:
            pass

    try:
        from services.prediction_service import prediction_service
        model_status = {
            "lstm_loaded": prediction_service.lstm_model is not None,
            "cnn_loaded": prediction_service.cnn_classifier is not None,
            "unet_loaded": prediction_service.unet_adapter is not None and prediction_service.unet_adapter.is_ready(),
        }
    except Exception:
        model_status = {"lstm_loaded": False, "cnn_loaded": False, "unet_loaded": False}

    return {
        "status": "healthy" if (model_status["lstm_loaded"] and model_status["cnn_loaded"]) else "degraded",
        "model_version": "v3",
        "feature_spec_version": "v3",
        "feature_count": FEATURES,
        "feature_spec_hash": spec_hash,
        "training_date": training_date,
        "features": FEATURE_NAMES,
        "models": {
            "lstm": "FloodLSTM-v3 (BiLSTM + AttentionPooling)",
            "cnn": "FloodCNN-MobileNetV2-Transfer",
            "unet": "FloodUNet-ResNet34 (Sen1Floods11 Pixel Segmentation)",
            **model_status,
        },
    }


@app.get("/")
async def root():
    return {
        "name": "FloodSentinel AI",
        "version": "2.0.0",
        "model_version": "v3",
        "status": "operational",
        "data_sources": {
            "weather": "Open-Meteo (real-time)",
            "river_discharge": "Open-Meteo Flood API (GloFAS)",
            "water_bodies": "OpenStreetMap (Overpass)",
            "routing": "OSRM (real roads)",
            "infrastructure": "OpenStreetMap (Overpass)",
            "elevation": "Open-Meteo Elevation API",
        },
        "endpoints": {
            "health": "/health",
            "weather": "/api/weather/current?lat=&lng=",
            "flood_zones": "/api/map/flood-zones?lat=&lng=",
            "routing": "/api/map/evacuation-route?origin_lat=&origin_lng=&dest_lat=&dest_lng=",
            "infrastructure": "/api/map/infrastructure?lat=&lng=",
            "sensors": "/api/sensors/levels?lat=&lng=",
            "risk_score": "/api/predict/risk-score?lat=&lng=",
            "websocket": "/ws/alerts, /ws/sensors",
            "docs": "/docs",
        },
    }
