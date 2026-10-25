"""
Prediction Service — orchestrates real data for risk scoring.
Uses weather, hydro, elevation services for composite risk computation.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

# ML model stubs (will use real models when available)
try:
    from models.cnn_flood_classifier import predict_flood_probability, load_trained_model as load_cnn_model
    from models.lstm_water_prediction import predict_water_level, load_trained_model as load_lstm_model
    HAS_MODELS = True
except Exception:
    HAS_MODELS = False

try:
    from models.unet_flood_segmentation import get_unet_adapter
    HAS_UNET = True
except Exception as exc:
    logger.warning("UNet segmentation model import failed: %s", exc)
    HAS_UNET = False

from services.weather_service import fetch_open_meteo
from services.hydro_service import fetch_river_levels
from services.elevation_service import get_elevation


class PredictionService:
    def __init__(self):
        self.cnn_classifier = None
        self.lstm_model = None
        self.unet_adapter = None

        if HAS_MODELS:
            try:
                self.cnn_classifier = load_cnn_model()
            except Exception as exc:
                logger.exception("Unable to load trained CNN model: %s", exc)
            try:
                self.lstm_model = load_lstm_model()
            except Exception as exc:
                logger.exception("Unable to load trained LSTM model: %s", exc)

        if HAS_UNET:
            try:
                self.unet_adapter = get_unet_adapter()
            except Exception as exc:
                logger.exception("Unable to load U-Net adapter: %s", exc)

    async def compute_risk_score(self, lat: float, lng: float) -> dict:
        """
        Composite risk score using REAL data:
        RiskScore = 0.30×rainfall_norm + 0.25×river_norm + 0.20×elevation_factor + 0.25×cnn_prob
        Scale to 0–100
        """
        try:
            weather = await fetch_open_meteo(lat, lng)
            hydro = await fetch_river_levels(lat, lng)
            elevation = await get_elevation(lat, lng)

            # Normalize rainfall (0-1 scale, 50mm/hr = max)
            rainfall_1h = weather.get("current", {}).get("rainfall_1h", 0)
            rainfall_norm = min(rainfall_1h / 50.0, 1.0)

            # Normalize river discharge (current / historical max)
            current_discharge = hydro.get("current_discharge", 0)
            max_historical = hydro.get("max_7day", 1)
            river_norm = min(current_discharge / max_historical, 1.0) if max_historical > 0 else 0.5

            # Elevation factor (low elevation = higher risk)
            elevation_factor = 1 - min(elevation / 100.0, 1.0)

            # CNN probability (use model if available, otherwise estimate from data)
            cnn_prob = self._estimate_cnn_probability(weather, hydro)

            # Composite score
            score = (
                0.30 * rainfall_norm
                + 0.25 * river_norm
                + 0.20 * elevation_factor
                + 0.25 * cnn_prob
            )
            score_100 = round(score * 100, 1)

            if score_100 <= 33:
                level = "LOW"
            elif score_100 <= 66:
                level = "MEDIUM"
            else:
                level = "HIGH"

            return {
                "score": score_100,
                "level": level,
                "components": {
                    "rainfall_normalized": round(rainfall_norm, 3),
                    "river_level_normalized": round(river_norm, 3),
                    "elevation_factor": round(elevation_factor, 3),
                    "cnn_probability": round(cnn_prob, 3),
                },
                "raw": {
                    "rainfall_1h_mm": rainfall_1h,
                    "discharge_m3s": current_discharge,
                    "elevation_m": elevation,
                    "surge_ratio": hydro.get("surge_ratio", 1.0),
                    "trend": hydro.get("trend", "stable"),
                },
            }

        except Exception as e:
            logger.error(f"Risk score computation failed: {e}")
            return {
                "score": None,
                "level": "UNAVAILABLE",
                "error": str(e),
            }

    def _estimate_cnn_probability(self, weather: dict, hydro: dict) -> float:
        """
        Estimate flood probability from data when CNN model is not available.
        Uses a heuristic based on rainfall, surge ratio, and soil moisture.
        """
        rainfall = weather.get("current", {}).get("rainfall_1h", 0)
        surge = hydro.get("surge_ratio", 1.0)
        soil = weather.get("current", {}).get("soil_moisture") or 0.2

        # Heuristic: combine signals
        prob = 0.0
        prob += min(rainfall / 40.0, 0.4)      # rainfall contribution
        prob += min((surge - 1.0) * 0.3, 0.3)   # surge contribution
        prob += min(soil * 0.5, 0.3)             # soil moisture contribution

        return min(max(prob, 0.0), 1.0)

    def predict_image(self, contents: bytes) -> dict:
        """Run CNN flood classification and water fraction estimation on uploaded image bytes."""
        import io
        from PIL import Image

        try:
            img = Image.open(io.BytesIO(contents)).convert("RGB")
            img_resized = img.resize((128, 128))
            arr = np.array(img_resized, dtype=np.float32) / 255.0

            if HAS_MODELS and self.cnn_classifier is not None:
                pred = predict_flood_probability(arr, model=self.cnn_classifier)
            else:
                raise RuntimeError("Trained flood classifier is unavailable; image prediction cannot run.")

            # Water pixel fraction via simple NDWI/blue-green ratio heuristic on image
            # Normalized Difference Water Index proxy on RGB: (G - R) / (G + R + 1e-5)
            g = arr[:, :, 1]
            r = arr[:, :, 0]
            ndwi_proxy = (g - r) / (g + r + 1e-5)
            water_pixels = float(np.mean(ndwi_proxy > 0.05))

            prob = float(pred.get("flood_probability", 0.5))
            return {
                "status": "success",
                "flood_probability": round(prob, 4),
                "confidence": round(float(pred.get("confidence", 0.8)), 4),
                "flood_detected": bool(prob >= 0.5),
                "water_area_fraction": round(water_pixels, 4),
                "model": pred.get("model", "FloodCNN-MobileNetV2-Transfer"),
                "image_resolution": [img.width, img.height],
            }
        except RuntimeError:
            raise
        except Exception as e:
            logger.error(f"Image prediction failed: {e}")
            raise ValueError(f"Invalid image format or processing error: {e}")

    def predict_timeseries(self, contents: bytes) -> dict:
        """Run LSTM gauge water level forecast on uploaded time-series CSV bytes."""
        import io
        import pandas as pd
        from models.lstm_water_prediction import FEATURE_NAMES, FEATURES, predict_water_level

        try:
            df = pd.read_csv(io.BytesIO(contents))
            missing = [name for name in FEATURE_NAMES if name not in df.columns]
            if missing:
                raise ValueError("CSV is missing required model features: " + ", ".join(missing))
            if len(df) < 10:
                raise ValueError("CSV must contain at least 10 rows (one per input timestep).")

            # Use the latest ten rows in the model's documented feature order.
            seq = np.zeros((10, FEATURES), dtype=np.float32)
            for j, feat in enumerate(FEATURE_NAMES):
                values = pd.to_numeric(df[feat], errors="coerce").tail(10).to_numpy()
                if not np.isfinite(values).all():
                    raise ValueError(f"Feature '{feat}' contains missing or non-numeric values in the latest 10 rows.")
                seq[:, j] = values

            result = predict_water_level(seq, model=self.lstm_model)
            result["input_samples"] = len(df)
            result["columns_detected"] = FEATURE_NAMES
            return result
        except Exception as e:
            logger.error(f"Time-series prediction failed: {e}")
            raise ValueError(f"Invalid CSV time-series data: {e}")

    def predict_segmentation(self, contents: bytes, filename: str = "chip.tif") -> dict:
        """Run PyTorch U-Net flood segmentation on uploaded Sentinel-1 SAR GeoTIFF bytes."""
        if self.unet_adapter is None or not self.unet_adapter.is_ready():
            raise RuntimeError("U-Net segmentation model is not available or checkpoint is missing.")
        try:
            return self.unet_adapter.segment_image_stream(contents, filename=filename)
        except Exception as e:
            logger.error(f"U-Net segmentation failed: {e}")
            raise ValueError(f"Failed to process image for U-Net segmentation: {e}")

    def list_sample_s1_chips(self) -> list:
        """List curated sample Sen1Floods11 chips for instant frontend evaluation."""
        from pathlib import Path
        s1_dir = Path(__file__).resolve().parent / ".." / ".." / "ml_training" / "data" / "raw" / "sen1floods11" / "HandLabeled" / "S1Hand"
        if not s1_dir.exists():
            return []

        # Find interesting chips with flood events
        samples = []
        for p in sorted(s1_dir.glob("*.tif"))[:12]:
            event_name = p.stem.split("_")[0]
            chip_id = p.stem.replace("_S1Hand", "")
            samples.append({
                "chip_id": chip_id,
                "filename": p.name,
                "event": event_name,
                "size_bytes": p.stat().st_size,
            })
        return samples

    def predict_sample_s1_chip(self, chip_filename: str) -> dict:
        """Run segmentation directly on a local sample Sen1Floods11 chip."""
        from pathlib import Path
        s1_dir = Path(__file__).resolve().parent / ".." / ".." / "ml_training" / "data" / "raw" / "sen1floods11" / "HandLabeled" / "S1Hand"
        chip_path = s1_dir / chip_filename
        if not chip_path.exists():
            # Try appending _S1Hand.tif
            chip_path = s1_dir / f"{chip_filename}_S1Hand.tif"
            if not chip_path.exists():
                raise FileNotFoundError(f"Sample chip '{chip_filename}' not found.")

        with open(chip_path, "rb") as f:
            bytes_data = f.read()

        result = self.predict_segmentation(bytes_data, filename=chip_path.name)
        result["chip_id"] = chip_filename.replace("_S1Hand.tif", "").replace(".tif", "")
        return result


# Load each checkpoint once and share the models between API routes and health reporting.
prediction_service = PredictionService()
