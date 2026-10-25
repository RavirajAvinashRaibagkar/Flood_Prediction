"""
test_backend_endpoints.py  --  FloodSentinel AI (Phase 6 Deliverable)
=====================================================================
Tests all FastAPI endpoints end-to-end using TestClient:
  - GET  /health (model version, spec hash, training date)
  - GET  / (API root and endpoints map)
  - GET  /api/predict/risk-score (composite risk)
  - POST /api/predict/image (CNN classifier on satellite patch)
  - POST /api/predict/timeseries (LSTM forecast on time series CSV)
"""

import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np
import pandas as pd

from main import app

client = TestClient(app)

class TestBackendAPI:

    def test_health_endpoint(self):
        """GET /health must return model_version v3, feature_spec_hash, and training_date."""
        resp = client.get("/health")
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["model_version"] == "v3"
        assert "feature_spec_hash" in data
        assert len(data["feature_spec_hash"]) > 0
        assert data["feature_count"] == 11
        assert "training_date" in data
        assert "antecedent_precipitation_index" in data["features"]

    def test_root_endpoint(self):
        """GET / returns status operational and endpoints catalog."""
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "operational"
        assert "weather" in data["data_sources"]
        assert "routing" in data["data_sources"]

    def test_risk_score_endpoint(self):
        """GET /api/predict/risk-score returns composite risk 0-100."""
        resp = client.get("/api/predict/risk-score?lat=25.5941&lng=85.1376")
        assert resp.status_code == 200
        data = resp.json()
        assert "score" in data
        assert 0.0 <= data["score"] <= 100.0
        assert data["level"] in ["LOW", "MEDIUM", "HIGH"]
        assert "components" in data

    def test_predict_image_endpoint(self):
        """POST /api/predict/image with synthetic satellite patch."""
        # Create a 128x128 synthetic test image
        img = Image.new("RGB", (128, 128), color=(30, 80, 150)) # bluish water patch
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        resp = client.post(
            "/api/predict/image",
            files={"file": ("test_chip.png", buf, "image/png")}
        )
        assert resp.status_code == 200, f"Failed image predict: {resp.text}"
        data = resp.json()
        assert data["status"] == "success"
        assert "flood_probability" in data
        assert 0.0 <= data["flood_probability"] <= 1.0
        assert "water_area_fraction" in data
        assert "flood_detected" in data

    def test_predict_timeseries_endpoint(self):
        """POST /api/predict/timeseries with valid CSV timeseries."""
        # Create minimal 10-row time series CSV with 11 features
        df = pd.DataFrame({
            "rainfall_cumulative_mm": np.linspace(10, 80, 10),
            "rainfall_delta_mm": np.ones(10) * 5.0,
            "peak_discharge_cumec": np.linspace(200, 800, 10),
            "antecedent_precipitation_index": np.linspace(15, 60, 10),
            "event_duration_days": np.ones(10) * 4.0,
            "dfsi_severity_index": np.ones(10) * 0.45,
            "lat": np.ones(10) * 25.59,
            "lon": np.ones(10) * 85.13,
            "danger_level_m": np.ones(10) * 50.0,
            "warning_level_m": np.ones(10) * 48.0,
            "catchment_area_km2": np.ones(10) * 4500.0,
        })
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        buf.seek(0)

        resp = client.post(
            "/api/predict/timeseries",
            files={"file": ("test_event.csv", buf, "text/csv")}
        )
        assert resp.status_code == 200, f"Failed timeseries predict: {resp.text}"
        data = resp.json()
        assert "predictions" in data
        assert len(data["predictions"]) > 0
        assert "predicted_level" in data["predictions"][0]

    def test_sample_s1_endpoint(self):
        """GET /api/predict/sample-s1 lists available preloaded Sentinel-1 chips."""
        resp = client.get("/api/predict/sample-s1")
        assert resp.status_code == 200
        data = resp.json()
        assert "samples" in data
        assert isinstance(data["samples"], list)
        if len(data["samples"]) > 0:
            assert "chip_id" in data["samples"][0]
            assert "filename" in data["samples"][0]

    def test_segmentation_endpoint_synthetic(self):
        """POST /api/predict/segmentation with image stream."""
        img = Image.new("RGB", (256, 256), color=(40, 90, 160))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        resp = client.post(
            "/api/predict/segmentation",
            files={"file": ("test_chip.png", buf, "image/png")},
        )
        assert resp.status_code == 200, f"Failed segmentation: {resp.text}"
        data = resp.json()
        assert data["status"] == "success"
        assert "U-Net" in data["model"]
        assert "flood_pixel_count" in data
        assert "flood_area_fraction" in data
        assert "mask_overlay_b64" in data
        assert data["mask_overlay_b64"].startswith("data:image/png;base64,")

    def test_segment_sample_chip_endpoint(self):
        """POST /api/predict/segment-sample runs real S1 GeoTIFF through U-Net."""
        samples_resp = client.get("/api/predict/sample-s1")
        assert samples_resp.status_code == 200
        samples = samples_resp.json().get("samples", [])
        if len(samples) > 0:
            chip_file = samples[0]["filename"]
            resp = client.post(f"/api/predict/segment-sample?chip_filename={chip_file}")
            assert resp.status_code == 200, f"Failed segment sample: {resp.text}"
            data = resp.json()
            assert data["dimensions"]["width"] == 512
            assert data["dimensions"]["height"] == 512
            assert "flood_pixel_count" in data
            assert "mask_overlay_b64" in data
            assert data["mask_overlay_b64"].startswith("data:image/png;base64,")

    def test_sensor_history_endpoint_returns_501(self):
        """GET /api/sensors/{id} must return 501 No sensor history source configured."""
        resp = client.get("/api/sensors/1")
        assert resp.status_code == 501
        assert "No sensor history source configured" in resp.json()["detail"]


class TestHydroService:
    def test_hydro_service_failure_returns_available_false(self, monkeypatch):
        """hydro_service returns available=false (not 0) when the HTTP call fails."""
        import asyncio
        import httpx
        from services.hydro_service import fetch_river_levels, _cache
        from tenacity import stop_after_attempt

        _cache.clear()

        orig_stop = fetch_river_levels.retry.stop
        fetch_river_levels.retry.stop = stop_after_attempt(1)

        async def mock_get(*args, **kwargs):
            raise httpx.ConnectError("GloFAS API connection timed out")

        monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

        try:
            result = asyncio.run(fetch_river_levels(28.6139, 77.2090))
        finally:
            fetch_river_levels.retry.stop = orig_stop

        assert result["available"] is False
        assert result["current_discharge"] is None
        assert result["danger_threshold_discharge"] is None
        assert result["trend"] == "unknown"
        assert result["daily"] == []
        assert "error" in result
        assert result["source"] == "Open-Meteo GloFAS (modelled)"

    def test_thresholds_never_swapped(self):
        """danger_threshold_discharge (m3/s) and danger_level_m (stage) are never swapped."""
        from services.hydro_service import _parse_river_response

        mock_data = {
            "daily": {
                "time": [f"2026-10-{i:02d}" for i in range(1, 11)],
                "river_discharge": [10.0, 12.0, 15.0, 20.0, 18.0, 14.0, 16.0, 22.0, 19.0, 17.0],
                "river_discharge_p25": [9.0] * 10,
                "river_discharge_p75": [24.0] * 10,
            }
        }

        # 7-day max is 20.0, so 1.3 * 20.0 = 26.0 m3/s
        result = _parse_river_response(mock_data, 28.6139, 77.2090)
        assert result["available"] is True
        assert result["danger_threshold_discharge"] == 26.0
        # If danger_level_m is calibrated from CWC gauge, verify distinct units and never swapped
        if result["danger_level_m"] is not None:
            assert result["danger_threshold_discharge"] != result["danger_level_m"]
        assert result["source"] == "Open-Meteo GloFAS (modelled)"
        assert "danger_source" in result
        assert "fetched_at" in result
        # Check bands
        assert len(result["daily"]) > 0
        assert result["daily"][0]["lower"] == 9.0
        assert result["daily"][0]["upper"] == 24.0

