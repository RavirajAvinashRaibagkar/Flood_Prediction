"""
Threshold Calibrator — Flood AI
=========================================
Loads the CWC danger thresholds exported by data_pipeline.py
(sourced from real INDOFLOODS metadata) and provides a nearest-gauge
lookup that calibrates hydro_service.py with real values.

Usage (in hydro_service.py or prediction_service.py):
    from services.threshold_calibrator import get_danger_level_for_location
    danger_m = get_danger_level_for_location(lat, lng)
"""

import os
import json
import math
import logging

logger = logging.getLogger(__name__)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_THRESHOLDS_PATH = os.path.join(
    _THIS_DIR, "..", "..", "ml_training", "data", "processed", "danger_thresholds.json"
)

_thresholds: dict = {}


def _load_thresholds():
    global _thresholds
    if _thresholds:
        return
    if not os.path.exists(_THRESHOLDS_PATH):
        logger.warning("danger_thresholds.json not found — run data_pipeline.py first")
        return
    with open(_THRESHOLDS_PATH, encoding="utf-8") as f:
        _thresholds = json.load(f)
    logger.info(f"Loaded {len(_thresholds):,} CWC gauge thresholds from INDOFLOODS metadata")


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in km."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def get_danger_level_for_location(lat: float, lng: float, max_radius_km: float = 200.0) -> dict:
    """
    Find the nearest CWC gauge within max_radius_km and return its
    danger level, warning level, and metadata.

    Returns:
        dict with keys: danger_level_m, warning_level_m, gauge_id,
                        distance_km, river, state
        Falls back to None values if no gauge found within radius.
    """
    _load_thresholds()

    if not _thresholds:
        return {
            "danger_level_m": None,
            "warning_level_m": None,
            "gauge_id": None,
            "distance_km": None,
            "river": None,
            "state": None,
        }

    best_gauge  = None
    best_dist   = float("inf")
    best_info   = {}

    for gauge_id, info in _thresholds.items():
        g_lat = info.get("lat")
        g_lng = info.get("lng")
        if g_lat is None or g_lng is None:
            continue
        dist = _haversine_km(lat, lng, g_lat, g_lng)
        if dist < best_dist:
            best_dist  = dist
            best_gauge = gauge_id
            best_info  = info

    if best_dist > max_radius_km:
        logger.debug(f"Nearest gauge is {best_dist:.1f} km away — outside {max_radius_km} km radius")
        return {
            "danger_level_m": None,
            "warning_level_m": None,
            "gauge_id": best_gauge,
            "distance_km": round(best_dist, 1),
            "river": best_info.get("river"),
            "state": best_info.get("state"),
        }

    return {
        "danger_level_m":  best_info.get("danger_level_m"),
        "warning_level_m": best_info.get("warning_level_m"),
        "gauge_id":        best_gauge,
        "distance_km":     round(best_dist, 1),
        "river":           best_info.get("river"),
        "state":           best_info.get("state"),
    }


def get_all_thresholds() -> dict:
    """Return the full threshold dictionary."""
    _load_thresholds()
    return _thresholds
