"""
Hydro Service — fetches real river discharge data from Open-Meteo Flood API.
Uses GloFAS model for global river discharge coverage.
"""

import httpx
import time
import logging
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

try:
    from services.threshold_calibrator import get_danger_level_for_location
    _HAS_CALIBRATOR = True
except ImportError:
    _HAS_CALIBRATOR = False

logger = logging.getLogger(__name__)

# ─── Cache ──────────────────────────────────────────────────
_cache = {}
CACHE_TTL = 600  # 10 minutes

def _cache_key(lat: float, lng: float) -> str:
    return f"hydro_{round(lat, 2)}_{round(lng, 2)}"


def _get_cached(lat: float, lng: float):
    key = _cache_key(lat, lng)
    entry = _cache.get(key)
    if entry and (time.time() - entry["fetched_at"]) < CACHE_TTL:
        return entry["data"]
    return None


def _set_cached(lat: float, lng: float, data: dict):
    key = _cache_key(lat, lng)
    _cache[key] = {"data": data, "fetched_at": time.time()}


# ─── Open-Meteo Flood API ──────────────────────────────────
FLOOD_API_URL = "https://flood-api.open-meteo.com/v1/flood"

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    reraise=True,
)
async def fetch_river_levels(lat: float, lng: float) -> dict:
    """
    Fetch river discharge data from Open-Meteo Flood API.
    Returns current discharge, 7-day stats, danger threshold, and trend.
    """
    cached = _get_cached(lat, lng)
    if cached:
        return cached

    params = {
        "latitude": lat,
        "longitude": lng,
        "daily": "river_discharge",
        "past_days": 7,
        "forecast_days": 3,
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(FLOOD_API_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

        result = _parse_river_response(data, lat, lng)
        _set_cached(lat, lng, result)
        logger.info(f"Fetched river discharge for ({lat}, {lng})")
        return result

    except Exception as e:
        logger.error(f"Flood API fetch failed for ({lat}, {lng}): {e}")
        cached = _get_cached(lat, lng)
        if cached:
            logger.warning("Returning stale hydro cache after API failure")
            return cached
        # Return safe defaults instead of crashing
        return {
            "lat": lat, "lng": lng,
            "current_discharge": 0,
            "avg_7day": 0, "max_7day": 0,
            "danger_threshold": 100,
            "surge_ratio": 1.0,
            "trend": "unknown",
            "daily": [],
            "error": str(e),
        }


def _parse_river_response(data: dict, lat: float, lng: float) -> dict:
    """Parse Open-Meteo Flood API response into analysis-ready structure."""
    daily = data.get("daily", {})
    times = daily.get("time", [])
    discharge_values = daily.get("river_discharge", [])

    # Filter out None values
    valid_discharges = [d for d in discharge_values if d is not None]

    if not valid_discharges:
        return {
            "lat": lat, "lng": lng,
            "current_discharge": 0,
            "avg_7day": 0, "max_7day": 0,
            "danger_threshold": 100,
            "surge_ratio": 1.0,
            "trend": "stable",
            "daily": [],
        }

    # Past 7 days = indices 0-6, today = index 7, forecast = 8-9
    past_7 = valid_discharges[:7] if len(valid_discharges) >= 7 else valid_discharges
    current_idx = min(7, len(valid_discharges) - 1)
    current = valid_discharges[current_idx]

    avg_7day = sum(past_7) / len(past_7) if past_7 else 0
    max_7day = max(past_7) if past_7 else 0

    # Dynamic danger threshold: prefer real CWC gauge data, fallback to 30% above 7d max
    indofloods_threshold = None
    if _HAS_CALIBRATOR:
        try:
            cal = get_danger_level_for_location(lat, lng)
            if cal.get("danger_level_m") is not None and cal["distance_km"] < 150:
                indofloods_threshold = cal["danger_level_m"]
                logger.info(
                    f"Using CWC danger level {indofloods_threshold}m from "
                    f"{cal['gauge_id']} ({cal['distance_km']}km, river: {cal['river']})"
                )
        except Exception as e:
            logger.debug(f"Threshold calibrator error: {e}")

    if indofloods_threshold is not None:
        danger_threshold = indofloods_threshold
    else:
        danger_threshold = max_7day * 1.3 if max_7day > 0 else 100

    # Surge ratio
    surge_ratio = current / avg_7day if avg_7day > 0 else 1.0

    # Trend detection
    if len(past_7) >= 2:
        recent = past_7[-2:]
        trend = "rising" if recent[-1] > recent[0] else "falling" if recent[-1] < recent[0] else "stable"
    else:
        trend = "stable"

    # Build daily records
    daily_records = []
    for i, t in enumerate(times):
        val = discharge_values[i] if i < len(discharge_values) else None
        daily_records.append({
            "date": t,
            "discharge": val,
            "is_forecast": i >= 7,
        })

    return {
        "lat": lat,
        "lng": lng,
        "current_discharge": round(current, 3),
        "avg_7day": round(avg_7day, 3),
        "max_7day": round(max_7day, 3),
        "danger_threshold": round(danger_threshold, 3),
        "surge_ratio": round(surge_ratio, 3),
        "trend": trend,
        "daily": daily_records,
        "fetched_at": datetime.utcnow().isoformat(),
    }
