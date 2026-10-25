"""
Hydro Service — fetches real river discharge data from Open-Meteo Flood API.
Uses GloFAS model for global river discharge coverage.
"""

import httpx
import time
import logging
from datetime import datetime, timezone
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
    Returns current discharge, 7-day stats, distinct discharge/stage thresholds,
    trend, and band percentiles (p25/p75).
    """
    cached = _get_cached(lat, lng)
    if cached and cached.get("available"):
        return cached

    params = {
        "latitude": lat,
        "longitude": lng,
        "daily": ["river_discharge", "river_discharge_p25", "river_discharge_p75"],
        "past_days": 7,
        "forecast_days": 3,
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(FLOOD_API_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

        result = _parse_river_response(data, lat, lng)
        if result.get("available"):
            _set_cached(lat, lng, result)
        logger.info(f"Fetched river discharge for ({lat}, {lng})")
        return result

    except Exception as e:
        logger.error(f"Flood API fetch failed for ({lat}, {lng}): {e}")
        cached = _get_cached(lat, lng)
        if cached:
            logger.warning("Returning stale hydro cache after API failure")
            return cached

        gauge_id = None
        distance_km = None
        danger_level_m = None
        if _HAS_CALIBRATOR:
            try:
                cal = get_danger_level_for_location(lat, lng)
                if cal.get("danger_level_m") is not None and cal.get("distance_km", 999) < 150:
                    danger_level_m = cal["danger_level_m"]
                    gauge_id = cal.get("gauge_id")
                    distance_km = cal.get("distance_km")
            except Exception:
                pass

        return {
            "available": False,
            "source": "Open-Meteo GloFAS (modelled)",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "lat": lat,
            "lng": lng,
            "current_discharge": None,
            "avg_7day": None,
            "max_7day": None,
            "danger_threshold_discharge": None,
            "danger_threshold": None,
            "danger_level_m": danger_level_m,
            "gauge_id": gauge_id,
            "distance_km": distance_km,
            "danger_source": "none",
            "surge_ratio": None,
            "trend": "unknown",
            "daily": [],
            "error": str(e),
        }


def _parse_river_response(data: dict, lat: float, lng: float) -> dict:
    """Parse Open-Meteo Flood API response into analysis-ready structure."""
    daily = data.get("daily", {})
    times = daily.get("time", [])
    discharge_values = daily.get("river_discharge", [])
    p25_values = daily.get("river_discharge_p25", [])
    p75_values = daily.get("river_discharge_p75", [])

    gauge_id = None
    distance_km = None
    danger_level_m = None
    if _HAS_CALIBRATOR:
        try:
            cal = get_danger_level_for_location(lat, lng)
            if cal.get("danger_level_m") is not None and cal.get("distance_km", 999) < 150:
                danger_level_m = cal["danger_level_m"]
                gauge_id = cal.get("gauge_id")
                distance_km = cal.get("distance_km")
                logger.info(
                    f"CWC danger level {danger_level_m}m from gauge {gauge_id} ({distance_km}km)"
                )
        except Exception as e:
            logger.debug(f"Threshold calibrator error: {e}")

    # Filter out None values
    valid_discharges = [d for d in discharge_values if d is not None]

    if not valid_discharges:
        return {
            "available": False,
            "source": "Open-Meteo GloFAS (modelled)",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "lat": lat,
            "lng": lng,
            "current_discharge": None,
            "avg_7day": None,
            "max_7day": None,
            "danger_threshold_discharge": None,
            "danger_threshold": None,
            "danger_level_m": danger_level_m,
            "gauge_id": gauge_id,
            "distance_km": distance_km,
            "danger_source": "none",
            "surge_ratio": None,
            "trend": "unknown",
            "daily": [],
            "error": "No valid discharge observations in response",
        }

    # Past 7 days = indices 0-6, today = index 7, forecast = 8-9
    past_7 = valid_discharges[:7] if len(valid_discharges) >= 7 else valid_discharges
    current_idx = min(7, len(valid_discharges) - 1)
    current = valid_discharges[current_idx]

    avg_7day = sum(past_7) / len(past_7) if past_7 else 0.0
    max_7day = max(past_7) if past_7 else 0.0

    # Separate thresholds:
    # 1. danger_threshold_discharge (m3/s): 1.3 * max past 7 days
    danger_threshold_discharge = round(max_7day * 1.3, 3) if max_7day > 0 else None

    # 2. danger_source describing which one is used for discharge
    if danger_threshold_discharge is not None:
        danger_source = "1.3x 7-day max discharge (GloFAS modelled)"
    elif danger_level_m is not None:
        danger_source = f"CWC gauge stage ({gauge_id})"
    else:
        danger_source = "none"

    # Surge ratio
    surge_ratio = round(current / avg_7day, 3) if avg_7day > 0 else 1.0

    # Trend detection
    if len(past_7) >= 2:
        recent = past_7[-2:]
        trend = "rising" if recent[-1] > recent[0] else "falling" if recent[-1] < recent[0] else "stable"
    else:
        trend = "stable"

    # Build daily records with real lower/upper percentile band if present
    daily_records = []
    for i, t in enumerate(times):
        val = discharge_values[i] if i < len(discharge_values) else None
        rec = {
            "date": t,
            "discharge": val,
            "is_forecast": i >= 7,
        }
        if (
            i < len(p25_values) and p25_values[i] is not None
            and i < len(p75_values) and p75_values[i] is not None
        ):
            rec["lower"] = p25_values[i]
            rec["upper"] = p75_values[i]
        daily_records.append(rec)

    return {
        "available": True,
        "source": "Open-Meteo GloFAS (modelled)",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "lat": lat,
        "lng": lng,
        "current_discharge": round(current, 3),
        "avg_7day": round(avg_7day, 3),
        "max_7day": round(max_7day, 3),
        "danger_threshold_discharge": danger_threshold_discharge,
        "danger_threshold": danger_threshold_discharge,
        "danger_level_m": danger_level_m,
        "gauge_id": gauge_id,
        "distance_km": distance_km,
        "danger_source": danger_source,
        "surge_ratio": surge_ratio,
        "trend": trend,
        "daily": daily_records,
    }
