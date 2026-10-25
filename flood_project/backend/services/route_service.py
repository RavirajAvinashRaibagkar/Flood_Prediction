"""
Route Service — OSRM public API for real road routing + flood overlap scoring.
"""

import logging
from math import radians, sin, cos, sqrt, atan2
from typing import Tuple

import httpx
from shapely.geometry import LineString, shape

logger = logging.getLogger(__name__)

OSRM_URL = "https://router.project-osrm.org/route/v1/driving"


def _haversine_km(lat1, lng1, lat2, lng2) -> float:
    R = 6371.0
    dlat, dlng = radians(lat2 - lat1), radians(lng2 - lng1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlng / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


async def _fetch_osrm(url: str, params: dict) -> dict:
    """GET with one retry. (The previous tenacity decorator never fired because the
    caller swallowed every exception, so retries are done explicitly here.)"""
    last_exc = None
    for _ in range(2):
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                return resp.json()
        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            last_exc = exc
    raise last_exc


async def compute_routes(
    origin_lat: float, origin_lng: float,
    dest_lat: float, dest_lng: float,
    flood_zones_geojson: dict = None,
) -> list:
    """
    Fetch alternative routes from OSRM and score each for flood risk.
    Returned list is sorted safest-first and ids are assigned AFTER sorting,
    so `route_0` is always the safest route.
    """
    url = f"{OSRM_URL}/{origin_lng},{origin_lat};{dest_lng},{dest_lat}"
    params = {"overview": "full", "geometries": "geojson", "alternatives": "true", "steps": "false"}

    try:
        data = await _fetch_osrm(url, params)
        if data.get("code") != "Ok":
            logger.warning("OSRM returned: %s", data.get("code"))
            return _fallback_route(origin_lat, origin_lng, dest_lat, dest_lng)

        scored = []
        for route in data.get("routes", []):
            geom = route.get("geometry", {})
            distance_m = route.get("distance", 0)
            duration_s = route.get("duration", 0)

            risk_score, river_crossings = 0.0, 0
            if flood_zones_geojson and geom.get("coordinates"):
                risk_score, river_crossings = _score_flood_overlap(geom, flood_zones_geojson)

            length_ratio = min(distance_m / 10000, 1.0)
            composite = (0.5 * risk_score) + (0.3 * length_ratio) + (0.2 * min(river_crossings / 3, 1.0))

            scored.append({
                "geometry": geom,
                "distance_km": round(distance_m / 1000, 2),
                "duration_min": round(duration_s / 60, 1),
                "risk_score": round(composite * 100, 1),
                "flood_overlap": round(risk_score * 100, 1),
                "river_crossings": river_crossings,
                "coordinates": _geojson_to_leaflet(geom),
                "fallback": False,
            })

        scored.sort(key=lambda r: r["risk_score"])
        for i, r in enumerate(scored):
            r["id"] = f"route_{i}"
            r["label"] = "Route A (Safest)" if i == 0 else f"Route {chr(65 + i)}"
        return scored

    except Exception as e:
        logger.error("OSRM routing failed: %s", e)
        return _fallback_route(origin_lat, origin_lng, dest_lat, dest_lng)


def _score_flood_overlap(route_geojson: dict, flood_zones_geojson: dict) -> Tuple[float, int]:
    """Score what fraction of a route intersects with flood zone geometry."""
    try:
        coords = route_geojson.get("coordinates", [])
        if len(coords) < 2:
            return 0.0, 0

        route_line = LineString(coords)
        if route_line.length == 0:
            return 0.0, 0
        flood_overlap, river_crossings = 0.0, 0

        for feature in flood_zones_geojson.get("features", []):
            geom = feature.get("geometry")
            if not geom:
                continue
            try:
                zone_shape = shape(geom)
                if route_line.intersects(zone_shape):
                    flood_overlap += route_line.intersection(zone_shape).length / route_line.length
                    river_crossings += 1
            except Exception:
                continue

        return min(flood_overlap, 1.0), river_crossings

    except Exception as e:
        logger.warning("Flood overlap scoring failed: %s", e)
        return 0.0, 0


def _geojson_to_leaflet(geom: dict) -> list:
    """Convert GeoJSON [lng, lat] to Leaflet [lat, lng]."""
    return [[c[1], c[0]] for c in geom.get("coordinates", [])]


def _fallback_route(olat, olng, dlat, dlng) -> list:
    """Straight line used only when OSRM is unreachable. Flagged so the UI can say so
    instead of presenting it as a real, scored road route."""
    return [{
        "id": "route_fallback",
        "label": "Direct line (road routing unavailable)",
        "geometry": {"type": "LineString", "coordinates": [[olng, olat], [dlng, dlat]]},
        "distance_km": round(_haversine_km(olat, olng, dlat, dlng), 2),
        "duration_min": 0,
        "risk_score": 50,
        "flood_overlap": 0,
        "river_crossings": 0,
        "coordinates": [[olat, olng], [dlat, dlng]],
        "fallback": True,
    }]
