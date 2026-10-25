"""
OSM Service — fetches real water body geometry from Overpass API.
Converts to GeoJSON for flood zone overlay on the map.
"""

import httpx
import time
import logging
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# ─── Cache ──────────────────────────────────────────────────
_cache = {}
CACHE_TTL = 86400  # 24 hours — OSM data changes slowly

def _cache_key(lat: float, lng: float) -> str:
    return f"osm_{round(lat, 1)}_{round(lng, 1)}"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=3, max=15),
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    reraise=True,
)
async def fetch_water_bodies(lat: float, lng: float, radius_km: int = 20) -> dict:
    """
    Fetch water bodies (rivers, lakes, canals) from Overpass API.
    Returns GeoJSON FeatureCollection.
    Cached for 24 hours per ~10km grid cell.
    """
    key = _cache_key(lat, lng)
    entry = _cache.get(key)
    if entry and (time.time() - entry["fetched_at"]) < CACHE_TTL:
        return entry["data"]

    radius_m = radius_km * 1000
    query = f"""
    [out:json][timeout:30];
    (
      way["natural"="water"](around:{radius_m},{lat},{lng});
      way["waterway"~"river|stream|canal"](around:{radius_m},{lat},{lng});
      relation["natural"="water"](around:{radius_m},{lat},{lng});
    );
    out geom;
    """

    try:
        async with httpx.AsyncClient(timeout=35.0) as client:
            resp = await client.post(OVERPASS_URL, data={"data": query})
            resp.raise_for_status()
            osm_data = resp.json()

        geojson = _osm_to_geojson(osm_data)
        _cache[key] = {"data": geojson, "fetched_at": time.time()}
        logger.info(f"Fetched {len(geojson['features'])} water features for ({lat}, {lng})")
        return geojson

    except Exception as e:
        logger.error(f"Overpass fetch failed for ({lat}, {lng}): {e}")
        if entry:
            logger.warning("Returning stale OSM cache")
            return entry["data"]
        return {"type": "FeatureCollection", "features": [], "degraded": True}


def _osm_to_geojson(osm_data: dict) -> dict:
    """Convert Overpass API response to GeoJSON FeatureCollection."""
    features = []
    elements = osm_data.get("elements", [])

    for el in elements:
        geom = None
        props = {
            "osm_id": el.get("id"),
            "osm_type": el.get("type"),
            "name": el.get("tags", {}).get("name", "Unknown water body"),
            "waterway": el.get("tags", {}).get("waterway"),
            "natural": el.get("tags", {}).get("natural"),
        }

        if el["type"] == "way" and "geometry" in el:
            coords = [[n["lon"], n["lat"]] for n in el["geometry"]]
            if len(coords) >= 3:
                # Check if it forms a closed polygon
                if coords[0] == coords[-1]:
                    geom = {"type": "Polygon", "coordinates": [coords]}
                else:
                    geom = {"type": "LineString", "coordinates": coords}

        elif el["type"] == "relation" and "members" in el:
            # Build multipolygon from outer ways
            all_coords = []
            for member in el.get("members", []):
                if member.get("type") == "way" and "geometry" in member:
                    way_coords = [[n["lon"], n["lat"]] for n in member["geometry"]]
                    if way_coords:
                        all_coords.append(way_coords)
            if all_coords:
                # Simplified: treat as first ring
                if len(all_coords[0]) >= 4:
                    geom = {"type": "Polygon", "coordinates": [all_coords[0]]}
                elif len(all_coords[0]) >= 2:
                    geom = {"type": "LineString", "coordinates": all_coords[0]}

        if geom:
            features.append({
                "type": "Feature",
                "geometry": geom,
                "properties": props,
            })

    return {
        "type": "FeatureCollection",
        "features": features,
        "fetched_at": datetime.utcnow().isoformat(),
    }
