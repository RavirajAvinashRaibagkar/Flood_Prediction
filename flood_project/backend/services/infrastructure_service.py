"""
Infrastructure Service — fetches real hospitals, schools, power stations, shelters
from Overpass API (OpenStreetMap).
"""

import httpx
import time
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# ─── Cache ──────────────────────────────────────────────────
_cache = {}
CACHE_TTL = 86400  # 24 hours

def _cache_key(lat: float, lng: float) -> str:
    return f"infra_{round(lat, 1)}_{round(lng, 1)}"


# Facility types to query
FACILITY_QUERIES = {
    "hospital": '["amenity"="hospital"]',
    "school": '["amenity"~"school|college|university"]',
    "power": '["power"="plant"]',
    "shelter": '["amenity"="shelter"]',
    "police": '["amenity"="police"]',
    "fire_station": '["amenity"="fire_station"]',
}


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=3, max=15),
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    reraise=True,
)
async def fetch_critical_infrastructure(lat: float, lng: float, radius_km: int = 15) -> list:
    """
    Fetch critical infrastructure points from Overpass API.
    Returns list of { type, name, lat, lng, osm_id }.
    """
    key = _cache_key(lat, lng)
    entry = _cache.get(key)
    if entry and (time.time() - entry["fetched_at"]) < CACHE_TTL:
        return entry["data"]

    radius_m = radius_km * 1000

    # Build combined query for all facility types
    node_queries = []
    for ftype, selector in FACILITY_QUERIES.items():
        node_queries.append(f'node{selector}(around:{radius_m},{lat},{lng});')
        node_queries.append(f'way{selector}(around:{radius_m},{lat},{lng});')

    query = f"""
    [out:json][timeout:30];
    (
      {chr(10).join(node_queries)}
    );
    out center;
    """

    try:
        async with httpx.AsyncClient(timeout=35.0) as client:
            resp = await client.post(OVERPASS_URL, data={"data": query})
            resp.raise_for_status()
            osm_data = resp.json()

        facilities = _parse_infrastructure(osm_data)
        _cache[key] = {"data": facilities, "fetched_at": time.time()}
        logger.info(f"Fetched {len(facilities)} infrastructure points for ({lat}, {lng})")
        return facilities

    except Exception as e:
        logger.error(f"Infrastructure fetch failed for ({lat}, {lng}): {e}")
        if entry:
            return entry["data"]
        return []


def _parse_infrastructure(osm_data: dict) -> list:
    """Parse Overpass response into infrastructure point list."""
    facilities = []
    for el in osm_data.get("elements", []):
        tags = el.get("tags", {})
        
        # Determine type
        ftype = "unknown"
        if tags.get("amenity") == "hospital":
            ftype = "hospital"
        elif tags.get("amenity") in ("school", "college", "university"):
            ftype = "school"
        elif tags.get("power") == "plant":
            ftype = "power"
        elif tags.get("amenity") == "shelter":
            ftype = "shelter"
        elif tags.get("amenity") == "police":
            ftype = "police"
        elif tags.get("amenity") == "fire_station":
            ftype = "fire_station"

        # Get coordinates
        lat = el.get("lat") or el.get("center", {}).get("lat")
        lng = el.get("lon") or el.get("center", {}).get("lon")

        if lat and lng:
            facilities.append({
                "osm_id": el.get("id"),
                "type": ftype,
                "name": tags.get("name", f"{ftype.replace('_', ' ').title()}"),
                "lat": lat,
                "lng": lng,
            })

    return facilities
