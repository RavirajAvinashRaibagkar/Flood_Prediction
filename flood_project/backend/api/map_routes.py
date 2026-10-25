"""
Map API Routes — now with real data from OSM, OSRM, and infrastructure services.
GET /api/map/flood-zones       — Real water body GeoJSON from Overpass
GET /api/map/evacuation-route  — Real OSRM routing with flood overlap scoring
GET /api/map/sensors           — Location-aware virtual sensors
GET /api/map/infrastructure    — Real hospitals/schools/shelters from OSM
"""

from fastapi import APIRouter, Query, HTTPException
from services.osm_service import fetch_water_bodies
from services.route_service import compute_routes
from services.infrastructure_service import fetch_critical_infrastructure

router = APIRouter()


@router.get("/flood-zones")
async def get_flood_zones(
    lat: float = Query(None, description="Latitude"),
    lng: float = Query(None, description="Longitude"),
    radius_km: int = Query(20, description="Search radius in km"),
):
    """Return real GeoJSON of water bodies from OpenStreetMap."""
    if lat is None or lng is None:
        raise HTTPException(status_code=400, detail="lat and lng are required")
    try:
        geojson = await fetch_water_bodies(lat, lng, radius_km)
        return geojson
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OSM API unavailable: {str(e)}")


@router.get("/evacuation-route")
async def get_evacuation_route(
    origin_lat: float = Query(..., description="Origin latitude"),
    origin_lng: float = Query(..., description="Origin longitude"),
    dest_lat: float = Query(..., description="Destination latitude"),
    dest_lng: float = Query(..., description="Destination longitude"),
):
    """Compute safest evacuation route using OSRM + flood overlap scoring."""
    try:
        # Fetch flood zones for overlap scoring
        mid_lat = (origin_lat + dest_lat) / 2
        mid_lng = (origin_lng + dest_lng) / 2
        flood_zones = await fetch_water_bodies(mid_lat, mid_lng, radius_km=10)

        routes = await compute_routes(
            origin_lat, origin_lng,
            dest_lat, dest_lng,
            flood_zones_geojson=flood_zones,
        )
        return {
            "routes": routes,
            "count": len(routes),
            "flood_data_available": not flood_zones.get("degraded", False),
            "algorithm": "OSRM + Shapely flood overlap scoring",
        }
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Routing failed: {str(e)}")


@router.get("/sensors")
async def get_sensors(
    lat: float = Query(None, description="Latitude"),
    lng: float = Query(None, description="Longitude"),
):
    """Return virtual sensors near a location."""
    if lat is None or lng is None:
        return []
    # Generate virtual sensors around the location
    offsets = [
        (0.01, 0.01, "River Gauge North", "water_level"),
        (-0.01, 0.02, "Bridge Sensor East", "water_level"),
        (-0.02, -0.01, "Barrage Station South", "water_level"),
        (0.02, -0.005, "Upstream Monitor", "rainfall"),
        (-0.005, 0.015, "Drainage Sensor", "water_level"),
        (0.015, -0.02, "Downstream Gauge", "rainfall"),
    ]
    sensors = []
    for i, (dlat, dlng, name, stype) in enumerate(offsets, start=1):
        sensors.append({
            "id": i,
            "name": name,
            "lat": round(lat + dlat, 6),
            "lng": round(lng + dlng, 6),
            "type": stype,
            "status": "active",
        })
    return sensors


@router.get("/infrastructure")
async def get_infrastructure(
    lat: float = Query(None, description="Latitude"),
    lng: float = Query(None, description="Longitude"),
    radius_km: int = Query(15, description="Search radius in km"),
):
    """Return real critical infrastructure points from OpenStreetMap."""
    if lat is None or lng is None:
        raise HTTPException(status_code=400, detail="lat and lng are required")
    try:
        facilities = await fetch_critical_infrastructure(lat, lng, radius_km)
        return facilities
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Infrastructure API unavailable: {str(e)}")
