"""
Shelter Support API — unified institutional + community shelter management.

- Institutional shelters (hospital, school, government) are persistent
- Community shelters are anonymous and auto-expire after 45 minutes
- Ownership is via private `shelter_id` (stored in client localStorage)
- List endpoints never expose the private `shelter_id`; only a short `public_id`
"""

from datetime import datetime, timedelta, timezone
from typing import Literal, Optional, List
from math import radians, cos, sin, sqrt, atan2

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from database.connection import get_db
from database.models import Shelter
from services.route_service import compute_routes
from services.osm_service import fetch_water_bodies

router = APIRouter()

COMMUNITY_EXPIRY_MINUTES = 45


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _cleanup_expired(db: Session) -> int:
    """Remove community shelters that haven't been updated recently."""
    cutoff = _utcnow() - timedelta(minutes=COMMUNITY_EXPIRY_MINUTES)
    deleted = (
        db.query(Shelter)
        .filter(Shelter.is_persistent == False, Shelter.updated_at < cutoff)
        .delete(synchronize_session=False)
    )
    if deleted:
        db.commit()
    return int(deleted or 0)


def _public_id(shelter_id: str) -> str:
    return shelter_id[:6]


def _haversine(lat1, lng1, lat2, lng2) -> float:
    """Distance in km between two points."""
    R = 6371
    dlat = radians(lat2 - lat1)
    dlng = radians(lng2 - lng1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlng / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


# ─── Pydantic Models ────────────────────────────────────────

class ShelterCreate(BaseModel):
    shelter_id: str = Field(min_length=8, max_length=64)
    type: Literal["hospital", "school", "government", "community"]
    name: Optional[str] = None
    lat: float
    lng: float
    capacity: str
    resources: List[str] = []


class ShelterUpdate(BaseModel):
    capacity: Optional[str] = None
    resources: Optional[List[str]] = None


class ShelterOut(BaseModel):
    public_id: str
    type: str
    name: Optional[str]
    lat: float
    lng: float
    capacity: str
    resources: List[str]
    is_persistent: bool
    created_at: str
    updated_at: str


class ShelterSummary(BaseModel):
    total: int
    hospitals: int
    schools: int
    government: int
    community: int
    estimated_capacity: int
    expired_cleaned: int = 0


# ─── Routes ─────────────────────────────────────────────────

@router.get("", response_model=List[ShelterOut])
def list_shelters(
    bbox: Optional[str] = Query(None, description="Optional bbox: south,west,north,east"),
    db: Session = Depends(get_db),
):
    _cleanup_expired(db)

    q = db.query(Shelter)
    if bbox:
        try:
            south, west, north, east = [float(x.strip()) for x in bbox.split(",")]
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid bbox format.")
        q = q.filter(
            Shelter.lat >= south, Shelter.lat <= north,
            Shelter.lng >= west, Shelter.lng <= east,
        )

    shelters = q.order_by(Shelter.updated_at.desc()).all()
    return [
        ShelterOut(
            public_id=_public_id(s.shelter_id),
            type=s.type,
            name=s.name,
            lat=s.lat,
            lng=s.lng,
            capacity=s.capacity,
            resources=s.resources or [],
            is_persistent=s.is_persistent,
            created_at=s.created_at.isoformat() + "Z",
            updated_at=s.updated_at.isoformat() + "Z",
        )
        for s in shelters
    ]


@router.get("/summary", response_model=ShelterSummary)
def shelter_summary(db: Session = Depends(get_db)):
    expired_cleaned = _cleanup_expired(db)
    all_rows = db.query(Shelter.type, Shelter.capacity).all()

    counts = {"hospital": 0, "school": 0, "government": 0, "community": 0}
    estimated_cap = 0

    for stype, cap in all_rows:
        if stype in counts:
            counts[stype] += 1
        # Estimate numeric capacity
        try:
            estimated_cap += int(cap)
        except ValueError:
            cap_map = {"1-2": 2, "3-5": 4, "5+": 6}
            estimated_cap += cap_map.get(cap, 2)

    return ShelterSummary(
        total=sum(counts.values()),
        hospitals=counts["hospital"],
        schools=counts["school"],
        government=counts["government"],
        community=counts["community"],
        estimated_capacity=estimated_cap,
        expired_cleaned=expired_cleaned,
    )


@router.post("", response_model=ShelterOut)
def create_shelter(payload: ShelterCreate, db: Session = Depends(get_db)):
    _cleanup_expired(db)

    existing = db.query(Shelter).filter(Shelter.shelter_id == payload.shelter_id).first()
    now = _utcnow()
    is_persistent = payload.type != "community"

    if existing:
        existing.type = payload.type
        existing.name = payload.name
        existing.lat = payload.lat
        existing.lng = payload.lng
        existing.capacity = payload.capacity
        existing.resources = payload.resources
        existing.is_persistent = is_persistent
        existing.updated_at = now
        db.commit()
        db.refresh(existing)
        s = existing
    else:
        s = Shelter(
            shelter_id=payload.shelter_id,
            type=payload.type,
            name=payload.name,
            lat=payload.lat,
            lng=payload.lng,
            capacity=payload.capacity,
            resources=payload.resources,
            is_persistent=is_persistent,
            created_at=now,
            updated_at=now,
        )
        db.add(s)
        db.commit()
        db.refresh(s)

    return ShelterOut(
        public_id=_public_id(s.shelter_id),
        type=s.type,
        name=s.name,
        lat=s.lat,
        lng=s.lng,
        capacity=s.capacity,
        resources=s.resources or [],
        is_persistent=s.is_persistent,
        created_at=s.created_at.isoformat() + "Z",
        updated_at=s.updated_at.isoformat() + "Z",
    )


@router.patch("/{shelter_id}", response_model=ShelterOut)
def update_shelter(shelter_id: str, payload: ShelterUpdate, db: Session = Depends(get_db)):
    _cleanup_expired(db)

    s = db.query(Shelter).filter(Shelter.shelter_id == shelter_id).first()
    if not s:
        raise HTTPException(status_code=404, detail="Shelter not found (or expired).")

    if payload.capacity is not None:
        s.capacity = payload.capacity
    if payload.resources is not None:
        s.resources = payload.resources
    s.updated_at = _utcnow()
    db.commit()
    db.refresh(s)

    return ShelterOut(
        public_id=_public_id(s.shelter_id),
        type=s.type,
        name=s.name,
        lat=s.lat,
        lng=s.lng,
        capacity=s.capacity,
        resources=s.resources or [],
        is_persistent=s.is_persistent,
        created_at=s.created_at.isoformat() + "Z",
        updated_at=s.updated_at.isoformat() + "Z",
    )


@router.delete("/{shelter_id}")
def delete_shelter(shelter_id: str, db: Session = Depends(get_db)):
    s = db.query(Shelter).filter(Shelter.shelter_id == shelter_id).first()
    if not s:
        raise HTTPException(status_code=404, detail="Shelter not found (or expired).")
    db.delete(s)
    db.commit()
    return {"deleted": True}


@router.get("/nearest")
def nearest_shelter(
    lat: float = Query(...),
    lng: float = Query(...),
    max_distance_km: float = Query(50, gt=0, le=200),
    db: Session = Depends(get_db),
):
    """Find the nearest shelter within the requested local search radius."""
    _cleanup_expired(db)
    all_shelters = db.query(Shelter).all()

    if not all_shelters:
        return {"shelter": None, "distance_km": None}

    best = None
    best_dist = float("inf")

    for s in all_shelters:
        d = _haversine(lat, lng, s.lat, s.lng)
        if d <= max_distance_km and d < best_dist:
            best_dist = d
            best = s

    if not best:
        return {"shelter": None, "distance_km": None}

    return {
        "shelter": ShelterOut(
            public_id=_public_id(best.shelter_id),
            type=best.type,
            name=best.name,
            lat=best.lat,
            lng=best.lng,
            capacity=best.capacity,
            resources=best.resources or [],
            is_persistent=best.is_persistent,
            created_at=best.created_at.isoformat() + "Z",
            updated_at=best.updated_at.isoformat() + "Z",
        ),
        "distance_km": round(best_dist, 2),
    }


@router.get("/safe-route")
async def safe_route(
    origin_lat: float = Query(...),
    origin_lng: float = Query(...),
    dest_lat: float = Query(...),
    dest_lng: float = Query(...),
):
    """Compute safest route from user location to a shelter using OSRM + flood scoring."""
    try:
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
            "recommendation": routes[0]["label"] if routes else "No route found",
        }
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Routing failed: {str(e)}")
