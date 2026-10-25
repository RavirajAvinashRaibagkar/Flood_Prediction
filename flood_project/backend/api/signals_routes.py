"""
Crowdsourced Emergency Signals API (privacy-safe).

- Anonymous signals only (no personal data)
- Ownership is via private `signal_id` (stored in client localStorage)
- List endpoints never expose the private `signal_id`; only a short `public_id`
- Automatic cleanup: signals expire after 30 minutes without updates
"""

from datetime import datetime, timedelta, timezone
from typing import Literal, Optional, List

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from database.connection import get_db
from database.models import EmergencySignal


router = APIRouter()

EXPIRY_MINUTES = 30


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _cleanup_expired(db: Session) -> int:
    cutoff = _utcnow() - timedelta(minutes=EXPIRY_MINUTES)
    deleted = (
        db.query(EmergencySignal)
        .filter(EmergencySignal.updated_at < cutoff)
        .delete(synchronize_session=False)
    )
    if deleted:
        db.commit()
    return int(deleted or 0)


def _public_id(signal_id: str) -> str:
    # Not reversible, but also not used for authorization.
    return signal_id[:6]


class SignalCreate(BaseModel):
    lat: float
    lng: float
    status: Literal["help", "stranded", "safe"]
    signal_id: str = Field(min_length=8, max_length=64)


class SignalUpdate(BaseModel):
    status: Literal["help", "stranded", "safe"]


class SignalOut(BaseModel):
    public_id: str
    lat: float
    lng: float
    status: Literal["help", "stranded", "safe"]
    updated_at: str


class SignalSummary(BaseModel):
    help: int
    stranded: int
    safe: int
    total: int
    expired_cleaned: int = 0


@router.get("", response_model=List[SignalOut])
def list_signals(
    bbox: Optional[str] = Query(
        None,
        description="Optional bbox: south,west,north,east (lat,lng,lat,lng).",
    ),
    db: Session = Depends(get_db),
):
    _cleanup_expired(db)

    q = db.query(EmergencySignal)
    if bbox:
        try:
            south, west, north, east = [float(x.strip()) for x in bbox.split(",")]
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid bbox. Expected 'south,west,north,east'.")
        q = q.filter(
            EmergencySignal.lat >= south,
            EmergencySignal.lat <= north,
            EmergencySignal.lng >= west,
            EmergencySignal.lng <= east,
        )

    signals = q.order_by(EmergencySignal.updated_at.desc()).all()
    return [
        SignalOut(
            public_id=_public_id(s.signal_id),
            lat=s.lat,
            lng=s.lng,
            status=s.status,
            updated_at=s.updated_at.isoformat() + "Z",
        )
        for s in signals
    ]


@router.get("/summary", response_model=SignalSummary)
def signals_summary(db: Session = Depends(get_db)):
    expired_cleaned = _cleanup_expired(db)
    all_rows = db.query(EmergencySignal.status).all()
    counts = {"help": 0, "stranded": 0, "safe": 0}
    for (status,) in all_rows:
        if status in counts:
            counts[status] += 1
    return SignalSummary(
        help=counts["help"],
        stranded=counts["stranded"],
        safe=counts["safe"],
        total=sum(counts.values()),
        expired_cleaned=expired_cleaned,
    )


@router.post("", response_model=SignalOut)
def create_signal(payload: SignalCreate, db: Session = Depends(get_db)):
    _cleanup_expired(db)

    existing = db.query(EmergencySignal).filter(EmergencySignal.signal_id == payload.signal_id).first()
    now = _utcnow()
    if existing:
        # Idempotent: update existing record owned by the same private id.
        existing.lat = payload.lat
        existing.lng = payload.lng
        existing.status = payload.status
        existing.updated_at = now
        db.commit()
        db.refresh(existing)
        s = existing
    else:
        s = EmergencySignal(
            signal_id=payload.signal_id,
            lat=payload.lat,
            lng=payload.lng,
            status=payload.status,
            created_at=now,
            updated_at=now,
        )
        db.add(s)
        db.commit()
        db.refresh(s)

    return SignalOut(
        public_id=_public_id(s.signal_id),
        lat=s.lat,
        lng=s.lng,
        status=s.status,
        updated_at=s.updated_at.isoformat() + "Z",
    )


@router.patch("/{signal_id}", response_model=SignalOut)
def update_signal(signal_id: str, payload: SignalUpdate, db: Session = Depends(get_db)):
    _cleanup_expired(db)

    s = db.query(EmergencySignal).filter(EmergencySignal.signal_id == signal_id).first()
    if not s:
        raise HTTPException(status_code=404, detail="Signal not found (or expired).")

    s.status = payload.status
    s.updated_at = _utcnow()
    db.commit()
    db.refresh(s)

    return SignalOut(
        public_id=_public_id(s.signal_id),
        lat=s.lat,
        lng=s.lng,
        status=s.status,
        updated_at=s.updated_at.isoformat() + "Z",
    )


@router.delete("/{signal_id}")
def delete_signal(signal_id: str, db: Session = Depends(get_db)):
    _cleanup_expired(db)

    s = db.query(EmergencySignal).filter(EmergencySignal.signal_id == signal_id).first()
    if not s:
        raise HTTPException(status_code=404, detail="Signal not found (or expired).")

    db.delete(s)
    db.commit()
    return {"deleted": True}

