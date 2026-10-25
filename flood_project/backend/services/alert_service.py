"""
Alert Service — rule-based alert engine using real weather and hydro data.
Replaces mock alert strings with actual threshold-based triggers.
"""

import hashlib
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# ─── Deduplication ──────────────────────────────────────────
_active_alerts = {}  # hash → triggered_at


def evaluate_alerts(region_data: dict) -> list:
    """
    Run rule-based alert evaluation against real sensor/weather data.
    Returns list of new alerts (deduped — won't fire same alert within 30 min).
    """
    raw_alerts = _run_rules(region_data)
    new_alerts = []

    for alert in raw_alerts:
        alert_hash = hashlib.md5(alert["message"].encode()).hexdigest()
        if alert_hash not in _active_alerts:
            _active_alerts[alert_hash] = datetime.utcnow()
            alert["id"] = int(alert_hash[:8], 16) % 100000
            alert["timestamp"] = datetime.utcnow().isoformat()
            alert["lat"] = region_data.get("lat", 0)
            alert["lng"] = region_data.get("lng", 0)
            new_alerts.append(alert)

    # Auto-clear alerts older than 30 minutes
    now = datetime.utcnow()
    stale = [k for k, v in _active_alerts.items() if (now - v).total_seconds() > 1800]
    for k in stale:
        del _active_alerts[k]

    return new_alerts


def _run_rules(data: dict) -> list:
    """Apply all alert rules against the current data."""
    alerts = []

    # ─── Rule 1: Rainfall spike ─────────────────────────────
    rainfall_1h = data.get("rainfall_1h", 0)
    if rainfall_1h > 20:
        alerts.append({
            "severity": "high",
            "message": f"Heavy rainfall: {rainfall_1h:.1f} mm/hr detected",
        })
    elif rainfall_1h > 10:
        alerts.append({
            "severity": "medium",
            "message": f"Moderate rainfall: {rainfall_1h:.1f} mm/hr",
        })

    # ─── Rule 2: River discharge surge ──────────────────────
    surge_ratio = data.get("surge_ratio")
    current_discharge = data.get("current_discharge")
    if surge_ratio is not None and current_discharge is not None:
        if surge_ratio > 1.5:
            alerts.append({
                "severity": "critical",
                "message": f"River discharge {(surge_ratio-1)*100:.0f}% above weekly average ({current_discharge:.1f} m³/s)",
            })
        elif surge_ratio > 1.2:
            alerts.append({
                "severity": "high",
                "message": f"River discharge elevated: {surge_ratio:.1f}x weekly average",
            })

    # ─── Rule 3: Discharge exceeds danger threshold ─────────
    danger_threshold = data.get("danger_threshold_discharge") or data.get("danger_threshold")
    if current_discharge is not None and danger_threshold is not None:
        if current_discharge > danger_threshold:
            alerts.append({
                "severity": "critical",
                "message": f"DANGER: River discharge ({current_discharge:.1f} m³/s) exceeds threshold ({danger_threshold:.1f} m³/s)",
            })

    # ─── Rule 4: Soil moisture saturation ───────────────────
    soil_moisture = data.get("soil_moisture", 0)
    if soil_moisture and soil_moisture > 0.45:
        alerts.append({
            "severity": "high",
            "message": f"Soil moisture at saturation: {soil_moisture:.2f} m³/m³",
        })

    # ─── Rule 5: High precipitation probability ─────────────
    precip_prob = data.get("precipitation_probability", 0)
    if precip_prob and precip_prob > 80:
        alerts.append({
            "severity": "medium",
            "message": f"Precipitation probability: {precip_prob}% in next hour",
        })

    # ─── Rule 6: Rising trend ───────────────────────────────
    trend = data.get("trend", "stable")
    if trend == "rising" and surge_ratio is not None and surge_ratio > 1.1:
        alerts.append({
            "severity": "medium",
            "message": "River discharge trend: RISING — monitor closely",
        })

    return alerts
