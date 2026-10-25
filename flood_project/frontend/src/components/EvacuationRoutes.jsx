import { useEffect } from 'react'
import { Polyline, Marker, Popup, CircleMarker, useMap } from 'react-leaflet'
import L from 'leaflet'
import { useRegion } from '../context/RegionContext'

const destIcon = L.divIcon({
  className: '',
  html: `<div style="width:30px;height:30px;border-radius:50% 50% 50% 0;transform:rotate(-45deg);
    background:linear-gradient(135deg,#3b82f6,#1d4ed8);border:2px solid rgba(255,255,255,0.85);
    box-shadow:0 0 14px rgba(59,130,246,0.6);display:flex;align-items:center;justify-content:center;">
    <span style="transform:rotate(45deg);font-size:14px;">📍</span></div>`,
  iconSize: [30, 30],
  iconAnchor: [15, 30],
})

// Fit the map to the routes whenever a new destination/route set arrives
function FitToRoutes({ routes, destinationKey }) {
  const map = useMap()
  useEffect(() => {
    const pts = routes.flatMap(r => r.path)
    if (pts.length > 1) map.fitBounds(L.latLngBounds(pts), { padding: [60, 60], animate: true })
  }, [destinationKey, routes.length, map]) // eslint-disable-line react-hooks/exhaustive-deps
  return null
}

export default function EvacuationRoutes({ routing }) {
  const { region } = useRegion()
  const { routes, destination, selectedRouteId, setSelectedRouteId } = routing
  if (!destination) return null

  const drawable = routes.filter(r => r.path.length >= 2)
  const selected = drawable.find(r => r.id === selectedRouteId) || drawable[0]

  return (
    <>
      <FitToRoutes routes={drawable} destinationKey={`${destination.lat},${destination.lng},${region.lat.toFixed(3)},${region.lng.toFixed(3)}`} />

      <CircleMarker center={[region.lat, region.lng]} radius={7}
        pathOptions={{ color: '#fff', weight: 2, fillColor: '#3b82f6', fillOpacity: 1 }} />

      {/* Unselected routes first so the selected one is drawn on top */}
      {[...drawable].sort((a, b) => (a.id === selected?.id) - (b.id === selected?.id)).map((route) => {
        const isSelected = route.id === selected?.id
        return (
          <div key={route.id}>
            {isSelected && (
              <Polyline positions={route.path}
                pathOptions={{ color: route.color, weight: 12, opacity: 0.3, lineCap: 'round', lineJoin: 'round', className: 'route-glow-effect' }} />
            )}
            <Polyline positions={route.path}
              eventHandlers={{ click: () => setSelectedRouteId(route.id) }}
              pathOptions={{
                color: route.color,
                weight: isSelected ? 6 : 4,
                opacity: isSelected ? 1 : 0.45,
                dashArray: route.fallback ? '4, 8' : (isSelected ? '' : '8, 8'),
                lineCap: 'round', lineJoin: 'round',
                className: isSelected && !route.fallback ? 'route-path-animated' : '',
              }}>
              <Popup className="route-popup">
                <div className="font-sans text-xs text-slate-800 space-y-0.5">
                  <div className="font-bold">{route.name}{route.isNearest ? ' · Nearest' : ''}</div>
                  <div>{route.distanceKm} km{route.durationMin ? ` · ${route.durationMin} min` : ''}</div>
                  <div>Flood risk: <b style={{ color: route.color }}>{route.risk === null ? 'unknown' : `${route.risk}% (${route.badge})`}</b></div>
                </div>
              </Popup>
            </Polyline>
          </div>
        )
      })}

      <Marker position={[destination.lat, destination.lng]} icon={destIcon}>
        <Popup><div className="font-bold text-slate-800 font-sans">📍 {destination.name}</div></Popup>
      </Marker>
    </>
  )
}
