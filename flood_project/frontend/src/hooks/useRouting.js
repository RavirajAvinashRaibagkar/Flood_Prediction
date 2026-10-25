import { useEffect, useMemo, useState } from 'react'
import { getEvacuationRoute } from '../services/api'

export function riskBucket(risk) {
  if (risk === null || risk === undefined || Number.isNaN(risk)) return { badge: 'UNKNOWN', color: '#64748b' }
  if (risk <= 33) return { badge: 'SAFE', color: '#22c55e' }
  if (risk <= 66) return { badge: 'MODERATE', color: '#eab308' }
  return { badge: 'DANGEROUS', color: '#ef4444' }
}

/**
 * Single source of truth for "route from where I am to the place I searched".
 * One OSRM request per (origin, destination) — replaces the two duplicate fetches
 * that RightCommandPanel and EvacuationRoutes used to make to a hard-coded fake point.
 */
export default function useRouting(origin) {
  const [destination, setDestination] = useState(null) // { lat, lng, name }
  const [raw, setRaw] = useState({ routes: [], floodDataAvailable: true })
  const [status, setStatus] = useState('idle') // idle | loading | ready | error
  const [error, setError] = useState('')
  const [selectedRouteId, setSelectedRouteId] = useState(null)

  useEffect(() => {
    if (!destination) return
    let cancelled = false
    ;(async () => {
      setStatus('loading')
      setError('')
      try {
        const res = await getEvacuationRoute(origin.lat, origin.lng, destination.lat, destination.lng)
        if (cancelled) return
        setRaw({
          routes: res.data?.routes || [],
          floodDataAvailable: res.data?.flood_data_available !== false,
        })
        setSelectedRouteId(res.data?.routes?.[0]?.id ?? null)
        setStatus('ready')
      } catch (err) {
        if (cancelled) return
        setRaw({ routes: [], floodDataAvailable: true })
        setError(err?.response?.data?.detail || err?.message || 'Routing failed')
        setStatus('error')
      }
    })()
    return () => { cancelled = true }
  }, [origin.lat, origin.lng, destination])

  const routes = useMemo(() => {
    if (!destination) return []
    const list = raw.routes
    const nearestId = list.length ? list.reduce((a, b) => (b.distance_km < a.distance_km ? b : a)).id : null
    return list.map((r, i) => {
      const { badge, color } = r.fallback ? riskBucket(null) : riskBucket(r.risk_score)
      return {
        id: r.id,
        name: r.label || `Route ${i + 1}`,
        distanceKm: r.distance_km,
        durationMin: r.duration_min,
        risk: r.fallback ? null : r.risk_score,
        badge, color,
        isSafest: i === 0 && !r.fallback,
        isNearest: r.id === nearestId && list.length > 1,
        fallback: !!r.fallback,
        path: r.coordinates || [],
      }
    })
  }, [raw, destination])

  const clear = () => {
    setDestination(null)
    setRaw({ routes: [], floodDataAvailable: true })
    setStatus('idle')
    setError('')
    setSelectedRouteId(null)
  }

  return {
    destination, setDestination, routes, status, error, clear,
    floodDataAvailable: raw.floodDataAvailable,
    selectedRouteId, setSelectedRouteId,
  }
}
