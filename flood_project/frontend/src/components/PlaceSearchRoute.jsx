import { useEffect, useRef, useState } from 'react'

const NOMINATIM = 'https://nominatim.openstreetmap.org/search'

function shortName(r) {
  return (r.name || r.display_name || '').split(',')[0].trim() || 'Selected place'
}

export default function PlaceSearchRoute({ routing, origin }) {
  const { destination, setDestination, routes, status, error, clear, floodDataAvailable, selectedRouteId, setSelectedRouteId } = routing
  const [query, setQuery] = useState('')
  const [results, setResults] = useState([])
  const [open, setOpen] = useState(false)
  const [searching, setSearching] = useState(false)
  const [searchError, setSearchError] = useState('')
  const boxRef = useRef(null)

  // Debounced place search (Nominatim asks for ≤1 req/s — 500 ms debounce + abort of stale calls)
  useEffect(() => {
    const q = query.trim()
    if (q.length < 3 || (destination && q === destination.name)) {
      setResults([]); setSearching(false)
      return
    }
    const ctrl = new AbortController()
    const t = setTimeout(async () => {
      setSearching(true); setSearchError('')
      try {
        // Bias (not restrict) results toward the area around the user
        const d = 0.5
        const viewbox = `${origin.lng - d},${origin.lat + d},${origin.lng + d},${origin.lat - d}`
        const url = `${NOMINATIM}?q=${encodeURIComponent(q)}&format=jsonv2&limit=6&viewbox=${viewbox}`
        const res = await fetch(url, { signal: ctrl.signal })
        if (!res.ok) throw new Error(`Search failed (${res.status})`)
        const data = await res.json()
        setResults(data)
        setOpen(true)
      } catch (e) {
        if (e.name !== 'AbortError') setSearchError('Place search is unavailable right now.')
      } finally {
        if (!ctrl.signal.aborted) setSearching(false)
      }
    }, 500)
    return () => { clearTimeout(t); ctrl.abort() }
  }, [query, origin.lat, origin.lng, destination])

  // Close dropdown on outside click
  useEffect(() => {
    const onDown = (e) => { if (boxRef.current && !boxRef.current.contains(e.target)) setOpen(false) }
    document.addEventListener('mousedown', onDown)
    return () => document.removeEventListener('mousedown', onDown)
  }, [])

  const pick = (r) => {
    const name = shortName(r)
    setDestination({ lat: parseFloat(r.lat), lng: parseFloat(r.lon), name, label: r.display_name })
    setQuery(name)
    setResults([]); setOpen(false)
  }

  const onClear = () => { setQuery(''); setResults([]); clear() }

  return (
    <div className="panel p-3 rounded-lg">
      <h3 className="text-[9px] uppercase font-bold tracking-widest text-slate-500 mb-2">🔍 Find a Place &amp; Route</h3>

      <div ref={boxRef} className="relative">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onFocus={() => results.length && setOpen(true)}
          onKeyDown={(e) => { if (e.key === 'Enter' && results[0]) pick(results[0]) }}
          placeholder="Search hospital, shelter, area…"
          className="w-full text-[11px] px-2.5 py-2 pr-7 rounded-md bg-white/5 border border-white/10 text-slate-200 placeholder-slate-500 focus:outline-none focus:border-blue-500/50"
        />
        {(query || destination) && (
          <button onClick={onClear} title="Clear" className="absolute right-2 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-200 text-xs">✕</button>
        )}
        {open && results.length > 0 && (
          <ul className="absolute z-[1500] left-0 right-0 mt-1 max-h-56 overflow-y-auto rounded-md border border-white/10 bg-[#0b1224] shadow-2xl">
            {results.map((r) => (
              <li key={r.place_id}>
                <button onClick={() => pick(r)} className="w-full text-left px-2.5 py-1.5 hover:bg-white/10 border-b border-white/5 last:border-0">
                  <div className="text-[11px] text-slate-200 font-medium truncate">{shortName(r)}</div>
                  <div className="text-[9px] text-slate-500 truncate">{r.display_name}</div>
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>

      {searching && <p className="mt-1.5 text-[9px] text-slate-500">Searching…</p>}
      {searchError && <p className="mt-1.5 text-[9px] text-red-300">{searchError}</p>}

      {!destination && !searching && (
        <p className="mt-2 text-[10px] text-slate-500 leading-snug">
          Search a place to get road routes from your location, ranked by flood exposure.
        </p>
      )}

      {destination && (
        <div className="mt-2 space-y-1.5">
          <p className="text-[10px] text-slate-400 truncate">To: <span className="text-slate-200 font-semibold">{destination.name}</span></p>

          {status === 'loading' && <p className="text-[10px] text-slate-400 animate-pulse">Finding routes…</p>}
          {status === 'error' && <p className="text-[10px] text-red-300">{error}</p>}
          {status === 'ready' && routes.length === 0 && <p className="text-[10px] text-amber-300">No drivable route found.</p>}

          {status === 'ready' && !floodDataAvailable && (
            <p className="text-[9px] text-amber-300 bg-amber-500/10 border border-amber-500/20 rounded p-1.5">
              Flood-zone data couldn’t be loaded, so risk scores below are not reliable. Distance and time are still valid.
            </p>
          )}
          {status === 'ready' && routes.some(r => r.fallback) && (
            <p className="text-[9px] text-amber-300 bg-amber-500/10 border border-amber-500/20 rounded p-1.5">
              Road routing is unavailable — showing a straight line only.
            </p>
          )}

          {routes.map((route) => {
            const selected = selectedRouteId === route.id
            return (
              <div key={route.id} onClick={() => setSelectedRouteId(route.id)}
                className={`p-2 rounded-md border cursor-pointer transition-all ${selected ? 'bg-blue-500/10 border-blue-500/30' : 'bg-white/3 border-white/5 hover:border-white/10 opacity-70'}`}>
                <div className="flex items-center justify-between mb-1 gap-1">
                  <span className="text-[10px] font-bold text-slate-200 truncate">{route.name}</span>
                  <span className="flex items-center gap-1 shrink-0">
                    {route.isNearest && <span className="text-[8px] font-bold uppercase px-1.5 py-0.5 rounded-full text-cyan-300 bg-cyan-500/15 border border-cyan-500/30">Nearest</span>}
                    <span className="text-[8px] font-bold uppercase px-1.5 py-0.5 rounded-full"
                      style={{ color: route.color, background: `${route.color}20`, border: `1px solid ${route.color}40` }}>{route.badge}</span>
                  </span>
                </div>
                <div className="grid grid-cols-3 gap-1 text-[9px] font-mono">
                  <div className="text-slate-500">Dist: <span className="text-slate-300">{route.distanceKm} km</span></div>
                  <div className="text-slate-500">Risk: <span style={{ color: route.color }}>{route.risk === null ? '—' : `${route.risk}%`}</span></div>
                  <div className="text-slate-500 text-right">ETA: <span className="text-cyan-400">{route.durationMin ? `${route.durationMin} min` : '—'}</span></div>
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
