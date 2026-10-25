import { useState, useMemo, useRef, useEffect } from 'react'
import { useRegion } from '../context/RegionContext'
import { useNavigate } from 'react-router-dom'

function debounce(fn, ms) {
  let timer
  return (...args) => { clearTimeout(timer); timer = setTimeout(() => fn(...args), ms) }
}

export default function TopNavBar({ riskScore, showAnalytics, setShowAnalytics, onOfferShelter }) {
  const { region, updateRegion, locateMe, locationStatus, locationError } = useRegion()
  const navigate = useNavigate()
  const [query, setQuery] = useState('')
  const [suggestions, setSuggestions] = useState([])
  const [showDropdown, setShowDropdown] = useState(false)
  const inputRef = useRef(null)
  const dropdownRef = useRef(null)

  useEffect(() => {
    const handler = (e) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target) &&
          inputRef.current && !inputRef.current.contains(e.target)) {
        setShowDropdown(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const searchNominatim = useMemo(() => debounce(async (q) => {
    if (q.length < 3) { setSuggestions([]); return }
    try {
      const res = await fetch(
        `https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(q)}&format=json&limit=5`,
        { headers: { 'User-Agent': 'FloodSentinel/1.0' } }
      )
      setSuggestions(await res.json())
      setShowDropdown(true)
    } catch (err) { console.warn('Search failed:', err) }
  }, 400), [])

  const handleSelect = (result) => {
    const lat = parseFloat(result.lat), lng = parseFloat(result.lon)
    const bbox = result.boundingbox
      ? result.boundingbox.map(Number) : [lat-0.15, lat+0.15, lng-0.25, lng+0.25]
    updateRegion({ lat, lng, regionName: result.display_name.split(',')[0], bbox: [bbox[0], bbox[2], bbox[1], bbox[3]], zoom: 12 })
    setQuery(result.display_name.split(',').slice(0, 2).join(','))
    setShowDropdown(false)
    setSuggestions([])
  }

  const level = riskScore?.level || 'LOADING'
  const riskColor = level === 'HIGH' ? '#ef4444' : level === 'LOW' ? '#22c55e' : '#f59e0b'

  return (
    <header className="top-navbar h-12 bg-[#080c18] border-b border-white/5 flex items-center justify-between px-4 shrink-0 z-50">
      {/* Logo */}
      <div className="flex items-center gap-3 flex-shrink-0">
        <div className="w-7 h-7 rounded-md bg-gradient-to-br from-blue-500 to-cyan-400 flex items-center justify-center text-xs font-bold text-white shadow-md shadow-blue-500/20">FS</div>
        <div className="hidden md:block">
          <h1 className="text-xs font-bold text-slate-100 tracking-wide leading-tight">FloodSentinel AI</h1>
          <p className="text-[8px] text-slate-500 uppercase tracking-widest">Command Center</p>
        </div>
      </div>

      {/* Search */}
      <div className="flex-1 flex justify-center mx-4 max-w-md">
        <div className="relative w-full">
          <input ref={inputRef} type="text" value={query}
            onChange={(e) => { setQuery(e.target.value); searchNominatim(e.target.value) }}
            onFocus={() => suggestions.length > 0 && setShowDropdown(true)}
            placeholder={`Search location (${region.regionName})...`}
            className="w-full bg-white/5 border border-white/8 rounded-md px-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-blue-500/50 placeholder:text-slate-500" />
          <span className="absolute right-2.5 top-1/2 -translate-y-1/2 text-slate-500 text-xs">🔍</span>
          {showDropdown && suggestions.length > 0 && (
            <div ref={dropdownRef} className="absolute top-full left-0 right-0 mt-1 bg-[#1e293b] border border-white/10 rounded-lg shadow-2xl z-[9999] max-h-60 overflow-y-auto">
              {suggestions.map((s, i) => (
                <button key={i} onClick={() => handleSelect(s)} className="w-full text-left px-3 py-2 text-xs text-slate-300 hover:bg-blue-500/20 hover:text-white transition-colors border-b border-white/5 last:border-0">
                  <div className="font-medium">{s.display_name.split(',').slice(0, 2).join(',')}</div>
                  <div className="text-[10px] text-slate-500 mt-0.5 truncate">{s.display_name}</div>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Right: Analytics button + Status Chips */}
      <div className="flex items-center gap-2 flex-shrink-0">
        <div className="relative">
          <button
            onClick={locateMe}
            disabled={locationStatus === 'locating'}
            title={locationError || 'Use your current location'}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-md border border-cyan-500/25 bg-cyan-500/10 text-[9px] uppercase font-bold tracking-wider text-cyan-300 hover:bg-cyan-500/20 disabled:cursor-wait disabled:opacity-70"
          >
            <span>{locationStatus === 'locating' ? '⌛' : '📍'}</span>
            {locationStatus === 'locating' ? 'Locating…' : 'My Location'}
          </button>
          {locationError && (
            <p className="absolute right-0 top-full mt-1 w-56 rounded-md border border-amber-500/30 bg-[#111827] p-2 text-[10px] normal-case tracking-normal text-amber-200 shadow-xl z-[9999]">
              {locationError}
            </p>
          )}
        </div>
        <button onClick={() => navigate('/models')} className="px-2.5 py-1 rounded-md border border-white/8 bg-white/5 text-[9px] uppercase font-bold tracking-wider text-slate-400 hover:text-cyan-300">
          AI Models
        </button>
        {/* Analytics Toggle Button (Item 5) */}
        <button
          onClick={() => setShowAnalytics(!showAnalytics)}
          className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md border text-[9px] uppercase font-bold tracking-wider transition-all ${
            showAnalytics
              ? 'bg-blue-500/20 border-blue-500/40 text-blue-400'
              : 'bg-white/5 border-white/8 text-slate-400 hover:border-white/15 hover:text-slate-300'
          }`}
        >
          <span>📊</span> Analytics
        </button>

        {/* Offer Shelter Button */}
        <button
          onClick={onOfferShelter}
          className="flex items-center gap-1.5 px-2.5 py-1 rounded-md border text-[9px] uppercase font-bold tracking-wider transition-all bg-green-500/12 border-green-500/25 text-green-400 hover:bg-green-500/20 hover:border-green-500/40"
        >
          <span>🏠</span> Offer Shelter
        </button>

        <StatusChip label="REGION" value={region.regionName} color="#06b6d4" />
        {region.locationSource === 'device' && <StatusChip label="GPS" value={`±${region.locationAccuracyM || '?'}m`} color="#38bdf8" />}
        <StatusChip label="RISK" value={level} color={riskColor} pulse={level === 'HIGH'} />
        <StatusChip label="DATA" value="● LIVE" color="#22c55e" />
      </div>
    </header>
  )
}

function StatusChip({ label, value, color, pulse }) {
  return (
    <div className={`hidden lg:flex items-center gap-1.5 px-2 py-1 rounded-md border text-[9px] uppercase font-bold tracking-wider ${pulse ? 'risk-pulse' : ''}`}
         style={{ borderColor: `${color}30`, background: `${color}08` }}>
      <span className="text-slate-500">{label}:</span>
      <span style={{ color }}>{value}</span>
      {pulse && <div className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ backgroundColor: color }}></div>}
    </div>
  )
}
