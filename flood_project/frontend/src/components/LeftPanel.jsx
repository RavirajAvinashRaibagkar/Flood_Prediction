import { useMemo, useState } from 'react'
import { useRegion } from '../context/RegionContext'

function debounce(fn, ms) {
  let timer
  return (...args) => { clearTimeout(timer); timer = setTimeout(() => fn(...args), ms) }
}

export default function LeftPanel({ layers, toggleLayer, viewMode, setViewMode }) {
  const { region, updateRegion, locateMe, locationStatus, locationError } = useRegion()
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState([])
  const [showResults, setShowResults] = useState(false)

  // Only show non-satellite overlays (Item 4: removed Satellite Layer)
  const overlays = [
    { id: 'floodPrediction', label: 'Flood Prediction', icon: '🌊', color: '#ef4444' },
    { id: 'populationExposure', label: 'Population Exposure', icon: '👥', color: '#a855f7' },
    { id: 'criticalInfrastructure', label: 'Critical Infra', icon: '🏥', color: '#f59e0b' },
    { id: 'flowDirection', label: 'Flow Direction', icon: '↘️', color: '#38bdf8' },
    { id: 'waterSensors', label: 'Water Sensors', icon: '📡', color: '#3b82f6' },
    { id: 'evacuationRoutes', label: 'Evacuation Routes', icon: '🚨', color: '#22c55e' },
    { id: 'traffic', label: 'Traffic Density', icon: '🚗', color: '#eab308' },
  ]

  const searchLocation = useMemo(() => debounce(async (q) => {
    if (q.length < 3) { setSearchResults([]); return }
    try {
      const res = await fetch(`https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(q)}&format=json&limit=5`, { headers: { 'User-Agent': 'FloodSentinel/1.0' } })
      const results = await res.json()
      setSearchResults(results)
      setShowResults(results.length > 0)
    } catch (err) { console.warn('Search failed:', err) }
  }, 400), [])

  const handleSelectLocation = (result) => {
    const lat = parseFloat(result.lat), lng = parseFloat(result.lon)
    const bbox = result.boundingbox ? result.boundingbox.map(Number) : [lat-0.15, lat+0.15, lng-0.25, lng+0.25]
    updateRegion({ lat, lng, regionName: result.display_name.split(',')[0], bbox: [bbox[0], bbox[2], bbox[1], bbox[3]], zoom: 12 })
    setSearchQuery(result.display_name.split(',')[0])
    setShowResults(false)
  }

  return (
    <div className="h-full flex flex-col gap-2.5">
      {/* SECTION 1: Region Focus */}
      <div className="panel p-3 rounded-lg">
        <h3 className="text-[9px] uppercase font-bold tracking-widest text-slate-500 mb-2">Region Focus</h3>
        <div className="relative">
          <input type="text" value={searchQuery}
            onChange={(e) => { setSearchQuery(e.target.value); searchLocation(e.target.value) }}
            onFocus={() => searchResults.length > 0 && setShowResults(true)}
            placeholder={region.regionName}
            className="w-full bg-white/5 border border-white/8 rounded-md px-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-blue-500/50 pr-8" />
          <button onClick={locateMe} disabled={locationStatus === 'locating'} title="Use my location" className="absolute right-2 top-1/2 -translate-y-1/2 text-sm hover:scale-110 transition-transform disabled:opacity-50">{locationStatus === 'locating' ? '⌛' : '📍'}</button>
          {showResults && searchResults.length > 0 && (
            <div className="absolute top-full left-0 right-0 mt-1 bg-[#0f172a] border border-white/10 rounded-lg shadow-xl z-[9999] max-h-48 overflow-y-auto">
              {searchResults.map((r, i) => (
                <button key={i} onClick={() => handleSelectLocation(r)} className="w-full text-left px-3 py-2 text-xs text-slate-300 hover:bg-blue-500/20 hover:text-white transition-colors border-b border-white/5 last:border-0">
                  <div className="font-medium truncate">{r.display_name.split(',').slice(0, 2).join(',')}</div>
                </button>
              ))}
            </div>
          )}
        </div>
        <div className="mt-2 text-[9px] leading-relaxed">
          {locationStatus === 'locating' && <span className="text-cyan-300">Finding your current location…</span>}
          {locationError && <span className="text-amber-300">{locationError}</span>}
          {region.locationSource === 'device' && locationStatus === 'ready' && (
            <span className="text-slate-400">GPS: {region.lat.toFixed(4)}, {region.lng.toFixed(4)} · ±{region.locationAccuracyM || '?'} m</span>
          )}
        </div>
      </div>

      {/* SECTION 2: View Mode (Item 4 — replaces satellite toggle) */}
      <div className="panel p-3 rounded-lg">
        <h3 className="text-[9px] uppercase font-bold tracking-widest text-slate-500 mb-2">View Mode</h3>
        <div className="space-y-1">
          {[
            { value: 'operational', label: 'Operational Mode', desc: 'Dark map, high-contrast overlays', icon: '⬛' },
            { value: 'terrain', label: 'Terrain Analysis', desc: 'Satellite imagery, terrain focus', icon: '🛰' },
          ].map(opt => (
            <label key={opt.value} className="flex items-center gap-2 cursor-pointer py-1 group">
              <div className={`w-3.5 h-3.5 rounded-full border-2 flex items-center justify-center transition-colors ${
                viewMode === opt.value ? 'border-blue-400' : 'border-slate-600 group-hover:border-slate-400'
              }`}>
                {viewMode === opt.value && <div className="w-1.5 h-1.5 rounded-full bg-blue-400"></div>}
              </div>
              <input type="radio" name="viewMode" className="hidden" checked={viewMode === opt.value} onChange={() => setViewMode(opt.value)} />
              <div>
                <span className={`text-[11px] font-medium ${viewMode === opt.value ? 'text-slate-100' : 'text-slate-400'}`}>{opt.icon} {opt.label}</span>
                <p className="text-[8px] text-slate-500 leading-tight">{opt.desc}</p>
              </div>
            </label>
          ))}
        </div>
      </div>

      {/* SECTION 3: Passive system state (no controls) */}
      <div className="panel p-3 rounded-lg">
        <h3 className="text-[9px] uppercase font-bold tracking-widest text-slate-500 mb-2">System</h3>
        <div className="p-2 rounded bg-cyan-500/10 border border-cyan-500/20 text-[10px] text-cyan-300 flex items-center gap-2">
          <div className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse"></div>
          🧠 Auto Monitoring Active
        </div>
      </div>

      {/* SECTION 4: Map Overlays */}
      <div className="panel p-3 rounded-lg flex-1 overflow-y-auto">
        <h3 className="text-[9px] uppercase font-bold tracking-widest text-slate-500 mb-2">Map Overlays</h3>
        <div className="space-y-0.5">
          {overlays.map(layer => (
            <label key={layer.id} className="flex items-center gap-2 cursor-pointer group py-0.5">
              <div className={`w-3.5 h-3.5 rounded border flex items-center justify-center transition-colors ${
                layers[layer.id] ? 'bg-blue-500 border-blue-500' : 'border-slate-600 group-hover:border-slate-400'
              }`}>
                {layers[layer.id] && <svg className="w-2.5 h-2.5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7"/></svg>}
              </div>
              <input type="checkbox" className="hidden" checked={layers[layer.id]} onChange={() => toggleLayer(layer.id)} />
              <span className="text-xs opacity-80">{layer.icon}</span>
              <span className={`text-[11px] font-medium transition-colors ${layers[layer.id] ? 'text-slate-100' : 'text-slate-400'}`}>{layer.label}</span>
              {layers[layer.id] && <div className="ml-auto w-1.5 h-1.5 rounded-full" style={{ backgroundColor: layer.color, boxShadow: `0 0 6px ${layer.color}` }}></div>}
            </label>
          ))}
        </div>
      </div>
    </div>
  )
}
