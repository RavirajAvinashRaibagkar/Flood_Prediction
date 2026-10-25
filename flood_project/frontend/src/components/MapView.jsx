import { MapContainer, TileLayer, ZoomControl, useMap, Pane } from 'react-leaflet'
import { useEffect, useState } from 'react'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import markerIcon2x from 'leaflet/dist/images/marker-icon-2x.png'
import markerIcon from 'leaflet/dist/images/marker-icon.png'
import markerShadow from 'leaflet/dist/images/marker-shadow.png'
import FloodLayer from './FloodLayer'
import SensorLayer from './SensorLayer'
import EvacuationRoutes from './EvacuationRoutes'
import PopulationExposureLayer from './PopulationExposureLayer'
import CriticalInfrastructureLayer from './CriticalInfrastructureLayer'
import FlowDirectionLayer from './FlowDirectionLayer'
import EmergencySignalsLayer from './EmergencySignalsLayer'
import ShelterLayer from './ShelterLayer'
import useEmergencySignals from '../hooks/useEmergencySignals'
import { createEmergencySignal, deleteEmergencySignal, updateEmergencySignal } from '../services/api'

// Ensure standard Leaflet marker assets resolve correctly in bundled builds
delete L.Icon.Default.prototype._getIconUrl
L.Icon.Default.mergeOptions({
  iconRetinaUrl: markerIcon2x,
  iconUrl: markerIcon,
  shadowUrl: markerShadow,
})

const TILE_LAYERS = {
  operational: {
    url: 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}',
    attribution: '&copy; <a href="https://www.esri.com/">Esri</a>, DeLorme, NAVTEQ',
    maxZoom: 16,
  },
  terrain: {
    url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attribution: '&copy; <a href="https://www.esri.com/">Esri</a>, Maxar, Earthstar Geographics',
    maxZoom: 18,
  },
  osm: {
    url: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    maxZoom: 19,
  },
}

function RegionFlyTo({ center, zoom, bbox }) {
  const map = useMap()
  useEffect(() => {
    if (bbox && bbox.length === 4) {
      map.fitBounds([[bbox[0], bbox[1]], [bbox[2], bbox[3]]], { animate: true, duration: 1.2 })
    } else if (center) {
      map.flyTo(center, zoom || 12, { duration: 1.2 })
    }
  }, [center, zoom, bbox, map])
  return null
}

// Invalidate map size when fullscreen changes
function MapResizer({ isFullscreen }) {
  const map = useMap()
  useEffect(() => {
    setTimeout(() => map.invalidateSize(), 300)
  }, [isFullscreen, map])
  return null
}

function MapCapture({ onMap }) {
  const map = useMap()
  useEffect(() => {
    onMap?.(map)
  }, [map, onMap])
  return null
}

function getOrCreateSignalId() {
  const key = 'fs_signal_id'
  const existing = localStorage.getItem(key)
  if (existing) return existing
  const id = (globalThis.crypto?.randomUUID?.() || `${Date.now().toString(16)}-${Math.random().toString(16).slice(2)}`).slice(0, 64)
  localStorage.setItem(key, id)
  return id
}

function getSignalId() {
  return localStorage.getItem('fs_signal_id')
}

function clearSignalId() {
  localStorage.removeItem('fs_signal_id')
}

export default function MapView({ layers = {}, selectedRouteId, center = [28.6139, 77.2090], zoom = 12, bbox, viewMode = 'operational', isFullscreen, toggleFullscreen, onOfferShelter }) {
  const tile = TILE_LAYERS[viewMode] || TILE_LAYERS.operational
  // Reduce overlay opacity in terrain mode
  const overlayOpacity = viewMode === 'terrain' ? 0.65 : 1.0
  const [map, setMap] = useState(null)
  const [reportOpen, setReportOpen] = useState(false)
  const [reportBusy, setReportBusy] = useState(false)
  const [geoError, setGeoError] = useState(null)

  const { signals, counts, refresh } = useEmergencySignals({
    getMapBounds: () => map?.getBounds?.(),
    pollMs: 5000,
  })

  async function getPosition() {
    setGeoError(null)
    return new Promise((resolve, reject) => {
      if (!navigator.geolocation) {
        reject(new Error('Geolocation not supported'))
        return
      }
      navigator.geolocation.getCurrentPosition(
        (pos) => resolve({ lat: pos.coords.latitude, lng: pos.coords.longitude }),
        (err) => reject(err),
        { enableHighAccuracy: true, timeout: 12000, maximumAge: 15000 }
      )
    })
  }

  async function submitStatus(status) {
    setReportBusy(true)
    try {
      const { lat, lng } = await getPosition()
      const signal_id = getSignalId() || getOrCreateSignalId()
      if (getSignalId()) {
        await updateEmergencySignal(signal_id, { status })
      } else {
        await createEmergencySignal({ lat, lng, status, signal_id })
      }
      await refresh()
      setReportOpen(false)
    } catch (e) {
      console.warn('Status submit failed:', e)
      setGeoError(e?.message || 'Unable to report status')
    } finally {
      setReportBusy(false)
    }
  }

  async function removeMySignal() {
    const signal_id = getSignalId()
    if (!signal_id) return
    setReportBusy(true)
    try {
      await deleteEmergencySignal(signal_id)
      clearSignalId()
      await refresh()
      setReportOpen(false)
    } catch (e) {
      console.warn('Remove signal failed:', e)
      setGeoError(e?.message || 'Unable to remove signal')
    } finally {
      setReportBusy(false)
    }
  }

  return (
    <div className="w-full h-full relative">
      <MapContainer center={center} zoom={zoom} zoomControl={false}
        style={{ height: '100%', width: '100%' }} className="z-0">
        <RegionFlyTo center={center} zoom={zoom} bbox={bbox} />
        <MapResizer isFullscreen={isFullscreen} />
        <MapCapture onMap={setMap} />
        <TileLayer key={viewMode} attribution={tile.attribution} url={tile.url} maxZoom={tile.maxZoom || 18} />
        {viewMode === 'operational' && (
          <TileLayer
            key="operational-labels"
            url="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}"
            maxZoom={16}
            opacity={0.85}
          />
        )}

        <Pane name="floodOverlayPane" style={{ opacity: overlayOpacity, zIndex: 450 }}>
          {layers.populationExposure && <PopulationExposureLayer />}
          {layers.floodPrediction && <FloodLayer />}
          {layers.flowDirection && <FlowDirectionLayer />}
          {layers.criticalInfrastructure && <CriticalInfrastructureLayer />}
          {layers.waterSensors && <SensorLayer sensors={[]} />}
          {layers.evacuationRoutes && <EvacuationRoutes routes={[]} selectedRouteId={selectedRouteId} />}
        </Pane>

        <EmergencySignalsLayer signals={signals} />
        <ShelterLayer />

        <ZoomControl position="bottomright" />
      </MapContainer>

      {/* Fullscreen Toggle Button */}
      <button
        onClick={toggleFullscreen}
        className="absolute top-3 right-3 z-[1000] flex items-center gap-1.5 px-2.5 py-1.5 rounded-md border transition-all text-[10px] font-medium"
        style={{
          background: isFullscreen ? 'rgba(239,68,68,0.15)' : 'rgba(15,23,42,0.85)',
          backdropFilter: 'blur(12px)',
          borderColor: isFullscreen ? 'rgba(239,68,68,0.3)' : 'rgba(255,255,255,0.1)',
          color: isFullscreen ? '#fca5a5' : '#94a3b8',
        }}
        title={isFullscreen ? 'Exit Tactical Mode' : 'Enter Tactical Mode'}
      >
        {isFullscreen ? '✕ Exit Tactical' : '⛶ Tactical Mode'}
      </button>

      {/* Fullscreen mode info badge */}
      {isFullscreen && (
        <div className="absolute top-3 left-3 z-[1000] flex items-center gap-2 px-3 py-1.5 rounded-md bg-[#0f172a]/90 backdrop-blur border border-white/10">
          <div className="w-1.5 h-1.5 rounded-full bg-red-500 animate-pulse"></div>
          <span className="text-[10px] font-bold text-slate-300 uppercase tracking-wider">Tactical Mode</span>
        </div>
      )}

      {/* Floating Emergency Signal Button */}
      <button
        onClick={() => { setGeoError(null); setReportOpen(true) }}
        className="absolute bottom-5 left-5 z-[1200] flex items-center gap-2 px-3 py-2 rounded-xl border border-white/10 bg-[#0f172a]/90 backdrop-blur text-slate-100 text-[11px] font-bold shadow-lg hover:border-white/20 transition-all"
        title="Report My Status"
      >
        <span className="text-base">📍</span>
        Report My Status
        <span className="ml-1 text-[9px] font-mono text-slate-400">{counts.help + counts.stranded + counts.safe}</span>
      </button>

      {/* Floating Offer Shelter Button */}
      <button
        onClick={onOfferShelter}
        className="absolute bottom-5 right-5 z-[1200] flex items-center gap-2 px-3 py-2 rounded-xl border border-green-500/20 bg-[#0f172a]/90 backdrop-blur text-green-200 text-[11px] font-bold shadow-lg hover:border-green-500/35 hover:bg-green-500/10 transition-all"
        title="Offer Shelter"
      >
        <span className="text-base">🏠</span>
        Offer Shelter
      </button>

      {/* Report Modal */}
      {reportOpen && (
        <div className="absolute inset-0 z-[1300] flex items-end sm:items-center justify-center p-4" style={{ background: 'rgba(0,0,0,0.55)' }}>
          <div className="w-full max-w-md rounded-2xl border border-white/10 bg-[#0b1224] shadow-2xl overflow-hidden">
            <div className="p-4 border-b border-white/5 flex items-center justify-between">
              <div>
                <div className="text-[11px] font-extrabold text-slate-100 tracking-wide">Crowdsourced Emergency Signals</div>
                <div className="text-[9px] text-slate-400">Anonymous • No personal data • Auto-expires in 30 minutes</div>
              </div>
              <button onClick={() => setReportOpen(false)} className="text-slate-400 hover:text-slate-200 text-sm">✕</button>
            </div>

            <div className="p-4 space-y-2.5">
              <button
                disabled={reportBusy}
                onClick={() => submitStatus('help')}
                className="w-full flex items-center justify-between px-3 py-2 rounded-xl border border-red-500/25 bg-red-500/10 hover:bg-red-500/15 transition-colors text-left"
              >
                <span className="text-[12px] font-bold text-red-200">🔴 Need Help</span>
                <span className="text-[10px] text-red-300/80 font-mono">{reportBusy ? 'SENDING…' : 'TAP'}</span>
              </button>

              <button
                disabled={reportBusy}
                onClick={() => submitStatus('stranded')}
                className="w-full flex items-center justify-between px-3 py-2 rounded-xl border border-amber-500/25 bg-amber-500/10 hover:bg-amber-500/15 transition-colors text-left"
              >
                <span className="text-[12px] font-bold text-amber-200">🟡 Waiting / Stranded</span>
                <span className="text-[10px] text-amber-300/80 font-mono">{reportBusy ? 'SENDING…' : 'TAP'}</span>
              </button>

              <button
                disabled={reportBusy}
                onClick={() => submitStatus('safe')}
                className="w-full flex items-center justify-between px-3 py-2 rounded-xl border border-green-500/25 bg-green-500/10 hover:bg-green-500/15 transition-colors text-left"
              >
                <span className="text-[12px] font-bold text-green-200">🟢 Safe</span>
                <span className="text-[10px] text-green-300/80 font-mono">{reportBusy ? 'SENDING…' : 'TAP'}</span>
              </button>

              {geoError && (
                <div className="mt-2 p-2 rounded-lg border border-red-500/20 bg-red-500/10 text-[10px] text-red-200">
                  {geoError}
                </div>
              )}

              <div className="pt-2 mt-2 border-t border-white/5 flex items-center justify-between">
                <div className="text-[9px] text-slate-500">
                  {getSignalId() ? 'You have an active signal (update anytime).' : 'No active signal yet.'}
                </div>
                <div className="flex items-center gap-2">
                  {getSignalId() && (
                    <button
                      disabled={reportBusy}
                      onClick={removeMySignal}
                      className="text-[10px] font-bold px-2.5 py-1 rounded-md border border-white/10 bg-white/5 hover:bg-white/10 text-slate-200"
                    >
                      Remove My Signal
                    </button>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
