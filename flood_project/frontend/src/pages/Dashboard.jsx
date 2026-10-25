import { useMemo, useState, useCallback, useEffect } from 'react'
import { useRegion } from '../context/RegionContext'
import TopNavBar from '../components/TopNavBar'
import LeftPanel from '../components/LeftPanel'
import RightCommandPanel from '../components/RightCommandPanel'
import BottomDataStream from '../components/BottomDataStream'
import MapView from '../components/MapView'
import AnalyticsPanel from '../components/AnalyticsPanel'
import ShelterRegistrationModal from '../components/ShelterRegistrationModal'
import useFloodData from '../hooks/useFloodData'
import useRouting from '../hooks/useRouting'

export default function Dashboard() {
  const { region } = useRegion()

  const [layers, setLayers] = useState({
    floodPrediction: true,
    waterSensors: true,
    evacuationRoutes: true,
    traffic: false,
    populationExposure: false,
    criticalInfrastructure: false,
    flowDirection: false
  })

  const routing = useRouting(region)
  const [viewMode, setViewMode] = useState('operational')
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [showAnalytics, setShowAnalytics] = useState(false)
  const [showShelterModal, setShowShelterModal] = useState(false)

  const {
    riskScore,
    rainfallData,
    waterLevelData,
    alerts,
    sourceStatus,
    dischargeMeta,
    rainfallLastFetched,
  } = useFloodData(region.lat, region.lng, 30000)

  // Automated alert status (derived; no manual controls)
  const alertStatus = useMemo(() => {
    const score = Number.isFinite(riskScore?.score) ? riskScore.score : null
    const level = riskScore?.level || 'LOW'
    const raw = riskScore?.raw || {}
    const trend = raw.trend || 'stable'

    const publicAlert = score === null ? 'UNAVAILABLE' : score > 70 && (trend === 'rising' || level === 'HIGH')
      ? 'RECOMMENDED'
      : 'NO ACTION'

    const authorities = score === null ? 'UNAVAILABLE' : score > 85 ? 'RECOMMENDED' : 'NO ACTION'

    const triggerReason = publicAlert === 'RECOMMENDED'
      ? `High rainfall + ${trend === 'rising' ? 'river surge' : 'elevated risk'} (Score: ${score}%)`
      : authorities === 'RECOMMENDED'
        ? `Critical risk level (Score: ${score}%)`
        : null

    return {
      publicAlert,
      authorities,
      lastTriggerTime: null,
      triggerReason,
    }
  }, [riskScore])

  // Body scroll lock in fullscreen
  useEffect(() => {
    if (isFullscreen) {
      document.body.classList.add('fullscreen-lock')
    } else {
      document.body.classList.remove('fullscreen-lock')
    }
    return () => document.body.classList.remove('fullscreen-lock')
  }, [isFullscreen])

  const toggleLayer = useCallback((layer) => {
    setLayers(prev => ({ ...prev, [layer]: !prev[layer] }))
  }, [])

  return (
    <div className={`h-screen w-screen flex flex-col ${isFullscreen ? 'fullscreen-mode' : ''}`} style={{ background: '#0a0e1a' }}>
      {/* Top Nav — 48px */}
      <TopNavBar
        riskScore={riskScore}
        showAnalytics={showAnalytics}
        setShowAnalytics={setShowAnalytics}
        onOfferShelter={() => setShowShelterModal(true)}
        className="top-navbar"
      />

      {/* Main Content */}
      <div className="main-content flex-1 flex overflow-hidden relative">
        {/* LEFT PANEL — 240px */}
        <div className="left-panel relative z-10 flex-shrink-0" style={{ width: 240 }}>
          <div className="h-full overflow-y-auto p-3 space-y-3" style={{ background: 'rgba(10,14,26,0.6)' }}>
            <LeftPanel
              layers={layers}
              toggleLayer={toggleLayer}
              viewMode={viewMode}
              setViewMode={setViewMode}
            />
          </div>
        </div>

        {/* MAP — center fills remaining */}
        <div className="map-wrapper flex-1 relative z-0">
          <MapView
            layers={layers}
            routing={routing}
            center={[region.lat, region.lng]}
            zoom={region.zoom || 12}
            bbox={region.bbox}
            viewMode={viewMode}
            isFullscreen={isFullscreen}
            toggleFullscreen={() => setIsFullscreen(prev => !prev)}
            onOfferShelter={() => setShowShelterModal(true)}
          />
          {/* Floating Analytics Panel (Item 5) */}
          {showAnalytics && (
            <AnalyticsPanel
              rainfallData={rainfallData}
              waterLevelData={waterLevelData}
              sourceStatus={sourceStatus}
              dischargeMeta={dischargeMeta}
              rainfallLastFetched={rainfallLastFetched}
              onClose={() => setShowAnalytics(false)}
            />
          )}
        </div>

        {/* RIGHT COMMAND PANEL — 320px */}
        <div className="right-panel relative z-10 flex-shrink-0" style={{ width: 320 }}>
          <div className="h-full overflow-y-auto p-3" style={{ background: 'rgba(10,14,26,0.6)' }}>
            <RightCommandPanel
              riskScore={riskScore}
              alertStatus={alertStatus}
              routing={routing}
              origin={region}
            />
          </div>
        </div>
      </div>

      {/* Bottom Alert Ticker — 36px */}
      <BottomDataStream liveAlerts={alerts} className="bottom-ticker" />

      {/* Shelter Registration Modal */}
      <ShelterRegistrationModal
        open={showShelterModal}
        onClose={() => setShowShelterModal(false)}
      />
    </div>
  )
}
