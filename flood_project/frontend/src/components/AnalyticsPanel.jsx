import { useMemo } from 'react'
import RainfallChart from './RainfallChart'
import WaterLevelChart from './WaterLevelChart'

export default function AnalyticsPanel({
  rainfallData = [],
  waterLevelData = [],
  sourceStatus = { rainfall: 'ok', discharge: 'ok' },
  dischargeMeta = null,
  rainfallLastFetched = null,
  onClose,
}) {
  // Compute summary stats
  const rainfallSummary = useMemo(() => {
    const status = sourceStatus?.rainfall
    if (status === 'error' || status === 'empty' || !rainfallData.length) {
      return { current: 'Data unavailable', trend: 'unknown' }
    }
    const last = rainfallData[rainfallData.length - 1]
    const current = last?.rainfall
    if (current == null) {
      return { current: 'Data unavailable', trend: 'unknown' }
    }
    const prev = rainfallData.length > 3 ? (rainfallData[rainfallData.length - 4]?.rainfall || 0) : current
    const trend = current > prev + 0.5 ? 'rising' : current < prev - 0.5 ? 'falling' : 'stable'
    return { current: current.toFixed(1), trend }
  }, [rainfallData, sourceStatus])

  const waterSummary = useMemo(() => {
    const status = sourceStatus?.discharge
    if (status === 'error' || status === 'empty' || !waterLevelData.length) {
      return { current: 'Data unavailable', danger: null, status: 'unavailable' }
    }
    const last = waterLevelData[waterLevelData.length - 1]
    const current = last?.level
    if (current == null) {
      return { current: 'Data unavailable', danger: null, status: 'unavailable' }
    }
    // Status must compare discharge to danger_threshold_discharge only
    const danger = last?.danger ?? dischargeMeta?.dangerThresholdDischarge
    let levelStatus = 'safe'
    if (danger != null && danger > 0) {
      levelStatus = current > danger ? 'danger' : current > danger * 0.8 ? 'warning' : 'safe'
    } else {
      levelStatus = 'safe'
    }
    return { current: current.toFixed(1), danger, status: levelStatus }
  }, [waterLevelData, sourceStatus, dischargeMeta])

  const trendIcon = { rising: '↑', falling: '↓', stable: '→', unknown: '—' }
  const trendColor = { rising: '#ef4444', falling: '#22c55e', stable: '#f59e0b', unknown: '#64748b' }
  const statusColor = { safe: '#22c55e', warning: '#f59e0b', danger: '#ef4444', unavailable: '#64748b' }

  return (
    <div className="analytics-float p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-xs font-bold text-slate-200 uppercase tracking-wider">📊 Analytics Panel</h3>
        <button onClick={onClose} className="w-5 h-5 rounded flex items-center justify-center text-slate-500 hover:text-white hover:bg-white/10 transition-colors text-xs">✕</button>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-2 gap-2 mb-3">
        <div className="bg-white/5 p-2 rounded-md border border-white/5">
          <p className="text-[8px] text-slate-500 uppercase tracking-wider mb-0.5">Current Rainfall</p>
          <div className="flex items-center gap-1.5">
            {rainfallSummary.current === 'Data unavailable' ? (
              <span className="text-xs font-bold text-slate-400">Data unavailable</span>
            ) : (
              <>
                <span className="text-sm font-bold text-slate-100">{rainfallSummary.current}</span>
                <span className="text-[10px] text-slate-400">mm/hr</span>
                <span className="text-xs font-bold ml-auto" style={{ color: trendColor[rainfallSummary.trend] }}>
                  {trendIcon[rainfallSummary.trend]} {rainfallSummary.trend}
                </span>
              </>
            )}
          </div>
        </div>
        <div className="bg-white/5 p-2 rounded-md border border-white/5">
          <p className="text-[8px] text-slate-500 uppercase tracking-wider mb-0.5">River Discharge (modelled)</p>
          <div className="flex items-center gap-1.5">
            {waterSummary.status === 'unavailable' ? (
              <span className="text-xs font-bold text-slate-400">Data unavailable</span>
            ) : (
              <>
                <span className="text-sm font-bold text-slate-100">{waterSummary.current}</span>
                <span className="text-[10px] text-slate-400">m³/s</span>
                <span className="text-[9px] font-bold uppercase px-1.5 py-0.5 rounded-full ml-auto"
                  style={{ color: statusColor[waterSummary.status], background: `${statusColor[waterSummary.status]}15`, border: `1px solid ${statusColor[waterSummary.status]}30` }}>
                  {waterSummary.status}
                </span>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Charts */}
      <div className="space-y-3">
        <RainfallChart
          data={rainfallData}
          status={sourceStatus?.rainfall || 'ok'}
          lastFetched={rainfallLastFetched}
        />
        <WaterLevelChart
          data={waterLevelData}
          status={sourceStatus?.discharge || 'ok'}
          fetchedAt={dischargeMeta?.fetchedAt}
        />
      </div>
    </div>
  )
}
