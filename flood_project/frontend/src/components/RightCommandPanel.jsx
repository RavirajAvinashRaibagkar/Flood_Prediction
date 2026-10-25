import { useState, useEffect, useMemo } from 'react'
import { useRegion } from '../context/RegionContext'
import { getEvacuationRoute } from '../services/api'
import { getEmergencySignalsSummary } from '../services/api'
import ShelterPanel from './ShelterPanel'

export default function RightCommandPanel({ riskScore, alertStatus, selectedRouteId, setSelectedRouteId }) {
  const { region } = useRegion()
  const [routes, setRoutes] = useState([])

  const level = riskScore?.level || 'LOADING'
  const score = Number.isFinite(riskScore?.score) ? riskScore.score : null

  // Fetch real routes
  useEffect(() => {
    async function fetchRoutes() {
      try {
        const res = await getEvacuationRoute(region.lat, region.lng, region.lat - 0.03, region.lng - 0.04)
        if (res.data?.routes) {
          const colors = ['#22c55e', '#eab308', '#ef4444']
          const names = ['Route A — Primary', 'Route B — Alternate', 'Route C — Avoid']
          const badges = ['SAFE', 'MODERATE', 'DANGEROUS']
          setRoutes(res.data.routes.map((r, i) => ({
            id: r.id || `route_${i}`, name: names[i] || `Route ${i+1}`,
            distance: `${r.distance_km} km`, risk: r.risk_score || 0,
            time: `${r.duration_min} min`, color: colors[i] || '#64748b',
            badge: badges[i] || 'UNKNOWN',
          })))
        }
      } catch (err) { console.warn('Route fetch failed:', err) }
    }
    fetchRoutes()
  }, [region.lat, region.lng])

  // Population estimation (Item 6 — density × affected area heuristic)
  const populationAtRisk = useMemo(() => {
    // avg urban density ~5000/km², affected area estimated from score
    const density = 5000
    if (score === null) return null
    const affectedAreaKm2 = (score / 100) * 8 // up to 8km² at 100% risk
    const total = Math.round(density * affectedAreaKm2)
    const zones = Math.max(1, Math.round(score / 25))
    return { total, zones }
  }, [score])

  // Crowdsourced emergency signals summary (real, anonymous)
  const [signalSummary, setSignalSummary] = useState({ help: 0, stranded: 0, safe: 0, total: 0 })
  useEffect(() => {
    let mounted = true
    async function tick() {
      try {
        const res = await getEmergencySignalsSummary()
        if (mounted && res.data) setSignalSummary(res.data)
      } catch (e) {
        console.warn('Signal summary fetch failed:', e)
      }
    }
    tick()
    const t = setInterval(tick, 5000)
    return () => { mounted = false; clearInterval(t) }
  }, [])

  const timeInsight = useMemo(() => {
    const trendRaw = riskScore?.raw?.trend || riskScore?.trend || 'stable'
    const trend = score === null ? 'Unavailable' : trendRaw === 'rising' ? 'Increasing' : trendRaw === 'falling' ? 'Decreasing' : 'Stable'
    const confidence = null
    const eta = 'Not available'
    const peak = 'Not available'
    return { eta, peak, trend, confidence }
  }, [riskScore, score])

  const riskBg = level === 'HIGH' ? 'bg-red-500/12 border-red-500/25' : level === 'MEDIUM' ? 'bg-amber-500/12 border-amber-500/25' : level === 'LOW' ? 'bg-green-500/12 border-green-500/25' : 'bg-slate-500/10 border-slate-500/20'
  const riskTextColor = level === 'HIGH' ? 'text-red-400' : level === 'MEDIUM' ? 'text-amber-400' : level === 'LOW' ? 'text-green-400' : 'text-slate-400'
  const riskBorderColor = level === 'HIGH' ? 'border-l-red-500' : level === 'MEDIUM' ? 'border-l-amber-500' : level === 'LOW' ? 'border-l-green-500' : 'border-l-slate-500'

  return (
    <div className="h-full flex flex-col gap-2.5">
      {/* ═══ BLOCK 1: RISK STATUS ═══ */}
      <div className={`panel p-3 rounded-lg border ${riskBg}`}>
        <div className="flex items-center justify-between mb-1.5">
          <span className={`text-xl font-extrabold tracking-tight ${riskTextColor}`}>{level} RISK</span>
          <span className="text-[10px] font-mono text-slate-400 bg-white/5 px-2 py-0.5 rounded">{score === null ? '—' : `${score}%`}</span>
        </div>
        <div className="space-y-0.5 text-[11px] text-slate-300">
          <p>⏱ {score === null ? 'Risk service unavailable' : level === 'HIGH' ? 'High risk — follow local authority guidance' : level === 'MEDIUM' ? 'Monitor local advisories' : 'No elevated risk detected'}</p>
          <p>👥 Population at Risk (rough estimate): {populationAtRisk ? populationAtRisk.total.toLocaleString() : '—'}</p>
          <p>🆘 People requesting help: {signalSummary.help}</p>
        </div>
        <div className={`mt-2 p-2 rounded-md bg-white/5 border-l-2 ${riskBorderColor}`}>
          <p className="text-[10px] font-medium text-slate-200">
            {score === null ? 'Risk assessment unavailable.' : level === 'HIGH' ? 'Follow evacuation instructions from local authorities.' : level === 'MEDIUM' ? 'Monitor local advisories and water levels.' : 'Continue monitoring local conditions.'}
          </p>
        </div>
      </div>

      {/* ═══ BLOCK 2: ROUTE INTELLIGENCE ═══ */}
      <div className="panel p-3 rounded-lg">
        <h3 className="text-[9px] uppercase font-bold tracking-widest text-slate-500 mb-2">🗺 Evacuation Routes</h3>
        <div className="space-y-1.5">
          {(routes.length > 0 ? routes : [
            { id: 'route_unavailable', name: 'Route data unavailable', distance: '—', risk: '—', time: '—', color: '#64748b', badge: 'UNKNOWN' },
          ]).map((route, idx) => {
            const isSelected = selectedRouteId === route.id || (!selectedRouteId && idx === 0)
            const bc = { SAFE: '#22c55e', MODERATE: '#eab308', DANGEROUS: '#ef4444' }[route.badge] || '#64748b'
            return (
              <div key={route.id} onClick={() => setSelectedRouteId(route.id)}
                className={`p-2 rounded-md border cursor-pointer transition-all ${
                  isSelected ? 'bg-blue-500/10 border-blue-500/30' : 'bg-white/3 border-white/5 hover:border-white/10 opacity-60'
                }`}>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-[10px] font-bold text-slate-200">{route.name}</span>
                  <span className="text-[8px] font-bold uppercase px-1.5 py-0.5 rounded-full" style={{ color: bc, background: `${bc}20`, border: `1px solid ${bc}40` }}>
                    {route.badge}
                  </span>
                </div>
                <div className="grid grid-cols-3 gap-1 text-[9px] font-mono">
                  <div className="text-slate-500">Dist: <span className="text-slate-300">{route.distance}</span></div>
                  <div className="text-slate-500">Risk: <span style={{ color: route.risk > 50 ? '#ef4444' : route.risk === '—' ? '#94a3b8' : '#22c55e' }}>{route.risk === '—' ? '—' : `${route.risk}%`}</span></div>
                  <div className="text-slate-500 text-right">ETA: <span className="text-cyan-400">{route.time}</span></div>
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* ═══ BLOCK 3: TIME INSIGHT ═══ */}
      <div className="panel p-3 rounded-lg">
        <h3 className="text-[9px] uppercase font-bold tracking-widest text-slate-500 mb-2">🕒 Time Insight</h3>
        <div className="space-y-1 text-[11px] text-slate-300">
          <div className="flex items-center justify-between">
            <span className="text-slate-500">Flood ETA</span>
            <span className="font-mono text-cyan-300">{timeInsight.eta}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-slate-500">Peak Risk</span>
            <span className="font-mono text-slate-200">{timeInsight.peak}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-slate-500">Trend</span>
            <span className="font-mono text-amber-300">{timeInsight.trend}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-slate-500">Confidence</span>
            <span className="font-mono text-slate-400">{timeInsight.confidence === null ? '—' : `${timeInsight.confidence}%`}</span>
          </div>
        </div>
      </div>

      {/* ═══ BLOCK 4: AUTOMATED ALERT STATUS ═══ */}
      <div className="panel p-3 rounded-lg">
        <h3 className="text-[9px] uppercase font-bold tracking-widest text-slate-500 mb-2">⚡ Alert Status</h3>
        <div className="space-y-1.5">
          <AlertRow label="Public Alert Recommendation" status={alertStatus.publicAlert} />
          <AlertRow label="Authority Recommendation" status={alertStatus.authorities} />
          {alertStatus.lastTriggerTime && (
            <div className="pt-1.5 border-t border-white/5 space-y-1">
              <div className="flex justify-between text-[9px]">
                <span className="text-slate-500">Last Trigger:</span>
                <span className="text-slate-300 font-mono">{alertStatus.lastTriggerTime}</span>
              </div>
              {alertStatus.triggerReason && (
                <p className="text-[9px] text-slate-400 italic leading-tight">{alertStatus.triggerReason}</p>
              )}
            </div>
          )}
        </div>
      </div>

      {/* ═══ BLOCK 5: EMERGENCY SIGNALS SUMMARY ═══ */}
      <div className="panel p-3 rounded-lg">
        <h3 className="text-[9px] uppercase font-bold tracking-widest text-slate-500 mb-2">🚨 Emergency Signals</h3>
        <div className="space-y-1 text-[11px] text-slate-300">
          <div className="flex items-center justify-between">
            <span>🔴 Need Help</span>
            <span className="font-mono text-red-300">{signalSummary.help}</span>
          </div>
          <div className="flex items-center justify-between">
            <span>🟡 Stranded</span>
            <span className="font-mono text-amber-300">{signalSummary.stranded}</span>
          </div>
          <div className="flex items-center justify-between">
            <span>🟢 Safe</span>
            <span className="font-mono text-green-300">{signalSummary.safe}</span>
          </div>
        </div>
        <div className="mt-2 p-2 rounded bg-white/5 border border-white/5 text-[10px] text-slate-300">
          {signalSummary.help >= 3 ? 'High Priority Zone detected — cluster of help requests.' : 'No high-priority clusters detected.'}
        </div>
      </div>

      {/* ═══ BLOCK 6: SHELTER INTELLIGENCE ═══ */}
      <ShelterPanel />
    </div>
  )
}

function AlertRow({ label, status }) {
  const statusColors = {
    'SENT': { bg: 'bg-green-500/15', text: 'text-green-400', border: 'border-green-500/30' },
    'NOTIFIED': { bg: 'bg-green-500/15', text: 'text-green-400', border: 'border-green-500/30' },
    'PENDING': { bg: 'bg-amber-500/10', text: 'text-amber-400', border: 'border-amber-500/20' },
    'STANDBY': { bg: 'bg-blue-500/10', text: 'text-blue-400', border: 'border-blue-500/20' },
    'NO ACTION': { bg: 'bg-white/5', text: 'text-slate-500', border: 'border-white/5' },
    'RECOMMENDED': { bg: 'bg-amber-500/10', text: 'text-amber-400', border: 'border-amber-500/20' },
    'UNAVAILABLE': { bg: 'bg-white/5', text: 'text-slate-500', border: 'border-white/5' },
  }
  const c = statusColors[status] || statusColors.UNAVAILABLE

  return (
    <div className="flex items-center justify-between">
      <span className="text-[10px] text-slate-400">{label}</span>
      <span className={`text-[9px] font-bold uppercase px-2 py-0.5 rounded-full border ${c.bg} ${c.text} ${c.border}`}>
      {status === 'RECOMMENDED' ? '• ' : ''}{status}
      </span>
    </div>
  )
}
