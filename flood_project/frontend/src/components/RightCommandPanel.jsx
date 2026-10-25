import { useState, useEffect, useMemo } from 'react'
import { getEmergencySignalsSummary } from '../services/api'
import ShelterPanel from './ShelterPanel'
import PlaceSearchRoute from './PlaceSearchRoute'

export default function RightCommandPanel({ riskScore, alertStatus, routing, origin }) {

  const level = riskScore?.level || 'LOADING'
  const score = Number.isFinite(riskScore?.score) ? riskScore.score : null

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
      {/* ═══ BLOCK 0: PLACE SEARCH + NEAREST/SAFEST ROUTE ═══ */}
      <PlaceSearchRoute routing={routing} origin={origin} />

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
