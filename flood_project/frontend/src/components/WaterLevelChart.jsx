import { useMemo } from 'react'
import { AreaChart, Area, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine, ReferenceDot, ReferenceArea } from 'recharts'

export default function WaterLevelChart({ data = [], status = 'ok', fetchedAt = null }) {
  const hasBand = useMemo(() => data.some(d => d.lower != null && d.upper != null), [data])

  const { peak, peakIdx, dangerVal, maxVal } = useMemo(() => {
    let peak = 0, peakIdx = 0, dv = null
    data.forEach((d, i) => {
      const v = d.level || 0
      if (v > peak) { peak = v; peakIdx = i }
      if (d.danger != null) dv = d.danger
    })
    const computedMax = Math.max(peak * 1.2, (dv || 0) * 1.3, 1.0)
    return { peak, peakIdx, dangerVal: dv, maxVal: computedMax }
  }, [data])

  const current = data.length > 0 ? data[data.length - 1] : null
  const isUnavailable = status === 'error' || status === 'empty' || data.length === 0

  if (isUnavailable) {
    return (
      <div style={{ background: 'rgba(15,23,42,0.95)', backdropFilter: 'blur(12px)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', padding: '12px' }}>
        <div className="flex items-center justify-between mb-2">
          <div>
            <h3 className="text-[10px] uppercase font-bold tracking-widest text-slate-400">River Discharge (modelled)</h3>
            <p className="text-[9px] text-slate-600">GloFAS estimate — m³/s</p>
          </div>
        </div>
        <div className="h-[140px] flex flex-col items-center justify-center text-slate-500 text-[10px] gap-1">
          <span>Data unavailable</span>
          <span className="text-[8px] text-slate-600">Modelled GloFAS river discharge could not be loaded</span>
        </div>
        <div className="mt-2 text-[8px] text-slate-500 text-center border-t border-white/5 pt-1.5 truncate">
          Source: Open-Meteo GloFAS - modelled estimate, not a gauge reading - updated {fetchedAt || 'unknown'}
        </div>
      </div>
    )
  }

  return (
    <div style={{ background: 'rgba(15,23,42,0.95)', backdropFilter: 'blur(12px)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', padding: '12px' }}>
      <div className="flex items-center justify-between mb-2">
        <div>
          <h3 className="text-[10px] uppercase font-bold tracking-widest text-slate-400">River Discharge (modelled)</h3>
          <p className="text-[9px] text-slate-600">GloFAS estimate — m³/s</p>
        </div>
        <div className="flex items-center gap-2 text-[9px]">
          <span className="flex items-center gap-1"><span className="w-2 h-0.5 bg-cyan-400 rounded-full"></span><span className="text-slate-500">Discharge</span></span>
          {dangerVal != null && (
            <span className="flex items-center gap-1"><span className="w-2 h-0.5 bg-red-400 rounded-full"></span><span className="text-slate-500">Danger Threshold</span></span>
          )}
          {hasBand && (
            <span className="flex items-center gap-1"><span className="w-2 h-0.5 bg-cyan-400/40 rounded-full"></span><span className="text-slate-500">p25–p75</span></span>
          )}
        </div>
      </div>

      <div style={{ height: 140 }}>
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 5, right: 5, left: -25, bottom: 0 }}>
            <defs>
              <linearGradient id="waterGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#06b6d4" stopOpacity={0.0} />
              </linearGradient>
            </defs>
            <XAxis dataKey="time" tick={{ fill: '#475569', fontSize: 8 }} tickLine={false} axisLine={false} interval={2} />
            <YAxis tick={{ fill: '#475569', fontSize: 8 }} tickLine={false} axisLine={false} />
            <Tooltip
              contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(6,182,212,0.2)', borderRadius: '8px', fontSize: '10px', color: '#e2e8f0' }}
              formatter={(val, name) => {
                const labels = { level: 'Discharge', upper: 'Upper Band (p75)', lower: 'Lower Band (p25)', danger: 'Danger Threshold' }
                if (!hasBand && (name === 'upper' || name === 'lower')) return null
                return [`${Number(val).toFixed(2)} m³/s`, labels[name] || name]
              }}
            />
            {dangerVal != null && (
              <ReferenceArea y1={dangerVal} y2={maxVal} fill="#ef4444" fillOpacity={0.06} />
            )}
            {hasBand && (
              <>
                <Area type="monotone" dataKey="upper" stroke="none" fill="url(#waterGrad)" />
                <Area type="monotone" dataKey="lower" stroke="none" fill="rgba(15,23,42,0.95)" />
              </>
            )}
            <Line type="monotone" dataKey="level" stroke="#06b6d4" strokeWidth={2.5} dot={false} />
            {dangerVal != null && (
              <ReferenceLine y={dangerVal} stroke="#ef4444" strokeWidth={1.5} strokeDasharray="6 3"
                label={{ value: `DANGER (${dangerVal.toFixed(1)})`, fill: '#ef4444', fontSize: 8, position: 'right' }} />
            )}
            {peak > 0 && (
              <ReferenceDot x={data[peakIdx]?.time} y={peak} r={4} fill="#ef4444" stroke="#0f172a" strokeWidth={2}
                label={{ value: `Peak: ${peak.toFixed(1)}`, fill: '#fca5a5', fontSize: 8, position: 'top' }} />
            )}
            {current && current.level != null && (
              <ReferenceDot x={current.time} y={current.level} r={4} fill="#ffffff" stroke="#06b6d4" strokeWidth={2} />
            )}
          </AreaChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-2 text-[8px] text-slate-500 text-center border-t border-white/5 pt-1.5 truncate">
        Source: Open-Meteo GloFAS - modelled estimate, not a gauge reading - updated {fetchedAt || 'unknown'}
      </div>
    </div>
  )
}
