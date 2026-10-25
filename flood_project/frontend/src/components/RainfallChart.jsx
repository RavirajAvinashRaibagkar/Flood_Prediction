import { useMemo } from 'react'
import { AreaChart, Area, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine, ReferenceDot } from 'recharts'

export default function RainfallChart({ data = [], status = 'ok', lastFetched = null }) {
  // Compute rolling 24-hour accumulation on hourly precipitation
  const chartData = useMemo(() => {
    return data.map((d, i) => {
      const start = Math.max(0, i - 23)
      let rolling24h = 0
      for (let k = start; k <= i; k++) {
        rolling24h += (data[k]?.rainfall || 0)
      }
      return {
        ...d,
        rolling24h: parseFloat(rolling24h.toFixed(1)),
      }
    })
  }, [data])

  const { peakHourly, peakRolling, maxVal } = useMemo(() => {
    let peakHourly = 0, peakRolling = 0
    chartData.forEach((d) => {
      if ((d.rainfall || 0) > peakHourly) peakHourly = d.rainfall
      if ((d.rolling24h || 0) > peakRolling) peakRolling = d.rolling24h
    })
    const maxVal = Math.max(peakRolling * 1.15, 75.0)
    return { peakHourly, peakRolling, maxVal }
  }, [chartData])

  const current = chartData.length > 0 ? chartData[chartData.length - 1] : null
  const isLive = Boolean(lastFetched && (Date.now() - lastFetched) < 5 * 60 * 1000)
  const isUnavailable = status === 'error' || status === 'empty' || chartData.length === 0

  if (isUnavailable) {
    return (
      <div style={{ background: 'rgba(15,23,42,0.95)', backdropFilter: 'blur(12px)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', padding: '12px' }}>
        <div className="flex items-center justify-between mb-2">
          <div>
            <h3 className="text-[10px] uppercase font-bold tracking-widest text-slate-400">Rainfall Trend</h3>
            <p className="text-[9px] text-slate-600">Source: Open-Meteo, hourly precipitation</p>
          </div>
        </div>
        <div className="h-[140px] flex items-center justify-center text-slate-500 text-[10px]">
          Rainfall data unavailable
        </div>
      </div>
    )
  }

  return (
    <div style={{ background: 'rgba(15,23,42,0.95)', backdropFilter: 'blur(12px)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', padding: '12px' }}>
      <div className="flex items-center justify-between mb-2">
        <div>
          <h3 className="text-[10px] uppercase font-bold tracking-widest text-slate-400">Rainfall Trend</h3>
          <p className="text-[9px] text-slate-600">Source: Open-Meteo, hourly precipitation</p>
        </div>
        {isLive && (
          <div className="flex items-center gap-1 px-2 py-0.5 rounded bg-blue-500/10 border border-blue-500/20">
            <div className="w-1 h-1 rounded-full bg-blue-400 animate-pulse"></div>
            <span className="text-[9px] font-medium text-blue-400">LIVE</span>
          </div>
        )}
      </div>

      <div style={{ height: 140 }}>
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={chartData} margin={{ top: 5, right: 5, left: -25, bottom: 0 }}>
            <defs>
              <linearGradient id="rainfallGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.35} />
                <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.0} />
              </linearGradient>
            </defs>
            <XAxis dataKey="time" tick={{ fill: '#475569', fontSize: 8 }} tickLine={false} axisLine={false} interval={17} />
            <YAxis tick={{ fill: '#475569', fontSize: 8 }} tickLine={false} axisLine={false} domain={[0, Math.ceil(maxVal)]} />
            <Tooltip
              contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(59,130,246,0.2)', borderRadius: '8px', fontSize: '10px', color: '#e2e8f0' }}
              formatter={(val, name) => {
                if (name === 'rolling24h') return [`${Number(val).toFixed(1)} mm/24h`, '24h Accumulation']
                if (name === 'rainfall') return [`${Number(val).toFixed(1)} mm/hr`, 'Hourly Rain']
                return [val, name]
              }}
            />
            {/* 24h accumulation area */}
            <Area type="monotone" dataKey="rolling24h" stroke="#3b82f6" strokeWidth={2} fill="url(#rainfallGrad)" dot={false} />
            {/* Hourly rainfall line */}
            <Line type="monotone" dataKey="rainfall" stroke="#93c5fd" strokeWidth={1} dot={false} strokeDasharray="3 3" />

            {/* IMD 24h Thresholds */}
            <ReferenceLine y={64.5} stroke="#eab308" strokeWidth={1} strokeDasharray="4 3"
              label={{ value: 'IMD Heavy: 64.5', fill: '#eab308', fontSize: 8, position: 'right' }} />
            {maxVal >= 115.6 && (
              <ReferenceLine y={115.6} stroke="#f97316" strokeWidth={1} strokeDasharray="4 3"
                label={{ value: 'IMD Very Heavy: 115.6', fill: '#f97316', fontSize: 8, position: 'right' }} />
            )}
            {maxVal >= 204.5 && (
              <ReferenceLine y={204.5} stroke="#ef4444" strokeWidth={1} strokeDasharray="4 3"
                label={{ value: 'IMD Extr. Heavy: 204.5', fill: '#ef4444', fontSize: 8, position: 'right' }} />
            )}

            {current && (
              <ReferenceDot x={current.time} y={current.rolling24h || 0} r={4} fill="#ffffff" stroke="#3b82f6" strokeWidth={2} />
            )}
          </AreaChart>
        </ResponsiveContainer>
      </div>

      <div className="flex items-center justify-between text-[8px] text-slate-500 mt-1 border-t border-white/5 pt-1">
        <span>Solid: 24h rolling sum (mm)</span>
        <span>Dashed: Hourly (mm/hr)</span>
      </div>
    </div>
  )
}
