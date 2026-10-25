import { useState, useEffect, useCallback } from 'react'
import { getRiskScore, getWeatherHistory, getSensorLevels } from '../services/api'

// ─── Hook ──────────────────────────────────────────────────
export default function useFloodData(lat = 28.6139, lng = 77.209, refreshInterval = 30000) {
    const [riskScore, setRiskScore] = useState(null)
    const [rainfallData, setRainfallData] = useState([])
    const [waterLevelData, setWaterLevelData] = useState([])
    const [alerts, setAlerts] = useState([])
    const [loading, setLoading] = useState(true)
    const [sourceStatus, setSourceStatus] = useState({ rainfall: 'empty', discharge: 'empty' })
    const [dischargeMeta, setDischargeMeta] = useState({
        fetchedAt: null,
        source: null,
        dangerThresholdDischarge: null,
    })
    const [rainfallLastFetched, setRainfallLastFetched] = useState(null)

    const fetchData = useCallback(async () => {
        try {
            // Fetch risk score from backend
            const riskRes = await getRiskScore(lat, lng)
            if (riskRes.data) setRiskScore(riskRes.data)
        } catch {
            setRiskScore({ score: null, level: 'UNAVAILABLE', error: 'Risk service is unavailable.' })
        }

        try {
            // Fetch real weather history for rainfall chart (past 72h)
            const weatherRes = await getWeatherHistory(lat, lng, 72)
            const hourly = weatherRes.data?.data || []
            if (hourly.length > 0) {
                const chartData = hourly.map(h => ({
                    time: h.time ? new Date(h.time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '',
                    fullTime: h.time,
                    rainfall: typeof h.precipitation === 'number' ? h.precipitation : 0,
                }))
                setRainfallData(chartData)
                setSourceStatus(prev => ({ ...prev, rainfall: 'ok' }))
                setRainfallLastFetched(Date.now())
            } else {
                setRainfallData([])
                setSourceStatus(prev => ({ ...prev, rainfall: 'empty' }))
            }
        } catch {
            setRainfallData([])
            setSourceStatus(prev => ({ ...prev, rainfall: 'error' }))
        }

        try {
            // Fetch real river discharge from GloFAS model
            const hydroRes = await getSensorLevels(lat, lng)
            const data = hydroRes.data
            const isAvail = data && data.available !== false
            const daily = data?.daily || []
            const dangerThreshold = data?.danger_threshold_discharge ?? null

            setDischargeMeta({
                fetchedAt: data?.fetched_at || null,
                source: data?.source || 'Open-Meteo GloFAS (modelled)',
                dangerThresholdDischarge: dangerThreshold,
            })

            if (isAvail && daily.length > 0) {
                const chartData = daily.map(d => {
                    const point = {
                        time: d.date,
                        fullTime: d.date,
                        level: d.discharge != null ? parseFloat(Number(d.discharge).toFixed(2)) : null,
                        danger: dangerThreshold,
                        isLive: !d.is_forecast,
                        isForecast: d.is_forecast,
                    }
                    // Use backend band if present, else omit
                    if (d.lower != null && d.upper != null) {
                        point.lower = parseFloat(Number(d.lower).toFixed(2))
                        point.upper = parseFloat(Number(d.upper).toFixed(2))
                    }
                    return point
                })
                setWaterLevelData(chartData)
                setSourceStatus(prev => ({ ...prev, discharge: 'ok' }))
            } else {
                setWaterLevelData([])
                setSourceStatus(prev => ({
                    ...prev,
                    discharge: !isAvail ? (data?.error ? 'error' : 'empty') : 'empty'
                }))
            }
        } catch {
            setWaterLevelData([])
            setSourceStatus(prev => ({ ...prev, discharge: 'error' }))
        }

        setLoading(false)
    }, [lat, lng])

    useEffect(() => {
        Promise.resolve().then(fetchData)
        const interval = setInterval(fetchData, refreshInterval)
        
        // ─── WebSocket Integration (alerts only) ───
        const apiBase = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api'
        const wsBase = apiBase.replace(/^http/, 'ws').replace(/\/api\/?$/, '')
        const alertWS = new WebSocket(`${wsBase}/ws/alerts`)

        alertWS.onmessage = (event) => {
            try {
                const newAlert = JSON.parse(event.data)
                setAlerts(prev => [newAlert, ...prev.slice(0, 19)])
            } catch {
                // Ignore parse errors
            }
        }

        return () => {
            clearInterval(interval)
            alertWS.close()
        }
    }, [fetchData, refreshInterval])

    return {
        riskScore,
        rainfallData,
        waterLevelData,
        alerts,
        loading,
        sourceStatus,
        dischargeMeta,
        rainfallLastFetched,
        refetch: fetchData,
    }
}

