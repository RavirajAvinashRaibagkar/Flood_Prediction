import { useState, useEffect, useCallback } from 'react'
import { getRiskScore, getWeatherHistory, getSensorLevels } from '../services/api'

// ─── Hook ──────────────────────────────────────────────────
export default function useFloodData(lat = 28.6139, lng = 77.209, refreshInterval = 30000) {
    const [riskScore, setRiskScore] = useState(null)
    const [rainfallData, setRainfallData] = useState([])
    const [waterLevelData, setWaterLevelData] = useState([])
    const [alerts, setAlerts] = useState([])
    const [loading, setLoading] = useState(true)

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
            if (weatherRes.data && weatherRes.data.data) {
                const hourly = weatherRes.data.data
                const chartData = hourly.map(h => ({
                    time: h.time ? new Date(h.time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '',
                    fullTime: h.time,
                    rainfall: h.precipitation || 0,
                }))
                setRainfallData(chartData)
            }
        } catch { setRainfallData([]) }

        try {
            // Fetch real river discharge for water level chart
            const hydroRes = await getSensorLevels(lat, lng)
            if (hydroRes.data && hydroRes.data.daily) {
                const chartData = hydroRes.data.daily.map(d => ({
                    time: d.date,
                    fullTime: d.date,
                    level: d.discharge != null ? parseFloat(d.discharge.toFixed(2)) : 0,
                    upper: d.discharge != null ? parseFloat((d.discharge * 1.15).toFixed(2)) : 0,
                    lower: d.discharge != null ? parseFloat((d.discharge * 0.85).toFixed(2)) : 0,
                    danger: hydroRes.data.danger_threshold || 100,
                    isLive: !d.is_forecast,
                    isForecast: d.is_forecast,
                }))
                setWaterLevelData(chartData)
            }
        } catch { setWaterLevelData([]) }

        setLoading(false)
    }, [lat, lng])

    useEffect(() => {
        Promise.resolve().then(fetchData)
        const interval = setInterval(fetchData, refreshInterval)
        
        // ─── WebSocket Integration ───
        const apiBase = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api'
        const wsBase = apiBase.replace(/^http/, 'ws').replace(/\/api\/?$/, '')
        const alertWS = new WebSocket(`${wsBase}/ws/alerts`)
        const sensorWS = new WebSocket(`${wsBase}/ws/sensors`)

        alertWS.onmessage = (event) => {
            const newAlert = JSON.parse(event.data)
            setAlerts(prev => [newAlert, ...prev.slice(0, 19)])
        }

        sensorWS.onmessage = (event) => {
            const update = JSON.parse(event.data)
            setWaterLevelData(prev => {
                const time = new Date(update.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
                const newPoint = {
                    time,
                    fullTime: update.timestamp,
                    level: update.water_level,
                    upper: parseFloat((update.water_level + 0.3).toFixed(2)),
                    lower: parseFloat((update.water_level - 0.3).toFixed(2)),
                    danger: 4.0,
                    isLive: true
                }
                return [...prev.slice(-23), newPoint]
            })
        }

        return () => {
            clearInterval(interval)
            alertWS.close()
            sensorWS.close()
        }
    }, [fetchData, refreshInterval])

    return { riskScore, rainfallData, waterLevelData, alerts, loading, refetch: fetchData }
}

