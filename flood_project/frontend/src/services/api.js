import axios from 'axios'

const api = axios.create({
    baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api',
    timeout: 30000,
    headers: { 'Content-Type': 'application/json' }
})

// ─── Prediction Endpoints ──────────────────────────────────
export const predictImage = (formData) =>
    api.post('/predict/image', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
    })

export const predictSegmentation = (formData) =>
    api.post('/predict/segmentation', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
    })

export const getSampleS1Chips = () =>
    api.get('/predict/sample-s1')

export const predictSegmentSample = (chipFilename) =>
    api.post('/predict/segment-sample', null, {
        params: { chip_filename: chipFilename }
    })

export const predictTimeseries = (formData) =>
    api.post('/predict/timeseries', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
    })

export const getRiskScore = (lat, lng) =>
    api.get('/predict/risk-score', { params: { lat, lng } })

// ─── Weather Endpoints (Open-Meteo via backend) ─────────────
export const getCurrentWeather = (lat, lng) =>
    api.get('/weather/current', { params: { lat, lng } })

export const getWeatherHistory = (lat, lng, hours = 48) =>
    api.get('/weather/history', { params: { lat, lng, hours } })

export const getWeatherForecast = (lat, lng) =>
    api.get('/weather/forecast', { params: { lat, lng } })

// ─── Map Endpoints ─────────────────────────────────────────
export const getFloodZones = (lat, lng) =>
    api.get('/map/flood-zones', { params: { lat, lng } })

export const getEvacuationRoute = (originLat, originLng, destLat, destLng) =>
    api.get('/map/evacuation-route', {
        params: { origin_lat: originLat, origin_lng: originLng, dest_lat: destLat, dest_lng: destLng }
    })

export const getSensors = (lat, lng) =>
    api.get('/map/sensors', { params: { lat, lng } })

export const getInfrastructure = (lat, lng) =>
    api.get('/map/infrastructure', { params: { lat, lng } })

// ─── Sensor Endpoints ──────────────────────────────────────
export const listSensors = (lat, lng) =>
    api.get('/sensors', { params: { lat, lng } })

export const getSensorLevels = (lat, lng) =>
    api.get('/sensors/levels', { params: { lat, lng } })

export const getSensorHistory = (id) =>
    api.get(`/sensors/${id}`)

export const ingestSensorData = (data) =>
    api.post('/sensors/ingest', data)

// ─── Emergency Signals (Crowdsourced) ───────────────────────
export const listEmergencySignals = ({ bbox } = {}) =>
    api.get('/signals', { params: bbox ? { bbox } : {} })

export const getEmergencySignalsSummary = () =>
    api.get('/signals/summary')

export const createEmergencySignal = ({ lat, lng, status, signal_id }) =>
    api.post('/signals', { lat, lng, status, signal_id })

export const updateEmergencySignal = (signal_id, { status }) =>
    api.patch(`/signals/${signal_id}`, { status })

export const deleteEmergencySignal = (signal_id) =>
    api.delete(`/signals/${signal_id}`)

// ─── Shelter Endpoints ──────────────────────────────────────
export const listShelters = ({ bbox } = {}) =>
    api.get('/shelters', { params: bbox ? { bbox } : {} })

export const getShelterSummary = () =>
    api.get('/shelters/summary')

export const createShelter = (data) =>
    api.post('/shelters', data)

export const updateShelter = (shelter_id, data) =>
    api.patch(`/shelters/${shelter_id}`, data)

export const deleteShelter = (shelter_id) =>
    api.delete(`/shelters/${shelter_id}`)

export const getNearestShelter = (lat, lng, maxDistanceKm = 50) =>
    api.get('/shelters/nearest', { params: { lat, lng, max_distance_km: maxDistanceKm } })

export const getShelterRoute = (lat, lng, shelter_lat, shelter_lng) =>
    api.get('/shelters/safe-route', {
        params: { origin_lat: lat, origin_lng: lng, dest_lat: shelter_lat, dest_lng: shelter_lng }
    })

export default api
