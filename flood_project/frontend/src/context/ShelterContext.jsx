/* eslint-disable react-refresh/only-export-components */
import { createContext, useContext, useState, useCallback, useEffect, useMemo } from 'react'
import {
  listShelters,
  createShelter,
  updateShelter as apiUpdateShelter,
  deleteShelter as apiDeleteShelter,
  getNearestShelter,
  getShelterRoute,
} from '../services/api'
import { useRegion } from './RegionContext'

const ShelterContext = createContext(null)

const SHELTER_ID_KEY = 'fs_shelter_id'

function getOrCreateShelterId() {
  const existing = localStorage.getItem(SHELTER_ID_KEY)
  if (existing) return existing
  const id = (globalThis.crypto?.randomUUID?.() || `${Date.now().toString(16)}-${Math.random().toString(16).slice(2)}`).slice(0, 64)
  localStorage.setItem(SHELTER_ID_KEY, id)
  return id
}

function getShelterId() {
  return localStorage.getItem(SHELTER_ID_KEY)
}

function clearShelterId() {
  localStorage.removeItem(SHELTER_ID_KEY)
}

export function ShelterProvider({ children }) {
  const { region } = useRegion()
  const [shelters, setShelters] = useState([])
  const [summary, setSummary] = useState({ total: 0, hospitals: 0, schools: 0, government: 0, community: 0, estimated_capacity: 0 })
  const [selectedShelter, setSelectedShelter] = useState(null)
  const [shelterRoute, setShelterRoute] = useState(null)
  const [routeLoading, setRouteLoading] = useState(false)
  const [nearestSafe, setNearestSafe] = useState(null)
  const [userLocation, setUserLocation] = useState(null)

  // Track user location
  useEffect(() => {
    if (!navigator.geolocation) return
    navigator.geolocation.getCurrentPosition(
      (pos) => setUserLocation({ lat: pos.coords.latitude, lng: pos.coords.longitude }),
      () => {},
      { enableHighAccuracy: true, timeout: 10000, maximumAge: 30000 }
    )
    const watcher = navigator.geolocation.watchPosition(
      (pos) => setUserLocation({ lat: pos.coords.latitude, lng: pos.coords.longitude }),
      () => {},
      { enableHighAccuracy: false, timeout: 15000, maximumAge: 60000 }
    )
    return () => navigator.geolocation.clearWatch(watcher)
  }, [])

  // Fetch shelters only within the current map region. Previously this requested the
  // complete database, which kept showing seeded Delhi shelters after the map moved.
  const refresh = useCallback(async () => {
    try {
      const bbox = region.bbox?.join(',')
      const listRes = await listShelters(bbox ? { bbox } : {})
      const nearbyShelters = listRes.data || []
      const nearbySummary = nearbyShelters.reduce((totals, shelter) => {
        totals.total += 1
        const typeKey = shelter.type === 'hospital' ? 'hospitals'
          : shelter.type === 'school' ? 'schools'
            : shelter.type
        if (typeKey in totals) totals[typeKey] += 1
        const capacity = Number.parseInt(shelter.capacity, 10)
        totals.estimated_capacity += Number.isFinite(capacity) ? capacity : 0
        return totals
      }, { total: 0, hospitals: 0, schools: 0, government: 0, community: 0, estimated_capacity: 0 })
      setShelters(nearbyShelters)
      setSummary(nearbySummary)
    } catch (e) {
      console.warn('Shelter refresh failed:', e)
    }
  }, [region.bbox])

  useEffect(() => {
    // A selected shelter may belong to the previous region, so do not retain a
    // route or destination marker after the user moves the map.
    setSelectedShelter(null)
  }, [region.lat, region.lng])

  useEffect(() => {
    refresh()
    const t = setInterval(refresh, 10000)
    return () => clearInterval(t)
  }, [refresh])

  // Fetch nearest shelter when user location is available
  useEffect(() => {
    if (!userLocation) return
    let mounted = true
    async function fetchNearest() {
      try {
        const res = await getNearestShelter(userLocation.lat, userLocation.lng, 50)
        if (mounted && res.data) setNearestSafe(res.data)
      } catch (e) {
        console.warn('Nearest shelter fetch failed:', e)
      }
    }
    fetchNearest()
    const t = setInterval(fetchNearest, 15000)
    return () => { mounted = false; clearInterval(t) }
  }, [userLocation])

  // Fetch route when a shelter is selected
  useEffect(() => {
    if (!selectedShelter || !userLocation) {
      setShelterRoute(null)
      return
    }
    let mounted = true
    async function fetchRoute() {
      setRouteLoading(true)
      try {
        const res = await getShelterRoute(
          userLocation.lat, userLocation.lng,
          selectedShelter.lat, selectedShelter.lng
        )
        if (mounted && res.data) setShelterRoute(res.data)
      } catch (e) {
        console.warn('Shelter route fetch failed:', e)
      } finally {
        if (mounted) setRouteLoading(false)
      }
    }
    fetchRoute()
    return () => { mounted = false }
  }, [selectedShelter, userLocation])

  const registerShelter = useCallback(async (data) => {
    const shelter_id = getShelterId() || getOrCreateShelterId()
    await createShelter({ ...data, shelter_id })
    await refresh()
  }, [refresh])

  const updateShelter = useCallback(async (data) => {
    const shelter_id = getShelterId()
    if (!shelter_id) return
    await apiUpdateShelter(shelter_id, data)
    await refresh()
  }, [refresh])

  const removeShelter = useCallback(async () => {
    const shelter_id = getShelterId()
    if (!shelter_id) return
    await apiDeleteShelter(shelter_id)
    clearShelterId()
    await refresh()
  }, [refresh])

  const hasActiveShelter = useMemo(() => {
    const sid = getShelterId()
    if (!sid) return false
    return shelters.some(s => s.public_id === sid.slice(0, 6))
  }, [shelters])

  return (
    <ShelterContext.Provider value={{
      shelters,
      summary,
      selectedShelter,
      setSelectedShelter,
      shelterRoute,
      routeLoading,
      nearestSafe,
      userLocation,
      registerShelter,
      updateShelter,
      removeShelter,
      hasActiveShelter,
      refresh,
    }}>
      {children}
    </ShelterContext.Provider>
  )
}

export function useShelter() {
  const ctx = useContext(ShelterContext)
  if (!ctx) throw new Error('useShelter must be used within ShelterProvider')
  return ctx
}
