/* eslint-disable react-refresh/only-export-components */
import { createContext, useContext, useState, useCallback, useEffect } from 'react'

const RegionContext = createContext(null)

const DEFAULT_REGION = {
    lat: 28.6139,
    lng: 77.2090,
    regionName: "Delhi NCR",
    bbox: [28.40, 76.84, 28.88, 77.35], // [south, west, north, east]
    zoom: 11
}

export function RegionProvider({ children }) {
    const [region, setRegion] = useState(DEFAULT_REGION)
    const [locationStatus, setLocationStatus] = useState('idle')
    const [locationError, setLocationError] = useState('')

    const updateRegion = useCallback((newRegion) => {
        setRegion(prev => ({ ...prev, ...newRegion }))
    }, [])

    const locateMe = useCallback(async () => {
        if (!navigator.geolocation) {
            setLocationStatus('error')
            setLocationError('This browser does not support location services.')
            return
        }

        setLocationStatus('locating')
        setLocationError('')

        try {
            const position = await new Promise((resolve, reject) => {
                navigator.geolocation.getCurrentPosition(resolve, reject, {
                    enableHighAccuracy: true,
                    timeout: 12000,
                    maximumAge: 30000,
                })
            })
            const { latitude, longitude, accuracy } = position.coords
            let regionName = 'Current location'

            // Coordinates are useful immediately. A reverse lookup adds a readable locality
            // when the public geocoding service is available.
            try {
                const response = await fetch(
                    `https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat=${latitude}&lon=${longitude}&zoom=10`,
                )
                if (response.ok) {
                    const place = await response.json()
                    const address = place.address || {}
                    regionName = address.city || address.town || address.village || address.county || place.display_name || regionName
                }
            } catch (error) {
                console.warn('Location name lookup failed:', error)
            }

            updateRegion({
                lat: latitude,
                lng: longitude,
                regionName,
                bbox: [latitude - 0.15, longitude - 0.25, latitude + 0.15, longitude + 0.25],
                zoom: 12,
                locationSource: 'device',
                locationAccuracyM: Math.round(accuracy),
                locationUpdatedAt: new Date().toISOString(),
            })
            setLocationStatus('ready')
        } catch (error) {
            const messages = {
                1: 'Location permission was denied. Choose a region from search instead.',
                2: 'Your location could not be determined. Try again outdoors or select a region.',
                3: 'Location request timed out. Please try again.',
            }
            setLocationStatus('error')
            setLocationError(messages[error?.code] || 'Unable to fetch your current location.')
        }
    }, [updateRegion])

    // The dashboard already needs the user's location for nearby shelters and
    // predictions, so request it once when the app opens. If permission is
    // declined, the Delhi demonstration region remains available as a fallback.
    useEffect(() => {
        locateMe()
    }, [locateMe])

    return (
        <RegionContext.Provider value={{ region, updateRegion, locateMe, locationStatus, locationError }}>
            {children}
        </RegionContext.Provider>
    )
}

export function useRegion() {
    const ctx = useContext(RegionContext)
    if (!ctx) throw new Error("useRegion must be used within RegionProvider")
    return ctx
}

// Intentionally no default export to keep Fast Refresh happy.
