export type EventFeature = {
  type: 'Feature'
  geometry?: { type?: string; coordinates?: [number, number] | number[] }
  properties?: Record<string, any>
}

export function sanitizeUrl(url: string | undefined): string | undefined {
  if (!url) return undefined
  try {
    const u = new URL(url, window.location.origin)
    const scheme = u.protocol.replace(':','')
    if (scheme === 'http' || scheme === 'https') return u.toString()
    if (u.origin === window.location.origin) return u.toString()
  } catch {}
  // disallow javascript:, data:, etc.
  return undefined
}

function toFeature(obj: any): EventFeature | null {
  if (!obj) return null
  // Already a GeoJSON-like feature
  if (obj.type === 'Feature') return obj as EventFeature
  // Convert a simple marker-like event to Feature
  const coords = obj.geometry?.coordinates || obj.properties?.coordinates || obj.coords
  if (Array.isArray(coords) && coords.length >= 2) {
    return {
      type: 'Feature',
      geometry: { type: 'Point', coordinates: [coords[0], coords[1]] },
      properties: { ...obj.properties, ...obj, coordinates: undefined },
    }
  }
  return null
}

export function useEventStream(url: string) {
  const [features, setFeatures] = useStateSafe<EventFeature[]>([])
  React.useEffect(() => {
    let cancelled = false
    let ws: WebSocket | null = null
    const add = (data: any) => {
      const list: EventFeature[] = []
      const push = (x: any) => {
        const f = toFeature(x)
        if (f) list.push(f)
      }
      if (Array.isArray(data)) data.forEach(push)
      else push(data)
      if (cancelled) return
      setFeatures((prev) => [...prev, ...list].slice(-300))
    }

    const openSSE = () => {
      const es = new EventSource(url)
      es.onmessage = (e) => { try { add(JSON.parse(e.data)) } catch {} }
      es.onerror = () => {
        es.close()
        // try WS fallback
        tryWS()
      }
      return es
    }

    const tryWS = () => {
      try {
        const loc = window.location
        const wsProto = loc.protocol === 'https:' ? 'wss' : 'ws'
        const wsUrl = `${wsProto}://${loc.host}/ws`
        ws = new WebSocket(wsUrl)
        ws.onmessage = (ev) => { try { add(JSON.parse(ev.data)) } catch {} }
        ws.onerror = () => { try { ws?.close() } catch {} }
      } catch {}
    }

    const es = openSSE()
    return () => { cancelled = true; try { es.close() } catch {}; try { ws?.close() } catch {} }
  }, [url])
  return features
}

function useStateSafe<T>(initial: T) {
  const [v, setV] = React.useState<T>(initial)
  const mounted = React.useRef(true)
  React.useEffect(() => () => { mounted.current = false }, [])
  const setSafe = React.useCallback((updater: React.SetStateAction<T>) => {
    if (mounted.current) setV(updater)
  }, [])
  return [v, setSafe] as const
}

import React from 'react'
