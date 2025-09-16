import React from 'react'
import { Polyline, Tooltip } from 'react-leaflet'

async function fetchRoute() {
  const host = window.location.origin
  const url = `${host}/route-engine/routes`
  try {
    const r = await fetch(url, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        start_lon: -122.58, start_lat: 37.70,
        goal_lon: -122.35, goal_lat: 37.88,
        alternates: 2,
      })
    })
    if (!r.ok) throw new Error('route failed')
    return await r.json()
  } catch (e) {
    return null
  }
}

export const RoutesLayer: React.FC<{ onExplain?: (info: any)=>void }> = ({ onExplain }) => {
  const [routes, setRoutes] = React.useState<any | null>(null)

  React.useEffect(() => {
    let cancelled = false
    const run = async () => {
      const data = await fetchRoute()
      if (!cancelled) setRoutes(data)
      // Refresh periodically in case ISR updates occur
      setTimeout(run, 5000)
    }
    run()
    return () => { cancelled = true }
  }, [])

  if (!routes) return null
  const lines: Array<{ coords: [number, number][], color: string, label: string }> = []
  const toLatLng = (p: [number, number]) => [p[1], p[0]] as [number, number]
  try {
    const primary = routes.primary?.path?.map(toLatLng) ?? []
    if (primary.length) lines.push({ coords: primary, color: '#2e7d32', label: 'Primary' })
    for (let i = 0; i < (routes.alternates?.length || 0); i++) {
      const alt = routes.alternates[i]?.path?.map(toLatLng) ?? []
      if (alt.length) lines.push({ coords: alt, color: '#0277bd', label: `Alt ${i+1}` })
    }
  } catch {}

  React.useEffect(() => {
    if (!onExplain || !routes) return
    const evidence = (routes.primary?.hazards || []).map((h: any) => h.evidence).filter(Boolean)
    onExplain({ compute_ms: routes.compute_ms, evidence })
  }, [routes, onExplain])

  return (
    <>
      {lines.map((l, idx) => (
        <Polyline key={`r-${idx}`} positions={l.coords} pathOptions={{ color: l.color, weight: 4, opacity: 0.9 }}>
          <Tooltip sticky>{l.label}</Tooltip>
        </Polyline>
      ))}
    </>
  )
}
