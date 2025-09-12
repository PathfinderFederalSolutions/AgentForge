import React from 'react'
import { MapContainer, TileLayer } from 'react-leaflet'
import 'leaflet/dist/leaflet.css'
import { useEventStream, type EventFeature, sanitizeUrl } from './services/events'
import { ThreatsLayer } from './layers/Threats'
import { AlertsLayer } from './layers/Alerts'

const center: [number, number] = [37.7749, -122.4194]

function Drawer({ feature, onClose }: { feature: EventFeature | null, onClose: () => void }) {
  if (!feature) return null
  const title = String(feature.properties?.title ?? feature.properties?.type ?? 'Detail')
  const desc = String(feature.properties?.description ?? '')
  const evidenceRaw = feature.properties?.evidence_link as string | undefined
  const evidence = sanitizeUrl(evidenceRaw)
  return (
    <div role="dialog" aria-modal="true" className="drawer" onClick={onClose}>
      <div className="panel" onClick={(e: React.MouseEvent<HTMLDivElement>) => e.stopPropagation()}>
        <div className="panel-header">
          <strong>{title}</strong>
          <button className="close" onClick={onClose} aria-label="Close">×</button>
        </div>
        <div className="panel-body">
          <p style={{whiteSpace:'pre-wrap'}}>{desc}</p>
          {evidence && (
            <p>
              <a href={evidence} target="_blank" rel="noopener noreferrer">View evidence</a>
            </p>
          )}
        </div>
      </div>
      <style>{`
        .drawer { position:fixed; inset:0; background:rgba(0,0,0,0.35); display:flex; justify-content:flex-end; z-index:2000; }
        .panel { width:min(420px, 90%); height:100%; background:#fff; box-shadow:-2px 0 12px rgba(0,0,0,0.2); display:flex; flex-direction:column; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
        .panel-header { display:flex; align-items:center; justify-content:space-between; padding:12px 14px; border-bottom:1px solid #eee; }
        .panel-body { padding:12px 14px; overflow:auto; }
        .close { all:unset; cursor:pointer; font-size:20px; line-height:1; padding:2px 6px; }
      `}</style>
    </div>
  )
}

export const App: React.FC = () => {
  const features = useEventStream('/events/stream')
  const [selected, setSelected] = React.useState<EventFeature | null>(null)

  const threats = React.useMemo(() => features.filter((f: EventFeature) => (f.properties?.layer ?? f.properties?.type) === 'threat'), [features])
  const alerts = React.useMemo(() => features.filter((f: EventFeature) => (f.properties?.layer ?? f.properties?.type) === 'alert'), [features])

  const backlogCount = alerts.length + threats.length

  return (
    <div className="map-container" style={{height:'100%'}}>
      <div className="badge" aria-live="polite" title="Backlog status">Backlog: {backlogCount}</div>
      <MapContainer center={center} zoom={11} style={{height:'100%'}}>
        <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
        <ThreatsLayer items={threats} onSelect={setSelected} />
        <AlertsLayer items={alerts} onSelect={setSelected} />
      </MapContainer>
      <Drawer feature={selected} onClose={() => setSelected(null)} />
    </div>
  )
}

export default App
