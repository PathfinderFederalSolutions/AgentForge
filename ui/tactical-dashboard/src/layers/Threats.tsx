import React from 'react'
import { Marker, Popup } from 'react-leaflet'
import L from 'leaflet'
import type { EventFeature } from '../services/events'
import { sanitizeUrl } from '../services/events'

const redIcon = new L.Icon({
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
})

export const ThreatsLayer: React.FC<{ items: EventFeature[]; onSelect: (f: EventFeature)=>void }> = ({ items, onSelect }: { items: EventFeature[]; onSelect: (f: EventFeature)=>void }) => {
  return (
    <>
      {items.map((f: EventFeature, idx: number) => {
        const coords = f.geometry?.coordinates
        if (!coords || coords.length < 2) return null
        const [lon, lat] = coords as [number, number]
        const title = String(f.properties?.title ?? 'Threat')
        const evid = f.properties?.evidence_link as string | undefined
        const evidSafe = sanitizeUrl(evid)
        return (
          <Marker key={`t-${idx}`} position={[lat, lon]} icon={redIcon} eventHandlers={{ click: () => onSelect(f) }}>
            <Popup>
              <div>
                <strong>{title}</strong>
                {evidSafe && <div><a href={evidSafe} target="_blank" rel="noopener noreferrer">evidence</a></div>}
              </div>
            </Popup>
          </Marker>
        )
      })}
    </>
  )
}
