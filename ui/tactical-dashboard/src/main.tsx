import React from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import 'leaflet/dist/leaflet.css'

const rootEl = document.getElementById('root')!
createRoot(rootEl).render(<App />)

const style = document.createElement('style')
style.textContent = `
  html, body, #root { height: 100%; margin: 0; }
  .badge {
    position: absolute; top: 12px; right: 12px; z-index: 1000;
    background: rgba(0,0,0,0.72); color: #fff; padding: 6px 10px; border-radius: 6px;
    font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
  }
  .blue-icon { filter: hue-rotate(190deg) saturate(1.2); }
`

document.head.appendChild(style)
