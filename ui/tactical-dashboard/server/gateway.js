// Simple SSE -> WS gateway. Useful when browsers behind proxies block SSE.
const express = require('express')
const { WebSocketServer } = require('ws')
const EventSource = require('eventsource')

const app = express()
const port = process.env.GATEWAY_PORT || 3001
const sseUrl = process.env.SSE_URL || 'http://localhost:8000/events/stream'

const wss = new WebSocketServer({ noServer: true })
const server = app.listen(port, () => console.log(`[gateway] WS on :${port}`))

server.on('upgrade', (req, socket, head) => {
  wss.handleUpgrade(req, socket, head, (ws) => {
    wss.emit('connection', ws, req)
  })
})

wss.on('connection', (ws) => {
  const es = new EventSource(sseUrl)
  es.onmessage = (e) => ws.readyState === ws.OPEN && ws.send(e.data)
  es.onerror = () => { try { es.close() } catch {}
    try { ws.close() } catch {}
  }
  ws.on('close', () => { try { es.close() } catch {} })
})
