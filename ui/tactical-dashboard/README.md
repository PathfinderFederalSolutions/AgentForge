Tactical Dashboard (React + Leaflet)

Status: scaffold. Consumes SSE from /events/stream.

Commands:
- pnpm install
- pnpm dev
- pnpm build
- pnpm snapshot

Usage:
- pnpm install
- pnpm dev  # proxies /events/stream to http://localhost:8000
- pnpm gateway  # optional SSE→WS gateway at ws://localhost:3001

Acceptance:
- Consumes /events/stream locally
- Lighthouse perf >= 80, no XSS (basic CSP set)
- Evidence link opens read-only drawer
- Snapshots saved under snapshots/

Artifacts:
- snapshots/*.png
