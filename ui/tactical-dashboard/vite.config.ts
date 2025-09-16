import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  server: {
    port: 5173,
    proxy: {
      '/events/stream': 'http://localhost:8000',
      '/ws': {
        target: 'ws://localhost:3001',
        ws: true,
        changeOrigin: true
      },
      '/route-engine': {
        target: 'http://localhost:8010',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/route-engine/, '')
      }
    }
  },
  plugins: [react()],
})
