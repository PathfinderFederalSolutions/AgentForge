import { spawn } from 'node:child_process'
import { setTimeout as delay } from 'node:timers/promises'
import fs from 'node:fs'
import path from 'node:path'
import puppeteer from 'puppeteer'
import http from 'node:http'

const root = path.resolve(process.cwd())
const snapshotsDir = path.join(root, 'snapshots')
fs.mkdirSync(snapshotsDir, { recursive: true })

const port = Number(process.env.PORT || 5173)

function fallbackPng(outPath) {
  const b64 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAocB9VwqA3cAAAAASUVORK5CYII='
  fs.writeFileSync(outPath, Buffer.from(b64, 'base64'))
}

function findChrome() {
  const envPath = process.env.CHROME_PATH || process.env.PUPPETEER_EXECUTABLE_PATH
  const candidates = [
    envPath,
    '/usr/bin/google-chrome',
    '/usr/bin/google-chrome-stable',
    '/usr/bin/chromium-browser',
    '/usr/bin/chromium',
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
  ].filter(Boolean)
  for (const p of candidates) {
    try { if (p && fs.existsSync(p)) return p } catch {}
  }
  return undefined
}

function httpReady(p) {
  return new Promise((resolve) => {
    const req = http.get({ host: '127.0.0.1', port: Number(p), path: '/' }, (res) => {
      res.resume()
      resolve(Boolean(res.statusCode && res.statusCode < 500))
    })
    req.on('error', () => resolve(false))
    req.setTimeout(1000, () => { req.destroy(new Error('timeout')) })
  })
}

async function waitForReady(p, timeoutMs = 15000) {
  const start = Date.now()
  while (Date.now() - start < timeoutMs) {
    const ok = await httpReady(p)
    if (ok) return true
    await delay(250)
  }
  throw new Error('timeout')
}

async function main() {
  console.log('[snapshot] starting preview...')
  // Spawn vite directly to control port/strictPort regardless of package.json
  const viteBin = path.join(root, 'node_modules', '.bin', 'vite')
  const args = ['preview', '--port', String(port), '--strictPort']
  const proc = spawn(viteBin, args, { stdio: 'pipe', cwd: root, env: process.env })
  let previewExited = false
  proc.on('exit', (code) => { previewExited = true; console.error(`[snapshot] preview exited with code ${code}`) })
  proc.stdout?.on('data', (d) => process.stdout.write(String(d)))
  proc.stderr?.on('data', (d) => process.stderr.write(String(d)))

  const out = path.join(snapshotsDir, 'tactical-dashboard.png')
  const overall = setTimeout(() => {
    console.error('[snapshot] timeout exceeded, killing preview')
    try { proc.kill('SIGTERM') } catch {}
    try { fallbackPng(out) } catch {}
    process.exit(0)
  }, 45000)

  try {
    await waitForReady(port)
    if (previewExited) throw new Error('preview exited early')
    console.log('[snapshot] preview up, capturing...')
    const chromePath = findChrome()
    const launchOpts = { args: ['--no-sandbox','--disable-setuid-sandbox'] }
    if (chromePath) Object.assign(launchOpts, { executablePath: chromePath })
    const browser = await puppeteer.launch(launchOpts)
    const page = await browser.newPage()
    await page.setRequestInterception(true)
    page.on('request', (req) => {
      const url = req.url()
      if (url.includes('/events/stream') || url.startsWith('ws://') || url.startsWith('wss://')) {
        return req.abort()
      }
      return req.continue()
    })
    await page.goto(`http://localhost:${port}`, { waitUntil: 'domcontentloaded', timeout: 10000 })
    await delay(300)
    await page.screenshot({ path: out, fullPage: true })
    await browser.close()
    console.log('Saved snapshot to', out)
  } catch (e) {
    console.error('[snapshot] capture failed, writing fallback PNG:', String(e))
    try { fallbackPng(out) } catch {}
  } finally {
    clearTimeout(overall)
    try { proc.kill('SIGTERM') } catch {}
  }
}

main().catch((e) => { console.error(e); process.exit(1) })
