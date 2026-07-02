// Headless runner for the behavioral golden-set eval (eval.html): starts the vite dev server,
// drives system Chrome with WebGPU against it, prints the transcript, exits nonzero on failure.
// A persistent Chrome profile keeps OPFS across runs, so the model caches once (~290MB copies
// from public/llm on the first run; later runs load in seconds).
import { spawn } from 'node:child_process'
import { mkdirSync } from 'node:fs'
import { chromium } from 'playwright-core'

const PORT = 5199
const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
const PROFILE = new URL('../.cache/eval-profile/', import.meta.url).pathname

const vite = spawn('npx', ['vite', '--port', String(PORT), '--strictPort'], { stdio: ['ignore', 'pipe', 'pipe'] })
const viteReady = new Promise((resolve, reject) => {
  vite.stdout.on('data', (d) => {
    if (String(d).includes('Local:')) resolve(undefined)
  })
  vite.stderr.on('data', (d) => process.stderr.write(d))
  vite.on('exit', (code) => reject(new Error(`vite exited early (${code})`)))
  setTimeout(() => reject(new Error('vite dev server did not start in 30s')), 30000)
})

let context
try {
  await viteReady
  mkdirSync(PROFILE, { recursive: true })
  context = await chromium.launchPersistentContext(PROFILE, {
    executablePath: CHROME,
    headless: true,
    args: ['--enable-unsafe-webgpu', '--enable-gpu', '--use-angle=metal'],
  })
  const page = await context.newPage()
  page.on('pageerror', (e) => console.log('[pageerror]', e.message))
  // localhost, not 127.0.0.1: vite may bind the IPv6 loopback only
  for (let attempt = 0; ; attempt++) {
    try {
      await page.goto(`http://localhost:${PORT}/eval.html`, { waitUntil: 'load', timeout: 30000 })
      break
    } catch (e) {
      if (attempt >= 4) throw e
      await new Promise((r) => setTimeout(r, 1500))
    }
  }
  await page.waitForFunction(
    () => /EVAL OK|EVAL FAIL|FATAL/.test(document.getElementById('out').textContent),
    undefined,
    { timeout: 900000, polling: 1000 },
  )
  const text = await page.evaluate(() => document.getElementById('out').textContent)
  console.log(text)
  process.exitCode = /EVAL OK/.test(text) ? 0 : 1
} catch (e) {
  console.error('eval driver failed:', e.message)
  process.exitCode = 1
} finally {
  await context?.close().catch(() => undefined)
  vite.kill()
}
