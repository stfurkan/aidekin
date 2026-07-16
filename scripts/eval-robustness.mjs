// Headless runner for the multi-seed robustness probe (eval-robustness.html): sweeps the seed-fragile
// abstention/confabulation/guard scenarios across seeds and prints per-scenario pass rates. Reuses the
// eval Chrome profile (model already cached in OPFS). Exits nonzero if any borderline behavior drops
// below the floor, so an f16 -> q8 regression fails the gate instead of passing silently.
import { spawn } from 'node:child_process'
import { mkdirSync } from 'node:fs'
import { chromium } from 'playwright-core'

const PORT = 5196
const FLOOR = 80 // every borderline scenario must hold in >= 80% of seeds
const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
const PROFILE = new URL('../.cache/eval-profile/', import.meta.url).pathname

const vite = spawn('npx', ['vite', '--port', String(PORT), '--strictPort'], { stdio: ['ignore', 'pipe', 'pipe'] })
const viteReady = new Promise((resolve, reject) => {
  vite.stdout.on('data', (d) => String(d).includes('Local:') && resolve(undefined))
  vite.stderr.on('data', (d) => process.stderr.write(d))
  vite.on('exit', (code) => reject(new Error(`vite exited early (${code})`)))
  setTimeout(() => reject(new Error('vite did not start in 30s')), 30000)
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
  for (let attempt = 0; ; attempt++) {
    try {
      await page.goto(`http://localhost:${PORT}/eval-robustness.html`, { waitUntil: 'load', timeout: 30000 })
      break
    } catch (e) {
      if (attempt >= 4) throw e
      await new Promise((r) => setTimeout(r, 1500))
    }
  }
  await page.waitForFunction(() => /ROBUSTNESS DONE|FATAL/.test(document.getElementById('out').textContent), undefined, { timeout: 900000, polling: 1000 })
  const text = await page.evaluate(() => document.getElementById('out').textContent)
  console.log(text)
  const worst = Number((text.match(/WORST (\d+)%/) || [])[1])
  process.exitCode = /ROBUSTNESS DONE/.test(text) && worst >= FLOOR ? 0 : 1
} catch (e) {
  console.error('robustness driver failed:', e.message)
  process.exitCode = 1
} finally {
  await context?.close().catch(() => undefined)
  vite.kill()
}
