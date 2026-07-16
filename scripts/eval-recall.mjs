// Headless runner for the recall-robustness probe (eval-recall.html): runs the personal-recall turn
// across many seeds + two phrasings and prints the pass rate. Reuses the eval Chrome profile.
import { spawn } from 'node:child_process'
import { mkdirSync } from 'node:fs'
import { chromium } from 'playwright-core'

const PORT = 5195
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
      await page.goto(`http://localhost:${PORT}/eval-recall.html`, { waitUntil: 'load', timeout: 30000 })
      break
    } catch (e) {
      if (attempt >= 4) throw e
      await new Promise((r) => setTimeout(r, 1500))
    }
  }
  await page.waitForFunction(() => /RECALL DONE|FATAL/.test(document.getElementById('out').textContent), undefined, { timeout: 900000, polling: 1000 })
  console.log(await page.evaluate(() => document.getElementById('out').textContent))
} catch (e) {
  console.error('recall driver failed:', e.message)
  process.exitCode = 1
} finally {
  await context?.close().catch(() => undefined)
  vite.kill()
}
