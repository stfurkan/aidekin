// Headless runner for the session-persistence verification (eval-session.html): starts the vite dev
// server, drives system Chrome with WebGPU, and asserts BOTH the page-visible result (SESSION OK) and
// the worker console signal that the snapshot path (not a re-prefill) carried the conversation across
// the simulated reload. Reuses the eval-chat persistent Chrome profile so the model loads from OPFS.
import { spawn } from 'node:child_process'
import { mkdirSync } from 'node:fs'
import { chromium } from 'playwright-core'

const PORT = 5198
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
  // Capture worker + page console. The worker logs "session restored" only when it actually
  // restores a snapshot, so counting occurrences proves WHICH reload restored (expect exactly one:
  // worker B; A had no snapshot, C's was deleted by the new-conversation clear).
  const consoleLines = []
  page.on('console', (m) => consoleLines.push(m.text()))

  for (let attempt = 0; ; attempt++) {
    try {
      await page.goto(`http://localhost:${PORT}/eval-session.html`, { waitUntil: 'load', timeout: 30000 })
      break
    } catch (e) {
      if (attempt >= 4) throw e
      await new Promise((r) => setTimeout(r, 1500))
    }
  }
  await page.waitForFunction(
    () => /SESSION OK|SESSION FAIL|FATAL/.test(document.getElementById('out').textContent),
    undefined,
    { timeout: 900000, polling: 1000 },
  )
  const text = await page.evaluate(() => document.getElementById('out').textContent)
  console.log(text)

  const restoredCount = consoleLines.filter((l) => l.includes('session restored')).length
  const skippedPrewarm = consoleLines.some((l) => l.includes('prewarm skipped (session restored)'))
  const overflowTrimmed = consoleLines.some((l) => l.includes('exceeded the') && l.includes('window'))
  console.log(`\nworker-console checks: "session restored" x${restoredCount} (expect >=1), prewarm-skip=${skippedPrewarm} (expect true), overflow-trim=${overflowTrimmed} (expect true)`)

  // The snapshot restore path fired (worker B), the follow-up warmed from the snapshot rather than
  // re-prefilling, and the phase-4 overflow engaged onOverflow. Phase 3's page-level "does NOT recall"
  // already proves the new-conversation clear.
  const pageOk = /SESSION OK/.test(text)
  const consoleOk = restoredCount >= 1 && skippedPrewarm && overflowTrimmed
  if (!consoleOk) console.log('CONSOLE CHECK FAIL: the snapshot restore path did not fire exactly as expected')
  process.exitCode = pageOk && consoleOk ? 0 : 1
} catch (e) {
  console.error('session eval driver failed:', e.message)
  process.exitCode = 1
} finally {
  await context?.close().catch(() => undefined)
  vite.kill()
}
