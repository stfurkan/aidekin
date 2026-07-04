// Device diagnostics page (TEMPORARY, remove after mobile testing). Measures the real
// production LLM path on whatever device opens it: adapter identity and features (subgroup
// sizes, shader-f16), engine kernel path, load time, prefill and decode speed, greedy
// bit-exactness vs the committed known-good ids, and memory behavior. Uses the exact
// production model URLs and the same OPFS cache as the widget. A sessionStorage breadcrumb
// survives an OS page kill, so a crash reports WHERE it died on the next open.
import { createEngine, WebGPUUnavailableError, type Engine } from 'bitgpu'
import { getModelAssetStream } from '@/core/modelStore'
import { LLM, llmMaxSeqLen, llmModelUrls } from '@/models/registry'

const out = document.getElementById('out')!
const t0 = performance.now()
const log = (s: string, cls?: string): void => {
  const line = `[${((performance.now() - t0) / 1000).toFixed(1).padStart(6)}s] ${s}`
  if (cls) {
    const span = document.createElement('span')
    span.className = cls
    span.textContent = line + '\n'
    out.appendChild(span)
  } else out.appendChild(document.createTextNode(line + '\n'))
  window.scrollTo(0, document.body.scrollHeight)
}
const pass = (ok: boolean, s: string): void => log(`${ok ? '[PASS]' : '[FAIL]'} ${s}`, ok ? 'pass' : 'fail')
const mb = (n: number): string => `${(n / 1048576).toFixed(0)}MB`

// Survives an OOM page kill: if the page dies mid-run, the next open names the fatal stage.
const STAGE_KEY = 'aidekin-diag-stage'
const setStage = (s: string): void => sessionStorage.setItem(STAGE_KEY, s)
const prior = sessionStorage.getItem(STAGE_KEY)
if (prior && prior !== 'done') log(`NOTE: the previous run ended during "${prior}" without finishing - the OS likely killed the page there (out of memory).`, 'fail')

const memLine = (): string => {
  const m = (performance as unknown as { memory?: { usedJSHeapSize: number; jsHeapSizeLimit: number } }).memory
  return m ? `js-heap ${mb(m.usedJSHeapSize)} / limit ${mb(m.jsHeapSizeLimit)}` : 'js-heap n/a (non-Chromium)'
}

// The verify-gate fixture prompt ("The capital of France is Paris. ...") and its known-good
// greedy continuation: bit-exact on every correct engine path, subgroup or fallback.
const IDS = [785, 6722, 315, 9625, 374, 12095, 13, 576, 6722, 315, 6323, 374]
const KNOWN_GOOD = [26194, 13, 576, 6722, 315, 279, 3639, 4180, 374, 6515, 11, 422, 727, 13, 576, 6722, 315, 279, 3639, 15072, 374, 7148, 13, 576, 6722, 315, 279, 25662, 374, 37741, 13, 576]
// Product sampler settings (mirrors SAMPLING in llm.worker.ts).
const SAMPLING = { temperature: 0.3, topK: 20, topP: 0.85, repetitionPenalty: 1.15 }

window.addEventListener('error', (e) => log(`window error: ${e.message}`, 'fail'))
window.addEventListener('unhandledrejection', (e) => log(`unhandled rejection: ${(e.reason as Error)?.message ?? e.reason}`, 'fail'))

async function run(): Promise<void> {
  out.textContent = ''
  let engine: Engine | null = null
  try {
    setStage('environment')
    log(`ua: ${navigator.userAgent}`)
    log(`cores ${navigator.hardwareConcurrency}, deviceMemory ${(navigator as unknown as { deviceMemory?: number }).deviceMemory ?? 'n/a'}GB, crossOriginIsolated ${crossOriginIsolated}, ${memLine()}`)

    setStage('adapter')
    if (!navigator.gpu) {
      log('navigator.gpu MISSING - no WebGPU in this browser', 'fail')
      setStage('done')
      return
    }
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
    if (!adapter) {
      log('requestAdapter returned null - WebGPU present but no usable GPU', 'fail')
      setStage('done')
      return
    }
    const info = adapter.info
    log(`adapter: vendor=${info.vendor || '?'} arch=${info.architecture || '?'} device=${info.device || '?'} desc=${info.description || '?'}`)
    log(`subgroups: min=${info.subgroupMinSize ?? '?'} max=${info.subgroupMaxSize ?? '?'}`)
    for (const f of ['subgroups', 'shader-f16', 'timestamp-query']) log(`feature ${f}: ${adapter.features.has(f as GPUFeatureName) ? 'yes' : 'NO'}`)
    log(`limits: maxBufferSize ${mb(adapter.limits.maxBufferSize)}, maxStorageBufferBindingSize ${mb(adapter.limits.maxStorageBufferBindingSize)}, wgStorage ${adapter.limits.maxComputeWorkgroupStorageSize}B`)

    setStage('model load')
    log(`\nloading model (production URLs, OPFS-cached, maxSeqLen ${llmMaxSeqLen()})...`)
    const urls = llmModelUrls()
    let lastPct = -10
    const progress = (p: { loaded: number; total?: number }): void => {
      const pct = Math.round((100 * p.loaded) / (p.total || 1))
      if (pct >= lastPct + 10) {
        lastPct = pct
        log(`  weights ${pct}% (${mb(p.loaded)})`)
      }
    }
    // Same streaming path as the widget: OPFS-cached chunks flow straight into GPU buffers.
    const fetchStream = async (url: string): Promise<ReadableStream<Uint8Array>> => {
      if (url === urls.dataUrl) return getModelAssetStream('llm-bonsai-1.7b-q1', url, progress)
      const res = await fetch(url)
      if (!res.ok) throw new Error(`fetch ${url} failed: HTTP ${res.status}`)
      return res.body as ReadableStream<Uint8Array>
    }
    const fetchArrayBuffer = async (url: string): Promise<ArrayBuffer> => {
      const res = await fetch(url)
      if (!res.ok) throw new Error(`fetch ${url} failed: HTTP ${res.status}`)
      return res.arrayBuffer()
    }
    const tLoad = performance.now()
    engine = await createEngine({ ...urls, maxSeqLen: llmMaxSeqLen(), kvCache: LLM.kvCache, fetchStream, fetchArrayBuffer, onProgress: (p) => log(`  [${p.phase}]`) })
    log(`engine ready in ${((performance.now() - tLoad) / 1000).toFixed(1)}s, ${memLine()}`)
    const cap = engine.capabilities
    log(`kernel path: ${cap.useSubgroups ? `subgroups SG=${cap.subgroupSize}` : 'WORKGROUP FALLBACK (no uniform SG32/64)'}, kv cache ${cap.kvCache}`, cap.useSubgroups ? 'pass' : undefined)
    void engine.lost.then((l) => {
      if (l.reason !== 'destroyed') log(`GPU DEVICE LOST: ${l.reason} ${l.message}`, 'fail')
    })

    setStage('greedy decode')
    log('\ngreedy 32 tokens (short prompt)...')
    const g = await engine.generate(IDS, { maxTokens: 32 })
    const exact = g.tokens.length === KNOWN_GOOD.length && g.tokens.every((t, i) => t === KNOWN_GOOD[i])
    pass(exact, `bit-exact vs known-good ids${exact ? '' : ` (got [${g.tokens.slice(0, 8).join(',')}...])`}`)
    log(`ttft ${g.prefillMs.toFixed(0)}ms, decode ${g.tokensPerSecond.toFixed(2)} tok/s (record ${g.timing.recordMs.toFixed(1)} gpu ${g.timing.gpuMs.toFixed(1)} readback ${g.timing.readbackMs.toFixed(1)} ms/tok)`)

    setStage('sampled decode')
    log('\nsampled 32 tokens (product sampler)...')
    engine.resetCache()
    const s = await engine.generate(IDS, { ...SAMPLING, maxTokens: 32 })
    log(`sampled ${s.tokensPerSecond.toFixed(2)} tok/s`)

    setStage('long prefill 512')
    log('\nprefill 512 tokens (approximates a real system+context prompt)...')
    engine.resetCache()
    const longIds: number[] = []
    while (longIds.length < 512) longIds.push(...IDS)
    const lp = await engine.generate(longIds.slice(0, 512), { maxTokens: 2 })
    log(`prefill(512) ${(lp.prefillMs / 1000).toFixed(1)}s (${(lp.prefillMs / 512).toFixed(1)} ms/tok -> ~${((lp.prefillMs / 512) * 1100 / 1000).toFixed(1)}s for a 1100-token first turn)`)
    log(`${memLine()}`)

    setStage('kv growth')
    log('\ncross-turn reuse + KV growth (cache grows past the 512 floor)...')
    const r = await engine.generate(IDS.slice(0, 6), { maxTokens: 8, reuseCache: true })
    pass(r.tokens.length === 8, `reuse turn generated (ttft ${r.prefillMs.toFixed(0)}ms)`)

    log(`\nDIAG DONE - ${cap.useSubgroups ? `SG=${cap.subgroupSize}` : 'fallback'} greedy ${g.tokensPerSecond.toFixed(2)} / sampled ${s.tokensPerSecond.toFixed(2)} tok/s, ${memLine()}`, 'pass')
    setStage('done')
  } catch (e) {
    if (e instanceof WebGPUUnavailableError) log(`WebGPU unavailable: ${e.message}`, 'fail')
    else log(`ERROR: ${(e as Error).message}\n${(e as Error).stack ?? ''}`, 'fail')
    setStage('done') // reached the handler, so it was an error, not an OS kill
  } finally {
    engine?.dispose()
  }
}

document.getElementById('run')!.onclick = () => void run()
document.getElementById('copy')!.onclick = () => {
  void navigator.clipboard.writeText(out.textContent ?? '').then(
    () => log('log copied to clipboard'),
    () => log('clipboard copy failed - long-press to select instead', 'fail'),
  )
}
