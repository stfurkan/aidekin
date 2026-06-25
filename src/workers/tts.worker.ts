/// <reference lib="webworker" />
// TTS worker: Supertonic-3 (4-stage flow-matching) on onnxruntime-web. Loads via
// the OPFS cache, synthesizes one (sentence-sized) text per `speak`, and posts the
// 44.1 kHz PCM for gapless playback. English-only (<en> wrapper in the engine).

import * as ort from 'onnxruntime-web/webgpu'
import { getModelAsset } from '../core/modelStore'
import { fmtBytes } from '../core/format'
import { wasmThreads } from '../core/runtime'
import { TTS, ORT_WASM_CDN } from '../models/registry'
import {
  SupertonicTts,
  makeVoiceStyle,
  type SupertonicConfig,
  type SupertonicSessions,
} from '../tts/supertonic'
import type { Device, TtsIn, TtsOut } from '../protocol/messages'

const ctx = self as unknown as DedicatedWorkerGlobalScope
const post = (m: TtsOut, transfer: Transferable[] = []): void => ctx.postMessage(m, transfer)

ort.env.wasm.wasmPaths = ORT_WASM_CDN
ort.env.logLevel = 'error' // hide benign "node not assigned to preferred EP" warnings
ort.env.wasm.numThreads = wasmThreads()

let tts: SupertonicTts | null = null
const aborted = new Set<number>()

// Serialize all work: ort WebGPU sessions share ONE GPU command queue, so two
// synthesize() calls running concurrently deadlock it. We chain messages so each
// `speak` fully completes before the next starts. `abort` is handled immediately
// (just sets a flag) so a queued/in-flight synth can bail at its next checkpoint.
let chain: Promise<void> = Promise.resolve()
ctx.onmessage = (ev: MessageEvent<TtsIn>) => {
  const msg = ev.data
  if (msg.kind === 'abort') {
    aborted.add(msg.id)
    return
  }
  chain = chain.then(() => handle(msg))
}

async function handle(msg: TtsIn): Promise<void> {
  try {
    if (msg.kind === 'init') await init(msg.modelBase, msg.device)
    else if (msg.kind === 'speak') await speak(msg.id, msg.text)
    else if (msg.kind === 'abort') aborted.add(msg.id)
  } catch (err) {
    post({ kind: 'error', message: `TTS: ${(err as Error).message}` })
  }
}

async function loadAsset(base: string, rel: string): Promise<ArrayBuffer> {
  return getModelAsset(`tts/${rel}`, `${base}/${rel}`, (p) =>
    post({ kind: 'load', label: 'TTS', detail: `${rel} · ${fmtBytes(p.loaded)} / ${fmtBytes(p.total)}`, loaded: p.loaded, total: p.total }),
  )
}

async function init(base: string, device: Device): Promise<void> {
  const f = TTS.files
  const eps: ort.InferenceSession.ExecutionProviderConfig[] = device === 'webgpu' ? ['webgpu'] : ['wasm']
  const opts = { executionProviders: eps, graphOptimizationLevel: 'all' as const, logSeverityLevel: 3 as const }
  const dec = new TextDecoder()

  const [cfgBuf, idxBuf, styleBuf] = await Promise.all([
    loadAsset(base, f.config),
    loadAsset(base, f.unicodeIndexer),
    loadAsset(base, f.voiceStyle),
  ])
  const cfg = JSON.parse(dec.decode(cfgBuf)) as SupertonicConfig
  const indexer = JSON.parse(dec.decode(idxBuf)) as number[]
  const style = makeVoiceStyle(JSON.parse(dec.decode(styleBuf)))

  const mk = async (rel: string): Promise<ort.InferenceSession> =>
    ort.InferenceSession.create(new Uint8Array(await loadAsset(base, rel)), opts)

  // Sequential to bound peak memory (vector_estimator alone is ~257 MB).
  const sessions: SupertonicSessions = {
    durationPredictor: await mk(f.durationPredictor),
    textEncoder: await mk(f.textEncoder),
    vectorEstimator: await mk(f.vectorEstimator),
    vocoder: await mk(f.vocoder),
  }

  tts = new SupertonicTts(cfg, indexer, sessions, style)

  // Warm up the WebGPU pipeline at load time: the FIRST synthesize() otherwise pays a
  // one-time ~10 s shader-compile/allocation cost mid-conversation (a long silent gap
  // before the first reply). Run a throwaway short synth now so the first real reply
  // is fast. Best-effort - never block readiness on a warmup hiccup.
  try {
    const w0 = performance.now()
    await tts.synthesize('Hi.')
    console.info(`[aidekin] TTS warmup done in ${(performance.now() - w0).toFixed(0)}ms`)
  } catch (err) {
    console.warn('[aidekin] TTS warmup skipped:', (err as Error).message)
  }

  post({ kind: 'ready', info: `Supertonic TTS on ${device} @ ${tts.sampleRate} Hz` })
}

async function speak(id: number, text: string): Promise<void> {
  if (!tts) throw new Error('TTS not initialized')
  const clean = text.trim()
  if (aborted.has(id) || clean.length === 0) {
    aborted.delete(id)
    post({ kind: 'done', id })
    return
  }
  const t0 = performance.now()
  console.info(`[aidekin] TTS speak id=${id} text="${clean.slice(0, 60)}"`)
  const pcm = await tts.synthesize(clean)
  console.info(
    `[aidekin] TTS synth id=${id} → ${pcm.length} samples ` +
      `(${(pcm.length / tts.sampleRate).toFixed(2)}s) in ${(performance.now() - t0).toFixed(0)}ms`,
  )
  if (aborted.has(id)) {
    aborted.delete(id)
    post({ kind: 'done', id })
    return
  }
  post({ kind: 'audio', id, pcm, sampleRate: tts.sampleRate }, [pcm.buffer])
  post({ kind: 'done', id })
}
