/// <reference lib="webworker" />
// ASR worker: Nemotron 3.5 streaming (FP16 export) on onnxruntime-web - the ONE
// engine. The heavy FastConformer encoder runs on WebGPU (real-time); decoder/joint
// run on WASM. Weights stream from the HF CDN (or VITE_MODEL_CDN) and cache to OPFS
// via modelStore; external .onnx.data blobs are supplied to the session explicitly.
// Buffers incoming mic audio into 320 ms windows, emits streaming partials + a final.

import * as ort from 'onnxruntime-web/webgpu'
import { SoniqoStreamingAsr, soniqoConfig } from '../asr/soniqoAsr'
import { NemotronDetokenizer } from '../asr/detokenizer'
import type { OrtSession, OrtTensor, TensorCtor, TensorData } from '../asr/ortTypes'
import { getModelAsset } from '../core/modelStore'
import { fmtBytes } from '../core/format'
import { ASR, ORT_WASM_CDN } from '../models/registry'
import type { AsrIn, AsrOut, Device } from '../protocol/messages'

const ctx = self as unknown as DedicatedWorkerGlobalScope
const post = (m: AsrOut, transfer: Transferable[] = []): void => ctx.postMessage(m, transfer)

ort.env.wasm.wasmPaths = ORT_WASM_CDN
ort.env.logLevel = 'error' // hide benign "node not assigned to preferred EP" warnings

const tensorCtor: TensorCtor = (type, data, dims) =>
  new ort.Tensor(type as 'float32', data as Float32Array, dims) as unknown as OrtTensor

let asr: SoniqoStreamingAsr | null = null
let detok: NemotronDetokenizer | null = null
let tokenIds: number[] = []
// Per-turn diagnostics (reset each turn): peak amplitude, total samples, start time.
let streamPeak = 0
let streamLen = 0
let streamT0 = 0

// Serialize all work onto one chain - ort sessions can't run concurrently, and a
// late transcribe (after a barge-in) must not overlap the previous one.
let chain: Promise<void> = Promise.resolve()
ctx.onmessage = (ev: MessageEvent<AsrIn>) => {
  chain = chain.then(() => handle(ev.data))
}

// English-only: the multilingual encoder is always driven with the English mask.
const EN_LANG_ID = ASR.langId.en

async function handle(msg: AsrIn): Promise<void> {
  try {
    if (msg.kind === 'init') await init(msg.modelBase, msg.device)
    else if (msg.kind === 'prefetch') await prefetch(msg.modelBase)
    else if (msg.kind === 'chunk') await onChunk(msg.id, msg.samples)
    else if (msg.kind === 'flush') await onFlush(msg.id)
    else if (msg.kind === 'reset') resetStream()
  } catch (err) {
    post({ kind: 'error', message: `ASR: ${(err as Error).message}` })
  }
}

async function loadAsset(base: string, rel: string): Promise<ArrayBuffer> {
  return getModelAsset(`asr/${rel}`, `${base}/${rel}`, (p) =>
    post({ kind: 'load', label: 'ASR', file: rel, detail: `${rel} · ${fmtBytes(p.loaded)} / ${fmtBytes(p.total)}`, loaded: p.loaded, total: p.total }),
  )
}

async function createSession(
  base: string,
  onnxRel: string,
  dataRel: string,
  eps: ort.InferenceSession.ExecutionProviderConfig[],
): Promise<ort.InferenceSession> {
  const [model, data] = await Promise.all([loadAsset(base, onnxRel), loadAsset(base, dataRel)])
  return ort.InferenceSession.create(new Uint8Array(model), {
    executionProviders: eps,
    graphOptimizationLevel: 'all',
    logSeverityLevel: 3, // error only - hide the benign constant-fold / EP-assignment warnings
    externalData: [{ path: dataRel.split('/').pop() as string, data: new Uint8Array(data) }],
  })
}

const wrap = (s: ort.InferenceSession): OrtSession => ({
  run: (feeds) => s.run(feeds as Record<string, ort.Tensor>) as Promise<Record<string, OrtTensor>>,
})

// Warm the OPFS cache for EVERY ASR weight in parallel (network-bound; each file streams
// straight to disk, so this does not raise peak memory). init() then reads them from cache,
// keeping the GPU/WASM session creation serial while the download is parallelized across files -
// and, run alongside the TTS worker's prefetch, across models too.
async function prefetch(base: string): Promise<void> {
  const f = ASR.files
  await Promise.all([
    loadAsset(base, f.vocab),
    loadAsset(base, f.decoder),
    loadAsset(base, f.decoderData),
    loadAsset(base, f.joiner),
    loadAsset(base, f.joinerData),
    loadAsset(base, f.encoder),
    loadAsset(base, f.encoderData),
  ])
  post({ kind: 'prefetched' })
}

// FP16 Nemotron: encoder on WebGPU (real-time), decoder/joint on WASM. WebGPU is
// required (as it is for the LLM); if the FP16 encoder self-test fails we surface a
// clear error rather than silently degrading - there is no second engine.
async function init(base: string, device: Device): Promise<void> {
  const f = ASR.files
  const vocabBuf = await loadAsset(base, f.vocab)
  // vocab.json is a {id: piece} map; adapt to the detokenizer's [piece, score][] vocab.
  const vocabObj = JSON.parse(new TextDecoder().decode(vocabBuf)) as Record<string, string>
  const pieces: [string, number][] = []
  for (const [k, v] of Object.entries(vocabObj)) pieces[Number(k)] = [v, 0]
  for (let i = 0; i < pieces.length; i++) if (!pieces[i]) pieces[i] = ['', 0]
  detok = new NemotronDetokenizer({ model: { vocab: pieces } })

  const cfg = soniqoConfig(ASR.contract)
  // decoder/joint always WASM (tiny); only the heavy encoder is a WebGPU candidate.
  const decoder = await createSession(base, f.decoder, f.decoderData, ['wasm'])
  const joint = await createSession(base, f.joiner, f.joinerData, ['wasm'])
  const mkEngine = (enc: ort.InferenceSession): SoniqoStreamingAsr =>
    new SoniqoStreamingAsr({ encoder: wrap(enc), decoder: wrap(decoder), joint: wrap(joint) }, tensorCtor, cfg)

  // WASM reference encoder (this matches the validated headless path exactly). Time a
  // warmed run so we can compare WebGPU's per-window cost against it below.
  const encWasm = await createSession(base, f.encoder, f.encoderData, ['wasm'])
  const wasmEngine = mkEngine(encWasm)
  await wasmEngine.selfTest() // warmup
  const tW = performance.now()
  const ref = await wasmEngine.selfTest()
  const wasmMs = performance.now() - tW
  console.info(`[aidekin] ASR encoder WASM self-test: ${ref.reason} (${wasmMs.toFixed(0)}ms/window)`)
  if (!ref.ok) throw new Error(`ASR encoder self-test failed (${ref.reason})`)

  if (device !== 'webgpu') {
    asr = mkEngine(encWasm)
    await finishInit('Nemotron ASR (fp16 · wasm)')
    return
  }

  // WebGPU candidate - VERIFY it (a) agrees with WASM and (b) is actually FASTER. Some
  // GPUs (notably Apple) run ort-web's WebGPU FP16 kernels incorrectly or fall back to
  // CPU op-by-op, producing output that is slow AND/OR wrong (blank) yet still passes a
  // NaN/variance check. If WebGPU either disagrees or isn't faster than WASM, use WASM.
  const refOut = ref.output
  await encWasm.release()
  const encGpu = await createSession(base, f.encoder, f.encoderData, ['webgpu', 'wasm'])
  const gpuEngine = mkEngine(encGpu)
  await gpuEngine.selfTest() // warmup (compiles WebGPU shaders - slow first call)
  const tG = performance.now()
  const gpu = await gpuEngine.selfTest()
  const gpuMs = performance.now() - tG
  const sim = cosineSim(gpu.output, refOut)
  console.info(`[aidekin] ASR encoder WebGPU↔WASM: cos=${sim.toFixed(4)} · gpu=${gpuMs.toFixed(0)}ms wasm=${wasmMs.toFixed(0)}ms/window · ${gpu.reason}`)

  if (gpu.ok && sim > 0.9 && gpuMs < wasmMs) {
    asr = mkEngine(encGpu)
    await finishInit('Nemotron ASR (fp16 · webgpu encoder)')
    return
  }

  const why = sim <= 0.9 ? `wrong output (cos=${sim.toFixed(3)})` : `not faster (${gpuMs.toFixed(0)}≥${wasmMs.toFixed(0)}ms - CPU fallback)`
  console.warn(`[aidekin] ⚠️ WebGPU encoder unreliable on this GPU: ${why}. Using the WASM encoder (correct${gpuMs < wasmMs ? '' : ', and not slower'}).`)
  await encGpu.release()
  asr = mkEngine(await createSession(base, f.encoder, f.encoderData, ['wasm']))
  await finishInit('Nemotron ASR (fp16 · wasm - WebGPU unreliable here)')
}

// Warm the full streaming path (encoder + decoder + joint) on the CHOSEN engine, then
// reset, then announce ready - so the first real utterance isn't a cold start.
async function finishInit(info: string): Promise<void> {
  if (asr) {
    const t = performance.now()
    try {
      await asr.warmup(EN_LANG_ID)
      console.info(`[aidekin] ASR full-path warmup ${(performance.now() - t).toFixed(0)}ms`)
    } catch (e) {
      console.warn('[aidekin] ASR warmup failed (non-fatal):', e)
    }
  }
  resetStream()
  post({ kind: 'ready', info })
}

/** Cosine similarity between two encoder-output vectors (EP correctness check). */
function cosineSim(a: Float32Array, b: Float32Array): number {
  const n = Math.min(a.length, b.length)
  let dot = 0
  let na = 0
  let nb = 0
  for (let i = 0; i < n; i++) {
    dot += a[i] * b[i]
    na += a[i] * a[i]
    nb += b[i] * b[i]
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb)
  return denom > 0 ? dot / denom : 0
}

// Live streaming: feed mic frames to the engine, which decodes whole windows as they
// complete and returns their tokens; emit a partial transcript on each.
async function onChunk(id: number, samples: Float32Array): Promise<void> {
  if (!asr || !detok) return

  // Running audio-health diagnostic (logged at flush): healthy speech peaks 0.1-1.0.
  if (streamLen === 0) streamT0 = performance.now()
  for (let i = 0; i < samples.length; i++) {
    const a = samples[i] < 0 ? -samples[i] : samples[i]
    if (a > streamPeak) streamPeak = a
  }
  streamLen += samples.length

  const ids = await asr.pushAudio(samples, EN_LANG_ID)
  if (ids.length > 0) {
    tokenIds.push(...ids)
    post({ kind: 'partial', id, text: detok.decode(tokenIds) })
  }
}

// Turn end: flush the trailing window(s) for the final transcript, then reset.
async function onFlush(id: number): Promise<void> {
  if (!asr || !detok) return
  const ids = await asr.endStream(EN_LANG_ID)
  if (ids.length > 0) tokenIds.push(...ids)
  const text = detok.decode(tokenIds)
  console.info(
    `[aidekin] ASR final id=${id} ${(streamLen / 16000).toFixed(2)}s ` +
      `peak=${streamPeak.toFixed(3)} → "${text}" (${(performance.now() - streamT0).toFixed(0)}ms)`,
  )
  if (streamPeak > 0.98) {
    console.warn(`[aidekin] ⚠️ mic clipping (peak=${streamPeak.toFixed(3)}) - lower system input gain; ASR accuracy is degraded by clipping`)
  }
  post({ kind: 'final', id, text })
  resetStream()
}

function resetStream(): void {
  tokenIds = []
  streamPeak = 0
  streamLen = 0
  streamT0 = 0
  asr?.reset()
}

// Touch the TensorData import so it is retained for downstream typing.
export type { TensorData }
