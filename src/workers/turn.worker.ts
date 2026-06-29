/// <reference lib="webworker" />
// Smart Turn v3 worker: semantic end-of-turn detection (WhisperForAudioClassification:
// Whisper-tiny encoder + linear END_OF_TURN head). The 80-mel input_features are computed by our own
// WhisperFeatureExtractor reimplementation (turnFeatures.ts, no transformers.js), and the single
// model_quantized.onnx runs directly on onnxruntime-web (WASM/CPU, so it doesn't contend with the LLM
// for GPU). Everything streams from the HF Hub (CORS-clean) and caches to OPFS.

import * as ort from 'onnxruntime-web/wasm'
import { getModelAsset } from '../core/modelStore'
import { fmtBytes } from '../core/format'
import { wasmThreads } from '../core/runtime'
import { TURN, ORT_WASM_CDN } from '../models/registry'
import type { TurnIn, TurnOut } from '../protocol/messages'
import { whisperFeatures } from './turnFeatures'

const ctx = self as unknown as DedicatedWorkerGlobalScope
const post = (m: TurnOut): void => ctx.postMessage(m)

ort.env.wasm.wasmPaths = ORT_WASM_CDN
ort.env.logLevel = 'error' // hide benign "node not assigned to preferred EP" warnings
ort.env.wasm.numThreads = wasmThreads()

// chunk_length 8 s · 16 kHz · 80 mels · 800 frames -> the ONNX input_features is fixed at [1, 80, 800].
const MAX_SAMPLES = 8 * 16000

let session: ort.InferenceSession | null = null

ctx.onmessage = (ev: MessageEvent<TurnIn>) => {
  void handle(ev.data)
}

async function handle(msg: TurnIn): Promise<void> {
  try {
    if (msg.kind === 'init') await init()
    else if (msg.kind === 'analyze') await analyze(msg.id, msg.samples)
  } catch (err) {
    post({ kind: 'error', message: `Turn: ${(err as Error).message}` })
  }
}

async function init(): Promise<void> {
  const url = `https://huggingface.co/${TURN.hfModelId}/resolve/main/onnx/model_quantized.onnx`
  const buf = await getModelAsset('turn/model_quantized.onnx', url, (p) =>
    post({ kind: 'load', label: 'Turn', detail: `model_quantized.onnx · ${fmtBytes(p.loaded)} / ${fmtBytes(p.total)}`, loaded: p.loaded, total: p.total }),
  )
  session = await ort.InferenceSession.create(new Uint8Array(buf), {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
    logSeverityLevel: 3,
  })
  post({ kind: 'ready', info: 'Smart Turn v3 (wasm)' })
}

async function analyze(id: number, samples: Float32Array): Promise<void> {
  if (!session) throw new Error('Turn model not initialized')
  const audio = samples.length > MAX_SAMPLES ? samples.slice(samples.length - MAX_SAMPLES) : samples

  const data = whisperFeatures(audio) // [80*800]
  const feats = new ort.Tensor('float32', data, [1, 80, 800])
  const out = await session.run({ input_features: feats })
  // WhisperForAudioClassification → single END_OF_TURN logit (sigmoid).
  const logitsTensor = out.logits ?? out[session.outputNames[0]]
  const logit = Number((logitsTensor.data as Float32Array)[0])
  const prob = 1 / (1 + Math.exp(-logit)) // sigmoid → P(END_OF_TURN)
  console.info(`[aidekin] Turn analyze id=${id} feats=[1,80,800] logit=${logit.toFixed(3)} p=${prob.toFixed(3)}`)
  post({ kind: 'verdict', id, complete: prob > 0.5, prob })
}
