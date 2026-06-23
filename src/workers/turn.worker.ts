/// <reference lib="webworker" />
// Smart Turn v3 worker: semantic end-of-turn detection (WhisperForAudioClassification:
// Whisper-tiny encoder + linear END_OF_TURN head). transformers.js v4 has no Whisper
// audio-classification class, so we use it ONLY for the WhisperFeatureExtractor
// (80-mel input_features) and run the single model_quantized.onnx directly on
// onnxruntime-web (WASM/CPU, so it doesn't contend with the LLM for GPU).
// Everything streams from the HF Hub (CORS-clean) and caches to OPFS.

import * as ort from 'onnxruntime-web/wasm'
import { AutoProcessor, env, type Processor } from '@huggingface/transformers'
import { getModelAsset } from '../core/modelStore'
import { fmtBytes } from '../core/format'
import { wasmThreads } from '../core/runtime'
import { TURN, ORT_WASM_CDN } from '../models/registry'
import type { TurnIn, TurnOut } from '../protocol/messages'

const ctx = self as unknown as DedicatedWorkerGlobalScope
const post = (m: TurnOut): void => ctx.postMessage(m)

env.allowRemoteModels = true
ort.env.wasm.wasmPaths = ORT_WASM_CDN
ort.env.logLevel = 'error' // hide benign "node not assigned to preferred EP" warnings
ort.env.wasm.numThreads = wasmThreads()

// From preprocessor_config.json: chunk_length 8 s · sampling_rate 16 kHz ·
// nb_max_frames 800 · feature_size 80. The ONNX graph's `input_features` is fixed
// at [1, 80, 800] — Whisper's default extractor pads to 3000 (30 s), which would
// shape-fault the run, so we pin the feature length to 800 ourselves.
const MAX_SAMPLES = 8 * 16000
const NUM_MELS = 80
const TARGET_FRAMES = 800

type FeatureTensor = { data: Float32Array; dims: readonly number[] }
type CallableProcessor = (audio: Float32Array) => Promise<{ input_features: FeatureTensor }>

/** Force the mel features to exactly [1, NUM_MELS, TARGET_FRAMES] (right-pad/truncate). */
function pinFeatures(t: FeatureTensor): { data: Float32Array; dims: number[] } {
  const mels = t.dims[t.dims.length - 2] ?? NUM_MELS
  const frames = t.dims[t.dims.length - 1] ?? TARGET_FRAMES
  if (frames === TARGET_FRAMES) return { data: t.data, dims: [1, mels, TARGET_FRAMES] }
  const out = new Float32Array(mels * TARGET_FRAMES)
  const copy = Math.min(frames, TARGET_FRAMES)
  for (let m = 0; m < mels; m++) {
    for (let f = 0; f < copy; f++) out[m * TARGET_FRAMES + f] = t.data[m * frames + f]
  }
  return { data: out, dims: [1, mels, TARGET_FRAMES] }
}

let processor: Processor | null = null
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
  const onProgress = (p: { file?: string; loaded?: number; total?: number }): void => {
    if (p.loaded != null && p.total) {
      post({ kind: 'load', label: 'Turn', detail: `${p.file ?? 'config'} · ${fmtBytes(p.loaded)} / ${fmtBytes(p.total)}`, loaded: p.loaded, total: p.total })
    }
  }
  processor = await AutoProcessor.from_pretrained(TURN.hfModelId, { progress_callback: onProgress as never })

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
  if (!processor || !session) throw new Error('Turn model not initialized')
  const audio = samples.length > MAX_SAMPLES ? samples.slice(samples.length - MAX_SAMPLES) : samples

  const { input_features } = await (processor as unknown as CallableProcessor)(audio)
  const { data, dims } = pinFeatures(input_features)
  const feats = new ort.Tensor('float32', data, dims)
  const out = await session.run({ input_features: feats })
  // WhisperForAudioClassification → single END_OF_TURN logit (sigmoid).
  const logitsTensor = out.logits ?? out[session.outputNames[0]]
  const logit = Number((logitsTensor.data as Float32Array)[0])
  const prob = 1 / (1 + Math.exp(-logit)) // sigmoid → P(END_OF_TURN)
  console.info(`[aidekin] Turn analyze id=${id} feats=[${dims.join(',')}] logit=${logit.toFixed(3)} p=${prob.toFixed(3)}`)
  post({ kind: 'verdict', id, complete: prob > 0.5, prob })
}
