/// <reference lib="webworker" />
// VAD worker: Silero v5 speech/silence gate via @ricky0123/vad-web's FrameProcessor
// (tested onset/offset state machine: redemption hangover, pre-speech padding,
// min-speech misfire). Runs on WASM/CPU so it doesn't contend with the LLM for GPU.
// Fed 512-sample @16 kHz frames from the mic worklet (Silero v5's native frame size).

import * as ort from 'onnxruntime-web/wasm'
// Import only the DOM-free submodules - the package index pulls in MicVAD /
// asset-path code that touches `document`, which throws inside a Web Worker.
import { FrameProcessor, type FrameProcessorOptions } from '@ricky0123/vad-web/dist/frame-processor'
import { Message } from '@ricky0123/vad-web/dist/messages'
import { SileroV5 } from '@ricky0123/vad-web/dist/models'
import { getModelAsset } from '../core/modelStore'
import { fmtBytes } from '../core/format'
import { wasmThreads } from '../core/runtime'
import { ORT_WASM_CDN } from '../models/registry'
import type { VadIn, VadOut } from '../protocol/messages'

const ctx = self as unknown as DedicatedWorkerGlobalScope
const post = (m: VadOut, transfer: Transferable[] = []): void => ctx.postMessage(m, transfer)

ort.env.wasm.wasmPaths = ORT_WASM_CDN
ort.env.logLevel = 'error' // hide benign "node not assigned to preferred EP" warnings
ort.env.wasm.numThreads = wasmThreads()

let fp: FrameProcessor | null = null

// Serialize all work onto one promise chain (mirrors asr.worker.ts). Mic frames
// arrive every ~32 ms; Silero v5 WASM inference can take longer than that while the
// CPU is busy with ASR/TTS, so unserialized handlers overlap. The hazard isn't the
// FrameProcessor's counters (those mutate synchronously after its single await) - it's
// that Silero is a stateful RNN that reads → awaits session.run → writes its hidden
// state, and ort sessions aren't safe to run concurrently. Overlap corrupts that state
// and the probability stream, so `speech-end` (31 consecutive sub-threshold frames)
// never completes and the turn hangs on "Listening". One chain = no overlap.
let chain: Promise<void> = Promise.resolve()
ctx.onmessage = (ev: MessageEvent<VadIn>) => {
  chain = chain.then(() => handle(ev.data))
}

async function handle(msg: VadIn): Promise<void> {
  try {
    if (msg.kind === 'init') await init(msg.assetBase)
    else if (msg.kind === 'frame') await onFrame(msg.samples)
    else if (msg.kind === 'reset') fp?.reset()
  } catch (err) {
    post({ kind: 'error', message: `VAD: ${(err as Error).message}` })
  }
}

async function init(assetBase: string): Promise<void> {
  const modelFetcher = async (): Promise<ArrayBuffer> =>
    getModelAsset('vad/silero_vad_v5.onnx', `${assetBase}/silero_vad_v5.onnx`, (p) =>
      post({ kind: 'load', label: 'VAD', detail: `silero_vad_v5.onnx · ${fmtBytes(p.loaded)} / ${fmtBytes(p.total)}`, loaded: p.loaded, total: p.total }),
    )

  const model = await SileroV5.new(ort, modelFetcher)
  // Silero v5 real-time defaults (avoids importing getDefaultRealTimeVADOptions,
  // which lives in the DOM-dependent real-time-vad module).
  const options: FrameProcessorOptions = {
    positiveSpeechThreshold: 0.5,
    negativeSpeechThreshold: 0.35,
    // ~1 s of silence before declaring end-of-speech, so natural mid-sentence pauses
    // ("Where can I find you… what's your status") don't fragment one turn into many
    // or trigger a premature reply. Smart Turn still decides completion after this.
    redemptionMs: 1000,
    preSpeechPadMs: 320,
    minSpeechMs: 250,
    submitUserSpeechOnPause: false,
  }
  // 512 samples @ 16 kHz = 32 ms per frame (Silero v5's native frame size).
  fp = new FrameProcessor(model.process, model.reset_state, options, 512 / 16)
  fp.resume()
  post({ kind: 'ready', info: 'Silero VAD v5 (wasm)' })
}

async function onFrame(samples: Float32Array): Promise<void> {
  if (!fp) return
  await fp.process(samples, (event) => {
    switch (event.msg) {
      case Message.SpeechStart:
        post({ kind: 'speech-start' })
        break
      case Message.SpeechEnd:
        post({ kind: 'speech-end', durationMs: event.audio.length / 16, audio: event.audio }, [
          event.audio.buffer,
        ])
        break
      case Message.VADMisfire:
        post({ kind: 'misfire' })
        break
      default:
        break
    }
  })
}
