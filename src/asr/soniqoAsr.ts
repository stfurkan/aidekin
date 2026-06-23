// Soniqo FP16 Nemotron streaming engine — the nvidia FastConformer-RNNT base, with an
// ONNX streaming contract whose heavy encoder runs on WebGPU. Contract highlights:
//   • audio_signal is MEL-MAJOR fixed [1, 128, 32]
//   • language is a 128-wide one-hot `language_mask`, not an int64 lang_id
//   • the 9-frame mel pre-encode cache is an EXPLICIT pre_cache I/O
//   • i32 lengths; decoder_output is already [1,1,640] (joint-ready, no reshape)
//
// CRITICAL (matches soniqo's reference speech-core): the mel is computed over the
// WHOLE continuous audio buffer and each fixed 320 ms window is SLICED from it (with
// n_fft/2 right-context available before a window is decoded). Computing mel per
// 320 ms chunk independently — reflect-padding each chunk — creates seam artifacts at
// every boundary that garble long words.

import { NemoMelExtractor, type MelConfig } from './melFeatures'
import type { OrtSession, OrtTensor, TensorCtor } from './ortTypes'

export interface SoniqoConfig extends MelConfig {
  readonly hidden: number //            1024
  readonly encoderLayers: number //     24
  readonly leftContext: number //       56
  readonly convContext: number //       8
  readonly preCacheFrames: number //    9
  readonly decoderHidden: number //     640
  readonly decoderLayers: number //     2
  readonly chunkSamples: number //      5120 (320 ms)
  readonly melFramesPerChunk: number // 32
  readonly blankId: number //           13087
  readonly numPrompts: number //        128
  readonly maxSymbolsPerStep: number // 10
}

export interface SoniqoSessions {
  readonly encoder: OrtSession
  readonly decoder: OrtSession
  readonly joint: OrtSession
}

const ENC_IN = {
  audio: 'audio_signal',
  length: 'audio_length',
  langMask: 'language_mask',
  preCache: 'pre_cache',
  cacheCh: 'cache_last_channel',
  cacheTime: 'cache_last_time',
  cacheChLen: 'cache_last_channel_len',
} as const
const ENC_OUT = {
  encoded: 'encoded_output',
  encodedLen: 'encoded_length',
  preCacheNext: 'new_pre_cache',
  cacheChNext: 'new_cache_last_channel',
  cacheTimeNext: 'new_cache_last_time',
  cacheChLenNext: 'new_cache_last_channel_len',
} as const
const DEC_IN = { token: 'token', h: 'h', c: 'c' } as const
const DEC_OUT = { out: 'decoder_output', h: 'h_out', c: 'c_out' } as const
const JOIN_IN = { enc: 'encoder_output', dec: 'decoder_output' } as const
const JOIN_OUT = { logits: 'logits' } as const

export class SoniqoStreamingAsr {
  private readonly mel: NemoMelExtractor
  // Carried streaming state.
  private preCache!: OrtTensor
  private cacheCh!: OrtTensor
  private cacheTime!: OrtTensor
  private cacheChLen!: OrtTensor
  private lstmH!: OrtTensor
  private lstmC!: OrtTensor
  private lastToken!: number
  private langMask!: OrtTensor
  private langIdx = -1
  // Continuous audio buffer + how many 320 ms windows we've already decoded.
  private streamAudio: Float32Array = new Float32Array(0)
  private decodedWindows = 0

  constructor(
    private readonly s: SoniqoSessions,
    private readonly T: TensorCtor,
    private readonly cfg: SoniqoConfig,
  ) {
    this.mel = new NemoMelExtractor(cfg)
    this.reset()
    this.setLang(0)
  }

  reset(): void {
    const c = this.cfg
    this.preCache = this.zeros([1, c.numMels, c.preCacheFrames])
    this.cacheCh = this.zeros([c.encoderLayers, 1, c.leftContext, c.hidden])
    this.cacheTime = this.zeros([c.encoderLayers, 1, c.hidden, c.convContext])
    this.cacheChLen = this.T('int32', new Int32Array([0]), [1])
    this.lstmH = this.zeros([c.decoderLayers, 1, c.decoderHidden])
    this.lstmC = this.zeros([c.decoderLayers, 1, c.decoderHidden])
    this.lastToken = c.blankId
    this.streamAudio = new Float32Array(0)
    this.decodedWindows = 0
  }

  /** Feed live mic audio; decode + return any newly-completed windows' tokens. */
  async pushAudio(samples: Float32Array, langId: number): Promise<number[]> {
    this.setLang(langId)
    this.streamAudio = concat(this.streamAudio, samples)

    const c = this.cfg
    const winSamples = c.chunkSamples
    const rightCtx = c.nFft >> 1 // need n_fft/2 future samples so a window's tail is final
    const emitted: number[] = []

    // Only recompute mel when at least one new window is fully covered + right-context.
    if (this.streamAudio.length < (this.decodedWindows + 1) * winSamples + rightCtx) return emitted

    const { mel, frames: produced } = this.mel.process(this.streamAudio)
    while (this.streamAudio.length >= (this.decodedWindows + 1) * winSamples + rightCtx) {
      const f0 = this.decodedWindows * c.melFramesPerChunk
      if (f0 + c.melFramesPerChunk > produced) break
      const ids = await this.runWindow(this.sliceWindow(mel, produced, f0))
      emitted.push(...ids)
      this.decodedWindows++
    }
    return emitted
  }

  /** End of turn: pad to a whole number of windows and decode the remainder. */
  async endStream(langId: number): Promise<number[]> {
    this.setLang(langId)
    const c = this.cfg
    const emitted: number[] = []
    if (this.streamAudio.length === 0) return emitted

    const winSamples = c.chunkSamples
    const totalWindows = Math.ceil(this.streamAudio.length / winSamples)
    if (this.streamAudio.length % winSamples !== 0) {
      const padded = new Float32Array(totalWindows * winSamples)
      padded.set(this.streamAudio)
      this.streamAudio = padded
    }
    const { mel, frames: produced } = this.mel.process(this.streamAudio)
    while (this.decodedWindows < totalWindows) {
      const f0 = this.decodedWindows * c.melFramesPerChunk
      const ids = await this.runWindow(this.sliceWindow(mel, produced, f0))
      emitted.push(...ids)
      this.decodedWindows++
    }
    this.streamAudio = new Float32Array(0)
    this.decodedWindows = 0
    return emitted
  }

  /** Run the encoder on one [1, numMels, 32] window, carry caches, greedy-decode. */
  private async runWindow(audioSignal: OrtTensor): Promise<number[]> {
    const c = this.cfg
    const encOut = await this.s.encoder.run({
      [ENC_IN.audio]: audioSignal,
      [ENC_IN.length]: this.T('int32', new Int32Array([c.melFramesPerChunk]), [1]),
      [ENC_IN.langMask]: this.langMask,
      [ENC_IN.preCache]: this.preCache,
      [ENC_IN.cacheCh]: this.cacheCh,
      [ENC_IN.cacheTime]: this.cacheTime,
      [ENC_IN.cacheChLen]: this.cacheChLen,
    })
    this.preCache = encOut[ENC_OUT.preCacheNext]
    this.cacheCh = encOut[ENC_OUT.cacheChNext]
    this.cacheTime = encOut[ENC_OUT.cacheTimeNext]
    this.cacheChLen = encOut[ENC_OUT.cacheChLenNext]

    const encoded = encOut[ENC_OUT.encoded]
    const encodedLen = Number((encOut[ENC_OUT.encodedLen].data as Int32Array)[0])
    return this.greedyDecode(encoded, encodedLen)
  }

  /** Smoke-test the encoder (native FP16 on WebGPU is experimental → may emit NaN). */
  async selfTest(): Promise<{ ok: boolean; reason: string; output: Float32Array }> {
    const c = this.cfg
    // A richer test signal than a pure tone (mix of tones + a little structure) so the
    // encoder output is discriminative enough to compare WebGPU vs WASM (see asr.worker).
    const x = new Float32Array(c.chunkSamples)
    for (let i = 0; i < x.length; i++) {
      const t = i / c.sampleRate
      x[i] = 0.2 * Math.sin(2 * Math.PI * 180 * t) + 0.15 * Math.sin(2 * Math.PI * 540 * t) + 0.1 * Math.sin(2 * Math.PI * 1200 * t)
    }
    const { mel, frames } = this.mel.process(x)
    const encOut = await this.s.encoder.run({
      [ENC_IN.audio]: this.sliceWindow(mel, frames, 0),
      [ENC_IN.length]: this.T('int32', new Int32Array([c.melFramesPerChunk]), [1]),
      [ENC_IN.langMask]: this.langMask,
      [ENC_IN.preCache]: this.preCache,
      [ENC_IN.cacheCh]: this.cacheCh,
      [ENC_IN.cacheTime]: this.cacheTime,
      [ENC_IN.cacheChLen]: this.cacheChLen,
    })
    const e = encOut[ENC_OUT.encoded].data as Float32Array
    const output = new Float32Array(e) // copy (the tensor's buffer may be reused)
    let bad = 0
    let nonzero = 0
    let mn = Infinity
    let mx = -Infinity
    for (let i = 0; i < e.length; i++) {
      const v = e[i]
      if (!Number.isFinite(v)) bad++
      else {
        if (v !== 0) nonzero++
        if (v < mn) mn = v
        if (v > mx) mx = v
      }
    }
    this.reset()
    if (bad > 0) return { ok: false, reason: `${bad} NaN/Inf in encoder output`, output }
    if (nonzero === 0) return { ok: false, reason: 'encoder output all-zero', output }
    if (mx - mn < 1e-4) return { ok: false, reason: 'encoder output degenerate (no variance)', output }
    return { ok: true, reason: `ok (min=${mn.toFixed(2)} max=${mx.toFixed(2)})`, output }
  }

  /** Warm the FULL streaming path — encoder + decoder + joint. selfTest() only runs the
   *  encoder, so without this the FIRST real window would cold-start the decoder/joint
   *  (the first utterance comes back empty/garbled, then it works). Feeds synthetic audio
   *  through the real push/end path, discards the output, and resets to clean state. */
  async warmup(langId: number): Promise<void> {
    this.reset()
    const c = this.cfg
    const x = new Float32Array(c.chunkSamples * 2) // ≥ one window + right-context
    for (let i = 0; i < x.length; i++) x[i] = 0.05 * Math.sin((2 * Math.PI * 220 * i) / c.sampleRate)
    await this.pushAudio(x, langId)
    await this.endStream(langId)
    this.reset()
  }

  private setLang(idx: number): void {
    if (idx === this.langIdx) return
    const mask = new Float32Array(this.cfg.numPrompts)
    if (idx >= 0 && idx < mask.length) mask[idx] = 1
    this.langMask = this.T('float32', mask, [1, this.cfg.numPrompts])
    this.langIdx = idx
  }

  /** Slice the fixed [1, numMels, 32] window at frame offset f0 from the continuous mel. */
  private sliceWindow(mel: Float32Array, produced: number, f0: number): OrtTensor {
    const c = this.cfg
    const F = c.melFramesPerChunk
    const out = new Float32Array(c.numMels * F)
    for (let m = 0; m < c.numMels; m++) {
      const srcRow = m * produced
      const dstRow = m * F
      for (let t = 0; t < F; t++) {
        const src = f0 + t
        if (src < produced) out[dstRow + t] = mel[srcRow + src]
      }
    }
    return this.T('float32', out, [1, c.numMels, F])
  }

  private async greedyDecode(encoded: OrtTensor, encodedLen: number): Promise<number[]> {
    const c = this.cfg
    const hidden = c.hidden
    const timeSteps = Math.min(encoded.dims[1], encodedLen)
    const data = encoded.data as Float32Array
    const emitted: number[] = []
    let decoderDirty = true
    let decOut: Record<string, OrtTensor> | null = null

    for (let t = 0; t < timeSteps; t++) {
      const encFrame = this.T('float32', data.slice(t * hidden, t * hidden + hidden), [1, 1, hidden])
      let symbols = 0
      while (symbols < c.maxSymbolsPerStep) {
        if (decoderDirty || decOut === null) {
          decOut = await this.s.decoder.run({
            [DEC_IN.token]: this.T('int64', new BigInt64Array([BigInt(this.lastToken)]), [1, 1]),
            [DEC_IN.h]: this.lstmH,
            [DEC_IN.c]: this.lstmC,
          })
          decoderDirty = false
        }
        const decFrame = decOut[DEC_OUT.out] // [1, 1, 640] — already joint-ready
        const joinOut = await this.s.joint.run({ [JOIN_IN.enc]: encFrame, [JOIN_IN.dec]: decFrame })
        const best = argmax(joinOut[JOIN_OUT.logits].data as Float32Array)

        if (best === c.blankId) break
        emitted.push(best)
        this.lastToken = best
        this.lstmH = decOut[DEC_OUT.h]
        this.lstmC = decOut[DEC_OUT.c]
        decoderDirty = true
        symbols++
      }
    }
    return emitted
  }

  private zeros(dims: number[]): OrtTensor {
    const n = dims.reduce((a, b) => a * b, 1)
    return this.T('float32', new Float32Array(n), dims)
  }
}

function concat(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(a.length + b.length)
  out.set(a)
  out.set(b, a.length)
  return out
}

function argmax(d: Float32Array): number {
  let best = 0
  let bestScore = d[0]
  for (let i = 1; i < d.length; i++) {
    if (d[i] > bestScore) {
      bestScore = d[i]
      best = i
    }
  }
  return best
}

/** Build a SoniqoConfig from the registry contract (so the engine stays runtime-agnostic). */
export function soniqoConfig(c: {
  sampleRate: number
  numMels: number
  nFft: number
  hopLength: number
  winLength: number
  preemph: number
  logEps: number
  hidden: number
  encoderLayers: number
  leftContext: number
  convContext: number
  preCacheFrames: number
  decoderHidden: number
  decoderLayers: number
  chunkSamples: number
  melFramesPerChunk: number
  blankId: number
  numPrompts: number
  maxSymbolsPerStep: number
}): SoniqoConfig {
  return {
    sampleRate: c.sampleRate,
    nFft: c.nFft,
    hopLength: c.hopLength,
    winLength: c.winLength,
    numMels: c.numMels,
    fMin: 0,
    fMax: c.sampleRate / 2,
    preemph: c.preemph,
    logEps: c.logEps,
    hidden: c.hidden,
    encoderLayers: c.encoderLayers,
    leftContext: c.leftContext,
    convContext: c.convContext,
    preCacheFrames: c.preCacheFrames,
    decoderHidden: c.decoderHidden,
    decoderLayers: c.decoderLayers,
    chunkSamples: c.chunkSamples,
    melFramesPerChunk: c.melFramesPerChunk,
    blankId: c.blankId,
    numPrompts: c.numPrompts,
    maxSymbolsPerStep: c.maxSymbolsPerStep,
  }
}
