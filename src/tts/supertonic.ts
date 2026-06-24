// Supertonic-3 TTS engine — TS port of the official web example
// (supertone-inc/supertonic /web/helper.js), specialized to single-utterance
// synthesis. 4-stage flow-matching pipeline on onnxruntime-web:
//   text → unicode ids → duration_predictor → text_encoder
//        → vector_estimator (denoising ODE loop) → vocoder → 44.1 kHz PCM
// English-only: text is wrapped in <en>…</en>.

import * as ort from 'onnxruntime-web/webgpu'

const LANG = 'en'

export interface SupertonicConfig {
  ae: { sample_rate: number; base_chunk_size: number }
  ttl: { chunk_compress_factor: number; latent_dim: number }
}
export interface VoiceStyle {
  ttl: ort.Tensor
  dp: ort.Tensor
}
export interface SupertonicSessions {
  durationPredictor: ort.InferenceSession
  textEncoder: ort.InferenceSession
  vectorEstimator: ort.InferenceSession
  vocoder: ort.InferenceSession
}

const DASH_QUOTE: Record<string, string> = {
  '–': '-', '‑': '-', '—': '-', _: ' ', '“': '"', '”': '"', '‘': "'", '’': "'",
  '´': "'", '`': "'", '[': ' ', ']': ' ', '|': ' ', '/': ' ', '#': ' ', '→': ' ', '←': ' ',
}

export function makeVoiceStyle(json: {
  style_ttl: { dims: number[]; data: unknown }
  style_dp: { dims: number[]; data: unknown }
}): VoiceStyle {
  const flat = (a: unknown): number[] => (Array.isArray(a) ? (a.flat(Infinity) as number[]) : [])
  return {
    ttl: new ort.Tensor('float32', new Float32Array(flat(json.style_ttl.data)), json.style_ttl.dims),
    dp: new ort.Tensor('float32', new Float32Array(flat(json.style_dp.data)), json.style_dp.dims),
  }
}

export class SupertonicTts {
  readonly sampleRate: number

  constructor(
    private readonly cfg: SupertonicConfig,
    private readonly indexer: number[],
    private readonly sessions: SupertonicSessions,
    private readonly style: VoiceStyle,
  ) {
    this.sampleRate = cfg.ae.sample_rate
  }

  /**
   * Synthesize one text into 44.1 kHz mono PCM (chunks long text, concatenated).
   * Supertonic-3 flow-matching is built for low NFE (the official demo ships usable
   * 2/4/8-step samples), so 8 steps halves synthesis cost vs 16 with no audible loss
   * for a conversational voice.
   */
  async synthesize(text: string, totalStep = 8, speed = 1.05): Promise<Float32Array> {
    const pieces = chunkText(text)
    const wavs: Float32Array[] = []
    for (const piece of pieces) wavs.push(await this.inferOne(piece, totalStep, speed))
    return concatWithSilence(wavs, Math.floor(0.12 * this.sampleRate))
  }

  private async inferOne(raw: string, totalStep: number, speed: number): Promise<Float32Array> {
    const text = this.preprocess(raw)
    // Iterate Unicode code points (not UTF-16 units) so non-BMP chars (e.g. emoji) map to one
    // id each instead of desyncing on surrogate pairs. Identical to char indexing for BMP text.
    const cps = [...text]
    const len = cps.length
    const ids = new BigInt64Array(len)
    for (let j = 0; j < len; j++) {
      const cp = cps[j].codePointAt(0) ?? 0
      ids[j] = BigInt(cp < this.indexer.length ? this.indexer[cp] : -1)
    }
    const textIds = new ort.Tensor('int64', ids, [1, len])
    const textMask = new ort.Tensor('float32', new Float32Array(len).fill(1), [1, 1, len])

    const dp = await this.sessions.durationPredictor.run({
      text_ids: textIds,
      style_dp: this.style.dp,
      text_mask: textMask,
    })
    const durationSec = Number((dp.duration.data as Float32Array)[0]) / speed

    const te = await this.sessions.textEncoder.run({
      text_ids: textIds,
      style_ttl: this.style.ttl,
      text_mask: textMask,
    })
    const textEmb = te.text_emb

    const chunkSize = this.cfg.ae.base_chunk_size * this.cfg.ttl.chunk_compress_factor
    const latentDimVal = this.cfg.ttl.latent_dim * this.cfg.ttl.chunk_compress_factor
    const wavLen = Math.floor(durationSec * this.sampleRate)
    const latentLen = Math.max(1, Math.floor((wavLen + chunkSize - 1) / chunkSize))
    const latentMask = new ort.Tensor('float32', new Float32Array(latentLen).fill(1), [1, 1, latentLen])

    // Gaussian noise (Box–Muller). latentMask is all-ones for a single utterance.
    let xt = new Float32Array(latentDimVal * latentLen)
    for (let i = 0; i < xt.length; i++) {
      const u1 = Math.max(0.0001, Math.random())
      const u2 = Math.random()
      xt[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
    }

    const totalStepT = new ort.Tensor('float32', new Float32Array([totalStep]), [1])
    for (let step = 0; step < totalStep; step++) {
      const out = await this.sessions.vectorEstimator.run({
        noisy_latent: new ort.Tensor('float32', xt, [1, latentDimVal, latentLen]),
        text_emb: textEmb,
        style_ttl: this.style.ttl,
        latent_mask: latentMask,
        text_mask: textMask,
        current_step: new ort.Tensor('float32', new Float32Array([step]), [1]),
        total_step: totalStepT,
      })
      xt = new Float32Array(out.denoised_latent.data as Float32Array)
    }

    const vocoded = await this.sessions.vocoder.run({
      latent: new ort.Tensor('float32', xt, [1, latentDimVal, latentLen]),
    })
    return new Float32Array(vocoded.wav_tts.data as Float32Array)
  }

  private preprocess(input: string): string {
    let text = input.normalize('NFKD')
    for (const [k, v] of Object.entries(DASH_QUOTE)) text = text.replaceAll(k, v)
    text = text.replace(/[♥☆♡©\\]/g, '').replaceAll('@', ' at ')
    text = text
      .replace(/ ,/g, ',')
      .replace(/ \./g, '.')
      .replace(/ !/g, '!')
      .replace(/ \?/g, '?')
      .replace(/\s+/g, ' ')
      .trim()
    if (!/[.!?;:,'")\]}…。」』】〉》›»]$/.test(text)) text += '.'
    return `<${LANG}>${text}</${LANG}>`
  }
}

// ── helpers ──────────────────────────────────────────────────────────────────
function chunkText(text: string, maxLen = 300): string[] {
  const sentences = text.trim().split(/(?<=[.!?])\s+/)
  const chunks: string[] = []
  let current = ''
  for (const s of sentences) {
    if (current.length + s.length + 1 <= maxLen) current += (current ? ' ' : '') + s
    else {
      if (current) chunks.push(current.trim())
      current = s
    }
  }
  if (current.trim()) chunks.push(current.trim())
  return chunks.length ? chunks : [text.trim()]
}

function concatWithSilence(wavs: Float32Array[], silenceSamples: number): Float32Array {
  if (wavs.length === 1) return wavs[0]
  const total = wavs.reduce((a, w) => a + w.length, 0) + silenceSamples * Math.max(0, wavs.length - 1)
  const out = new Float32Array(total)
  let off = 0
  for (let i = 0; i < wavs.length; i++) {
    out.set(wavs[i], off)
    off += wavs[i].length
    if (i < wavs.length - 1) off += silenceSamples
  }
  return out
}
