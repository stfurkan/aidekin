// NeMo FilterbankFeatures-equivalent log-mel extractor for the Nemotron streaming
// ASR encoder. Reconstructed from genai_config.json + audio_processor_config.json
// and NeMo's documented preprocessing — the ORT-GenAI C++ (nemo_mel_spectrogram.*)
// is generated at build time and is NOT in the public repo, so we replicate it and
// validate empirically by transcript (scripts/asr-spike.ts).
//
// Per chunk, matching ORT-GenAI's NemotronStreamingProcessor:
//   preemphasis(0.97) → STFT(n_fft=512, hop=160, win=400 symmetric-hann, center,
//   reflect-pad) → power(|·|²) → Slaney mel filterbank(128, 0–8000 Hz)
//   → ln(mel + log_eps)
// Output: Float32Array laid out [num_mels, num_frames] (frequency-major), exactly
// like the C++ extractor (the streaming processor transposes + caches afterwards).

import FFT from 'fft.js'

export interface MelConfig {
  readonly sampleRate: number
  readonly nFft: number
  readonly hopLength: number
  readonly winLength: number
  readonly numMels: number
  readonly fMin: number
  readonly fMax: number
  readonly preemph: number
  readonly logEps: number
}

export interface MelResult {
  /** [numMels * frames], frequency-major: value(m, t) = mel[m * frames + t]. */
  readonly mel: Float32Array
  readonly frames: number
}

export class NemoMelExtractor {
  private readonly cfg: MelConfig
  private readonly fft: FFT
  private readonly nFreq: number
  private readonly window: Float32Array // length nFft (symmetric hann centered, zero-padded)
  private readonly melFb: Float32Array // [numMels * nFreq]
  private readonly frame: Float64Array // reusable windowed frame (length nFft)
  private readonly spectrum: Float64Array // reusable complex output (length 2*nFft)

  constructor(cfg: MelConfig) {
    this.cfg = cfg
    this.fft = new FFT(cfg.nFft)
    this.nFreq = (cfg.nFft >> 1) + 1
    this.window = buildCenteredHann(cfg.winLength, cfg.nFft)
    this.melFb = buildSlaneyMelFilterbank(cfg.sampleRate, cfg.nFft, cfg.numMels, cfg.fMin, cfg.fMax)
    this.frame = new Float64Array(cfg.nFft)
    this.spectrum = new Float64Array(2 * cfg.nFft)
  }

  process(audio: Float32Array): MelResult {
    const { nFft, hopLength, preemph, numMels, logEps } = this.cfg
    const n = audio.length

    // 1) Pre-emphasis on the raw chunk: y[0]=x[0]; y[t]=x[t]-preemph*x[t-1].
    const pre = new Float32Array(n)
    if (n > 0) pre[0] = audio[0]
    for (let t = 1; t < n; t++) pre[t] = audio[t] - preemph * audio[t - 1]

    // 2) center=True reflect padding by nFft/2 on each side (torch.stft semantics).
    const pad = nFft >> 1
    const padded = reflectPad(pre, pad)

    // 3) STFT framing: ORT-GenAI's streaming extractor emits floor(n / hop) frames
    //    (e.g. 8960/160 = 56), centered (reflect-padded), not the 1+floor convention.
    const frames = Math.floor(n / hopLength)
    const mel = new Float32Array(numMels * frames)
    const { window, melFb, nFreq, frame, spectrum, fft } = this

    for (let i = 0; i < frames; i++) {
      const start = i * hopLength
      // windowed frame (window already zero outside the centered win_length span)
      for (let k = 0; k < nFft; k++) frame[k] = padded[start + k] * window[k]

      fft.realTransform(spectrum, frame)
      fft.completeSpectrum(spectrum)

      // power spectrum |·|² for bins 0..nFreq-1, then apply mel filterbank + log.
      for (let m = 0; m < numMels; m++) {
        const row = m * nFreq
        let acc = 0
        for (let f = 0; f < nFreq; f++) {
          const w = melFb[row + f]
          if (w !== 0) {
            const re = spectrum[2 * f]
            const im = spectrum[2 * f + 1]
            acc += w * (re * re + im * im)
          }
        }
        mel[m * frames + i] = Math.log(acc + logEps)
      }
    }

    return { mel, frames }
  }
}

// ── helpers ──────────────────────────────────────────────────────────────────

/** Symmetric Hann (periodic=False, as NeMo builds it) of winLength, centered in nFft. */
function buildCenteredHann(winLength: number, nFft: number): Float32Array {
  const w = new Float32Array(nFft)
  const offset = (nFft - winLength) >> 1
  const denom = winLength - 1
  for (let n = 0; n < winLength; n++) {
    w[offset + n] = 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / denom)
  }
  return w
}

/** torch 'reflect' padding (does not repeat the edge sample). */
function reflectPad(x: Float32Array, pad: number): Float32Array {
  const n = x.length
  const out = new Float32Array(n + 2 * pad)
  for (let i = 0; i < pad; i++) out[i] = x[pad - i] // left: x[pad..1]
  out.set(x, pad)
  for (let i = 0; i < pad; i++) out[pad + n + i] = x[n - 2 - i] // right: x[n-2..n-1-pad]
  return out
}

// Slaney mel scale (librosa htk=False).
function hzToMel(hz: number): number {
  const fSp = 200.0 / 3
  const minLogHz = 1000.0
  const minLogMel = minLogHz / fSp
  const logstep = Math.log(6.4) / 27.0
  return hz >= minLogHz ? minLogMel + Math.log(hz / minLogHz) / logstep : hz / fSp
}

function melToHz(mel: number): number {
  const fSp = 200.0 / 3
  const minLogHz = 1000.0
  const minLogMel = minLogHz / fSp
  const logstep = Math.log(6.4) / 27.0
  return mel >= minLogMel ? minLogHz * Math.exp(logstep * (mel - minLogMel)) : fSp * mel
}

/** librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax, htk=False, norm='slaney'). */
function buildSlaneyMelFilterbank(
  sr: number,
  nFft: number,
  nMels: number,
  fMin: number,
  fMax: number,
): Float32Array {
  const nFreq = (nFft >> 1) + 1
  const fb = new Float32Array(nMels * nFreq)

  // FFT bin center frequencies: linspace(0, sr/2, nFreq).
  const fftFreqs = new Float64Array(nFreq)
  for (let f = 0; f < nFreq; f++) fftFreqs[f] = (sr / 2) * (f / (nFreq - 1))

  // Mel band edges: nMels+2 points evenly spaced in mel, back to Hz.
  const melMin = hzToMel(fMin)
  const melMax = hzToMel(fMax)
  const melF = new Float64Array(nMels + 2)
  for (let i = 0; i < nMels + 2; i++) {
    melF[i] = melToHz(melMin + ((melMax - melMin) * i) / (nMels + 1))
  }

  for (let m = 0; m < nMels; m++) {
    const lo = melF[m]
    const ctr = melF[m + 1]
    const hi = melF[m + 2]
    const lowerDen = ctr - lo
    const upperDen = hi - ctr
    const enorm = 2.0 / (hi - lo) // Slaney normalization
    const row = m * nFreq
    for (let f = 0; f < nFreq; f++) {
      const freq = fftFreqs[f]
      const lower = (freq - lo) / lowerDen
      const upper = (hi - freq) / upperDen
      const tri = Math.max(0, Math.min(lower, upper))
      fb[row + f] = tri * enorm
    }
  }
  return fb
}
