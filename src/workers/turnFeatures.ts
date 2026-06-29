// Standalone WhisperFeatureExtractor for Smart Turn v3 (no transformers.js). Faithful reimplementation
// of the transformers.js recipe: pad/trim to 8s -> reflect pad -> periodic hann -> 400-pt DFT -> power
// -> slaney mel (80) -> log10 + (max(x, max-8)+4)/4. Output [1, 80, 800] matching the ONNX input.
// Verified to give identical turn-detection verdicts vs AutoProcessor (scripts/verify-turn.ts).
const N_FFT = 400
const HOP = 160
const N_MELS = 80
const N_SAMPLES = 8 * 16000 // 128000
const FRAMES = 800
const BINS = N_FFT / 2 + 1 // 201
const SR = 16000

// periodic hann: window_function(400,'hann') = hanning(401)[0:400] = 0.5 - 0.5 cos(2*pi*i/400)
const HANN = Float32Array.from({ length: N_FFT }, (_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / N_FFT))

const hzToMel = (f: number): number => (f >= 1000 ? 15 + Math.log(f / 1000) * (27 / Math.log(6.4)) : (3 * f) / 200)
const melToHz = (m: number): number => (m >= 15 ? 1000 * Math.exp((Math.log(6.4) / 27) * (m - 15)) : (200 * m) / 3)
const linspace = (a: number, b: number, n: number): Float64Array => Float64Array.from({ length: n }, (_, i) => a + ((b - a) / (n - 1)) * i)

// slaney mel filter bank, flat [N_MELS * BINS] (mel-major), matching mel_filter_bank(201,80,0,8000,16000,'slaney','slaney')
function buildMel(): Float32Array {
  const melFreqs = linspace(hzToMel(0), hzToMel(8000), N_MELS + 2)
  const filterFreqs = Float64Array.from(melFreqs, melToHz) // 82
  const fftFreqs = linspace(0, Math.floor(SR / 2), BINS) // 201
  const diff = Float64Array.from({ length: filterFreqs.length - 1 }, (_, i) => filterFreqs[i + 1] - filterFreqs[i])
  const mel = new Float32Array(N_MELS * BINS)
  for (let m = 0; m < N_MELS; m++) {
    const enorm = 2.0 / (filterFreqs[m + 2] - filterFreqs[m]) // slaney norm
    for (let b = 0; b < BINS; b++) {
      const down = -(filterFreqs[m] - fftFreqs[b]) / diff[m]
      const up = (filterFreqs[m + 2] - fftFreqs[b]) / diff[m + 1]
      mel[m * BINS + b] = Math.max(0, Math.min(down, up)) * enorm
    }
  }
  return mel
}

// DFT matrices for the 400-pt transform (onesided, 201 bins). 400 is not a power of 2, so use the
// direct transform via precomputed cos/sin tables (built once).
let COS: Float32Array | null = null
let SIN: Float32Array | null = null
let MEL: Float32Array | null = null
function init(): void {
  if (COS) return
  COS = new Float32Array(BINS * N_FFT)
  SIN = new Float32Array(BINS * N_FFT)
  for (let k = 0; k < BINS; k++) {
    for (let n = 0; n < N_FFT; n++) {
      const a = (-2 * Math.PI * k * n) / N_FFT
      COS[k * N_FFT + n] = Math.cos(a)
      SIN[k * N_FFT + n] = Math.sin(a)
    }
  }
  MEL = buildMel()
}

// numpy 'reflect' pad (mirror at the edges, edge not repeated)
function reflectPad(x: Float32Array, pad: number): Float32Array {
  const w = x.length - 1
  const out = new Float32Array(x.length + 2 * pad)
  out.set(x, pad)
  for (let i = 1; i <= pad; i++) out[pad - i] = x[i] // left
  for (let i = 1; i <= pad; i++) out[w + pad + i] = x[w - i] // right
  return out
}

/** Compute Smart Turn's [1, 80, 800] log-mel features from a 16kHz mono waveform. */
export function whisperFeatures(audio: Float32Array): Float32Array {
  init()
  const cos = COS!, sin = SIN!, mel = MEL!
  const wave = new Float32Array(N_SAMPLES)
  wave.set(audio.length > N_SAMPLES ? audio.subarray(0, N_SAMPLES) : audio) // trim or right-pad zeros
  const padded = reflectPad(wave, N_FFT / 2) // center=true, reflect

  // Frames whose window lies entirely in the zero-padded tail are all-zero power -> a constant after
  // log/norm, so skip the DFT for them (big speedup for the common <8s utterance; identical result).
  const realLen = Math.min(audio.length, N_SAMPLES)
  const lastFrame = Math.min(FRAMES, Math.ceil((N_FFT / 2 + realLen) / HOP) + 1)
  const power = new Float32Array(FRAMES * BINS)
  const frame = new Float32Array(N_FFT)
  for (let t = 0; t < lastFrame; t++) {
    const start = t * HOP
    for (let n = 0; n < N_FFT; n++) frame[n] = padded[start + n] * HANN[n]
    for (let k = 0; k < BINS; k++) {
      const cb = k * N_FFT
      let re = 0
      let im = 0
      for (let n = 0; n < N_FFT; n++) {
        const v = frame[n]
        re += v * cos[cb + n]
        im += v * sin[cb + n]
      }
      power[t * BINS + k] = re * re + im * im
    }
  }

  // mel = MEL[80,201] @ power.T -> [80, 800]; clamp 1e-10; log10; (max(x, max-8)+4)/4
  const out = new Float32Array(N_MELS * FRAMES)
  let logMax = -Infinity
  for (let m = 0; m < N_MELS; m++) {
    const mb = m * BINS
    for (let t = 0; t < FRAMES; t++) {
      let acc = 0
      const pb = t * BINS
      for (let b = 0; b < BINS; b++) acc += mel[mb + b] * power[pb + b]
      const lg = Math.log10(Math.max(1e-10, acc))
      out[m * FRAMES + t] = lg
      if (lg > logMax) logMax = lg
    }
  }
  const threshold = logMax - 8.0
  for (let i = 0; i < out.length; i++) out[i] = (Math.max(out[i], threshold) + 4.0) / 4.0
  return out
}
