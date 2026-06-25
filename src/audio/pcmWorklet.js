// AudioWorklet processor: capture mono mic audio, resample to 16 kHz, and emit
// fixed-size Float32 frames + an RMS level to the main thread.
//
// micCapture.ts captures at the device's NATIVE rate (it does NOT force a 16 kHz
// AudioContext) and we downsample HERE with an anti-aliased windowed-sinc (Lanczos)
// kernel - never the browser's resampler, which can alias enough to garble/blank
// transcripts (the Nemotron encoder is sensitive to aliasing).
//
// CANONICAL SOURCE: src/audio/resampler.ts. The worklet runs as a standalone asset
// (addModule) and cannot import ES modules, so `lanczos` + `SincResampler` below are
// a byte-for-byte copy. KEEP THEM IN SYNC; the algorithm is unit-tested headlessly
// against the ASR by scripts/asr-resample-test.ts.

const TARGET_RATE = 16000

function lanczos(x, a) {
  if (x === 0) return 1
  if (x <= -a || x >= a) return 0
  const px = Math.PI * x
  return (a * Math.sin(px) * Math.sin(px / a)) / (px * px)
}

// Streaming, state-preserving windowed-sinc resampler (arbitrary in → out).
class SincResampler {
  constructor(inRate, outRate, a = 16) {
    this.ratio = inRate / outRate // input samples advanced per output sample
    this.a = a
    this.cutoff = Math.min(1, outRate / inRate) // < 1 when downsampling → anti-alias low-pass
    this.half = Math.ceil(a / this.cutoff) // kernel half-width in input samples
    this.buf = new Float32Array(0)
    this.t = 0 // fractional input index of the next output sample
  }

  process(chunk) {
    const merged = new Float32Array(this.buf.length + chunk.length)
    merged.set(this.buf)
    merged.set(chunk, this.buf.length)
    this.buf = merged

    const { ratio, a, cutoff, half } = this
    const out = []
    while (this.t + half < this.buf.length) {
      const center = this.t
      const i0 = Math.max(0, Math.ceil(center - half))
      const i1 = Math.min(this.buf.length - 1, Math.floor(center + half))
      let acc = 0
      let norm = 0
      for (let i = i0; i <= i1; i++) {
        const w = lanczos((i - center) * cutoff, a)
        acc += this.buf[i] * w
        norm += w
      }
      out.push(norm > 0 ? acc / norm : 0)
      this.t += ratio
    }
    // Retain only the tail needed for subsequent output windows.
    const keepFrom = Math.max(0, Math.floor(this.t - half))
    if (keepFrom > 0) {
      this.buf = this.buf.slice(keepFrom)
      this.t -= keepFrom
    }
    return Float32Array.from(out)
  }
}

class PcmCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super()
    const opt = (options && options.processorOptions) || {}
    this.frameSize = opt.frameSize || 512 // samples per emitted frame @ 16 kHz
    this.inRate = sampleRate // AudioWorklet global == context rate
    this.needResample = Math.abs(this.inRate - TARGET_RATE) > 1
    this.resampler = this.needResample ? new SincResampler(this.inRate, TARGET_RATE) : null
    this.frame = new Float32Array(this.frameSize)
    this.framePos = 0
    this.port.postMessage({
      type: 'meta',
      inRate: this.inRate,
      targetRate: TARGET_RATE,
      resampling: this.needResample,
    })
  }

  process(inputs) {
    const input = inputs[0]
    const ch = input && input[0]
    if (!ch) return true
    const samples = this.needResample ? this.resampler.process(ch) : ch
    for (let i = 0; i < samples.length; i++) {
      this.frame[this.framePos++] = samples[i]
      if (this.framePos === this.frameSize) {
        let sum = 0
        for (let j = 0; j < this.frameSize; j++) sum += this.frame[j] * this.frame[j]
        const rms = Math.sqrt(sum / this.frameSize)
        const out = new Float32Array(this.frameSize)
        out.set(this.frame)
        this.port.postMessage({ type: 'frame', samples: out, rms }, [out.buffer])
        this.framePos = 0
      }
    }
    return true
  }
}

registerProcessor('pcm-capture', PcmCaptureProcessor)
