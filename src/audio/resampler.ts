// Streaming, state-preserving windowed-sinc (Lanczos) resampler - the SINGLE SOURCE
// OF TRUTH for mic resampling. The AudioWorklet (pcmWorklet.js) cannot import ES
// modules at runtime (it is loaded as a standalone asset via addModule), so it keeps
// a byte-for-byte copy of `lanczos` + `SincResampler` below. KEEP THEM IN SYNC.
// This module exists so the algorithm is unit-testable headlessly against the ASR -
// the Nemotron encoder degrades badly on aliased input, so a regression here silently
// breaks transcription in the browser.
//
// Quality: Lanczos-a=16 with an anti-alias cutoff at the target Nyquist when
// downsampling - comparable to ffmpeg's soxr, and far above linear decimation
// (which empirically yields blank/garbled transcripts).

export function lanczos(x: number, a: number): number {
  if (x === 0) return 1
  if (x <= -a || x >= a) return 0
  const px = Math.PI * x
  return (a * Math.sin(px) * Math.sin(px / a)) / (px * px)
}

export class SincResampler {
  private readonly ratio: number
  private readonly a: number
  private readonly cutoff: number
  private readonly half: number
  private buf: Float32Array
  private t: number

  constructor(inRate: number, outRate: number, a = 16) {
    this.ratio = inRate / outRate //                input samples advanced per output sample
    this.a = a
    this.cutoff = Math.min(1, outRate / inRate) //  < 1 when downsampling → anti-alias low-pass
    this.half = Math.ceil(a / this.cutoff) //       kernel half-width in input samples
    this.buf = new Float32Array(0)
    this.t = 0 //                                   fractional input index of the next output sample
  }

  process(chunk: Float32Array): Float32Array {
    const merged = new Float32Array(this.buf.length + chunk.length)
    merged.set(this.buf)
    merged.set(chunk, this.buf.length)
    this.buf = merged

    const { ratio, a, cutoff, half } = this
    const out: number[] = []
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
