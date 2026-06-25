// Fast streaming makeup-gain for the ASR-bound mic signal. Real laptop mics with
// AGC off capture quiet speech (peak ~0.13); this lifts it toward a healthy level so
// the encoder and VAD see a consistent amplitude.
//
// HONEST SCOPE: gain raises LOUDNESS, not SNR - it scales signal and noise together,
// so on its own it does NOT fix low-SNR garbling (that's what mic noiseSuppression is
// for). It pairs WITH noise suppression: denoise raises SNR, then this lands the clean
// signal at a good level. Kept gentle (noise-gated, smoothed, capped) so it never pumps
// or amplifies room tone in pauses.

export class AutoGain {
  private gain = 1
  private peakEnv = 0
  private readonly targetPeak: number
  private readonly maxGain: number
  private readonly gate: number //   below this running-peak, treat as noise/silence → don't boost
  private readonly release: number // per-frame peak-envelope decay (slow)
  private readonly smooth: number //  gain glide per frame (avoids pumping)

  constructor(opts: { targetPeak?: number; maxGain?: number; gate?: number } = {}) {
    this.targetPeak = opts.targetPeak ?? 0.5
    this.maxGain = opts.maxGain ?? 8 // ≤ +18 dB
    this.gate = opts.gate ?? 0.02
    this.release = 0.999
    this.smooth = 0.08
  }

  reset(): void {
    this.gain = 1
    this.peakEnv = 0
  }

  /** Process one frame in place-free; returns a new gained Float32Array (or the input
   *  unchanged when no boost is needed, to avoid a copy on already-healthy audio). */
  process(frame: Float32Array): Float32Array {
    let framePeak = 0
    for (let i = 0; i < frame.length; i++) {
      const a = frame[i] < 0 ? -frame[i] : frame[i]
      if (a > framePeak) framePeak = a
    }
    // Peak envelope: fast attack, slow release (tracks recent speech peaks).
    this.peakEnv = Math.max(framePeak, this.peakEnv * this.release)
    // Desired gain only when there's real signal above the noise gate.
    const desired = this.peakEnv > this.gate ? Math.min(this.maxGain, this.targetPeak / this.peakEnv) : 1
    this.gain += (desired - this.gain) * this.smooth
    if (this.gain <= 1.001) return frame // already loud enough → no work
    const out = new Float32Array(frame.length)
    const g = this.gain
    for (let i = 0; i < frame.length; i++) {
      const v = frame[i] * g
      out[i] = v > 1 ? 1 : v < -1 ? -1 : v // hard safety clip (rare; gain is capped)
    }
    return out
  }
}
