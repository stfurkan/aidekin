// Gapless chunked audio playback for streaming TTS. Schedules each PCM chunk
// back-to-back on a Web Audio timeline; stop() cuts everything instantly (barge-in).
// Chunks carry their own sample rate (Supertonic = 44.1 kHz); the AudioContext
// resamples to the device rate automatically.

export class PlaybackQueue {
  private ctx: AudioContext | null = null
  private gain: GainNode | null = null
  private nextTime = 0
  private readonly sources = new Set<AudioBufferSourceNode>()
  private playingState = false

  /** Called when playback starts (true) or fully drains / is stopped (false). */
  onPlayingChange?: (playing: boolean) => void

  get playing(): boolean {
    return this.playingState
  }

  private ensure(): AudioContext {
    if (!this.ctx) {
      this.ctx = new AudioContext()
      this.gain = this.ctx.createGain()
      this.gain.connect(this.ctx.destination)
    }
    return this.ctx
  }

  /** Must be called from a user gesture to satisfy autoplay policy. */
  async resume(): Promise<void> {
    const ctx = this.ensure()
    if (ctx.state === 'suspended') await ctx.resume()
  }

  enqueue(pcm: Float32Array, sampleRate: number): void {
    const ctx = this.ensure()
    const buffer = ctx.createBuffer(1, pcm.length, sampleRate)
    buffer.getChannelData(0).set(pcm)

    const src = ctx.createBufferSource()
    src.buffer = buffer
    src.connect(this.gain as GainNode)

    const startAt = Math.max(this.nextTime, ctx.currentTime)
    src.start(startAt)
    this.nextTime = startAt + buffer.duration
    this.sources.add(src)
    this.setPlaying(true)

    src.onended = () => {
      this.sources.delete(src)
      if (this.sources.size === 0) this.setPlaying(false)
    }
  }

  /** Stop and close the AudioContext (session teardown). Each leaked running context
   *  counts against the browser's cap (Safari allows ~4), so dispose must release it.
   *  Idempotent; enqueue after dispose would just create a fresh context. */
  dispose(): void {
    this.stop()
    const ctx = this.ctx
    this.ctx = null
    this.gain = null
    if (ctx && ctx.state !== 'closed') void ctx.close().catch(() => undefined)
  }

  /** Stop and clear everything immediately (barge-in). */
  stop(): void {
    for (const src of this.sources) {
      src.onended = null
      try {
        src.stop()
      } catch {
        /* already stopped */
      }
    }
    this.sources.clear()
    this.nextTime = this.ctx?.currentTime ?? 0
    this.setPlaying(false)
  }

  private setPlaying(p: boolean): void {
    if (p !== this.playingState) {
      this.playingState = p
      this.onPlayingChange?.(p)
    }
  }
}
