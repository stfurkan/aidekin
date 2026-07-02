// Microphone capture → 16 kHz mono Float32 PCM frames, via an AudioWorklet.
// We request a 16 kHz AudioContext so the browser resamples upstream (high quality);
// the worklet falls back to an anti-aliased windowed-sinc resampler otherwise.

import workletUrl from './pcmWorklet.js?url'

export interface MicFrame {
  /** frameSize samples of 16 kHz mono PCM in [-1, 1]. */
  readonly samples: Float32Array
  /** RMS level of this frame, for level meters. */
  readonly rms: number
}

export interface MicCaptureOptions {
  /** Samples per emitted frame at 16 kHz (default 512 = 32 ms). */
  readonly frameSize?: number
  /**
   * Enable the browser's noise suppression. Default true. Real laptop-mic capture in a
   * normal room is often only 15-25 dB SNR, so removing background noise meaningfully
   * improves ASR accuracy. Set false for a clean/studio mic where suppression's spectral
   * artifacts would otherwise hollow the voice.
   */
  readonly noiseSuppression?: boolean
  readonly onFrame: (frame: MicFrame) => void
  readonly onLevel?: (rms: number) => void
}

export class MicCapture {
  readonly targetRate = 16000
  private readonly opts: MicCaptureOptions
  private ctx?: AudioContext
  private node?: AudioWorkletNode
  private stream?: MediaStream
  private source?: MediaStreamAudioSourceNode
  // stop() during an in-flight start() must still release the mic: the pending
  // getUserMedia is tracked so its tracks can be stopped the moment it resolves,
  // and `stopped` makes start() bail out of wiring after each await.
  private pendingStream?: Promise<MediaStream>
  private stopped = false

  /** Actual context sample rate (16000 if the browser honored the request). */
  contextRate = 0
  /** True if the worklet is resampling (i.e. context rate ≠ 16 kHz). */
  resamplingInWorklet = false

  constructor(opts: MicCaptureOptions) {
    this.opts = opts
  }

  get running(): boolean {
    return this.ctx !== undefined
  }

  async start(): Promise<void> {
    if (this.ctx || this.stopped) return // single-shot: a stopped capture never restarts
    // Mic DSP policy (ASR-tuned, not telephony-tuned):
    //  - autoGainControl OFF: it over-amplifies close speech and CLIPS it (peak≈1.0 →
    //    broadband distortion), and boosting signal+noise together doesn't improve SNR.
    //  - echoCancellation OFF: its spectral suppression hollows the voice; barge-in echo
    //    is handled in software (mic suppressed while our TTS plays - see PlaybackQueue).
    //  - noiseSuppression: configurable, DEFAULT ON. Real laptop-mic-in-a-room capture is
    //    often only 15-25 dB SNR, so removing background noise improves ASR accuracy. Turn
    //    it off only for a clean/studio mic where suppression artifacts would hurt instead.
    const noiseSuppression = this.opts.noiseSuppression ?? true
    this.pendingStream = navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: false,
        noiseSuppression,
        autoGainControl: false,
      },
    })
    let stream: MediaStream
    try {
      stream = await this.pendingStream
    } finally {
      this.pendingStream = undefined
    }
    // stop() arrived while getUserMedia was pending: release the tracks now (nothing
    // else holds them yet) and do not wire the graph.
    if (this.stopped) {
      stream.getTracks().forEach((t) => t.stop())
      return
    }
    this.stream = stream
    // Confirm the device actually honored the constraints (some force DSP on).
    const settings = this.stream.getAudioTracks()[0]?.getSettings?.() ?? {}
    console.info(
      `[aidekin] mic DSP · echoCancellation=${settings.echoCancellation} ` +
        `autoGainControl=${settings.autoGainControl} noiseSuppression=${settings.noiseSuppression}`,
    )

    // Capture at the device's NATIVE rate and resample to 16 kHz with OUR Lanczos in
    // the worklet - do NOT force a 16 kHz AudioContext. Forcing it makes the BROWSER
    // resample, and its built-in resampler can alias enough to degrade the Nemotron
    // encoder; our anti-aliased Lanczos resample is verified clean (headless on 48→16
    // and 44.1→16), so we keep capture on our own path.
    this.ctx = new AudioContext()
    this.contextRate = this.ctx.sampleRate

    await this.ctx.audioWorklet.addModule(workletUrl)
    if (this.stopped) return // stop() already released the stream and closed the context

    this.source = this.ctx.createMediaStreamSource(this.stream)
    this.node = new AudioWorkletNode(this.ctx, 'pcm-capture', {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      outputChannelCount: [1],
      processorOptions: { frameSize: this.opts.frameSize ?? 512 },
    })

    this.node.port.onmessage = (ev: MessageEvent) => {
      const d = ev.data as
        | { type: 'meta'; inRate: number; targetRate: number; resampling: boolean }
        | { type: 'frame'; samples: Float32Array; rms: number }
      if (d.type === 'meta') {
        this.contextRate = d.inRate
        this.resamplingInWorklet = d.resampling
        console.info(
          `[aidekin] mic capture · contextRate=${d.inRate}Hz → ${d.targetRate}Hz · ` +
            `resampling-in-worklet=${d.resampling} (false = browser did high-quality resample)`,
        )
      } else {
        this.opts.onFrame({ samples: d.samples, rms: d.rms })
        this.opts.onLevel?.(d.rms)
      }
    }

    // Keep the node in the render graph (output is silent - we never write to it).
    this.source.connect(this.node)
    this.node.connect(this.ctx.destination)

    if (this.ctx.state === 'suspended') await this.ctx.resume()
  }

  async stop(): Promise<void> {
    this.stopped = true
    // A start() still awaiting getUserMedia stops its own tracks on resolve (see start),
    // but stop() must not resolve while the mic could still be live - so await it here
    // and stop the tracks too (double-stopping a track is harmless).
    if (this.pendingStream) {
      await this.pendingStream.then((s) => s.getTracks().forEach((t) => t.stop())).catch(() => undefined)
    }
    try {
      this.source?.disconnect()
      this.node?.disconnect()
      this.stream?.getTracks().forEach((t) => t.stop())
      await this.ctx?.close()
    } finally {
      this.ctx = undefined
      this.node = undefined
      this.source = undefined
      this.stream = undefined
    }
  }
}
