// Orchestrator: wires mic → VAD → (Smart Turn) → ASR → LLM → TTS → speaker into a
// full conversational loop, with streaming overlap and barge-in.
//
// Lifecycle is split so models load ONCE and the conversation can be toggled:
//   load()           → download + initialize all workers, SEQUENTIALLY (one heavy
//                      model at a time, to stay under Safari's memory ceiling)
//   startListening() → turn the mic on (begin a session)
//   stopListening()  → mic off, but models stay resident (instant restart)
//   dispose()        → tear everything down

import { AutoGain } from '../audio/autoGain'
import { MicCapture } from '../audio/micCapture'
import { PlaybackQueue } from '../audio/playbackQueue'
import { pruneIncompleteAssets } from '../core/modelStore'
import { ConversationEngine } from '../engine/conversationEngine'
import { LLM, llmModelUrls, modelSource } from '../models/registry'
import type {
  AsrIn, AsrOut, Device, LlmIn, LlmOut, LoadProgress,
  TtsIn, TtsOut, TurnIn, TurnOut, VadIn, VadOut,
} from '../protocol/messages'

export type AgentState = 'cold' | 'loading' | 'ready' | 'idle' | 'listening' | 'thinking' | 'speaking'

export type LoadStatus = 'pending' | 'loading' | 'ready' | 'error'
export interface ComponentLoad {
  label: string
  title: string
  status: LoadStatus
  detail: string
  fraction: number
}

export interface OrchestratorCallbacks {
  onState?: (state: AgentState) => void
  onLoadStatus?: (components: ComponentLoad[]) => void
  onUserTranscript?: (text: string, final: boolean) => void
  onAssistantText?: (text: string, done: boolean) => void
  onLevel?: (rms: number) => void
  onError?: (where: string, message: string) => void
}

export interface OrchestratorOptions {
  device: Device // for TTS (LLM + ASR are always WebGPU; VAD/turn always WASM)
  /** Streaming makeup gain on the ASR signal - lifts a quiet mic to a healthy level.
   *  Defaults to false (the gain envelope can clip the first word). */
  micAutoGain?: boolean
  /** Shared-brain mode: reuse this already-loaded engine (and its LLM) instead of
   *  creating a new one - so text and voice share ONE model with continuous context.
   *  When set, the orchestrator loads only the speech models (VAD/ASR/TTS/Turn). */
  engine?: ConversationEngine
  callbacks?: OrchestratorCallbacks
}

const SYSTEM_PROMPT =
  'You are aidekin, a warm, concise voice assistant running entirely on the user\'s device. ' +
  'Keep replies short and conversational (one or two sentences) because they are spoken aloud. ' +
  'Always reply in English.'

// After the VAD declares end-of-speech we ask Smart Turn whether the turn is
// semantically complete. Smart Turn only ACCELERATES finalization - if it says
// "not complete" (or errors, or is slow), this fallback fires and we finalize
// anyway, unless the user resumed speaking. Never let one model wedge the loop.
// Kept tight (the VAD already waited ~1 s of silence via redemptionMs) so a
// declined/slow Smart Turn verdict doesn't add a long dead pause before replying.
const TURN_FALLBACK_MS = 600

// ~500 ms @ 16 kHz of mic audio kept BEFORE speech onset, so the first word is never
// clipped (VAD detection always lags the actual onset by a frame or two).
const PRE_BUFFER_SAMPLES = 8000

interface Waiter {
  resolve: () => void
  reject: (e: Error) => void
  optional: boolean
}

export class Orchestrator {
  private readonly cb: OrchestratorCallbacks
  private readonly device: Device
  private readonly autoGain = new AutoGain()
  private readonly useAutoGain: boolean

  // These must stay as literal `new Worker(new URL('…', import.meta.url),
  // { type: 'module' })` calls - that exact form is how Vite bundles each worker.
  private readonly vad = new Worker(new URL('../workers/vad.worker.ts', import.meta.url), { type: 'module' })
  private readonly asr = new Worker(new URL('../workers/asr.worker.ts', import.meta.url), { type: 'module' })
  private llm?: Worker // owned only in standalone mode; shared mode reuses the engine's LLM
  private readonly tts = new Worker(new URL('../workers/tts.worker.ts', import.meta.url), { type: 'module' })
  private readonly turn = new Worker(new URL('../workers/turn.worker.ts', import.meta.url), { type: 'module' })

  private readonly playback = new PlaybackQueue()
  private readonly engine: ConversationEngine
  private readonly ownsEngine: boolean
  private mic?: MicCapture

  private state: AgentState = 'cold'
  private loaded = false
  private disposed = false

  private inUserTurn = false
  private vadSpeaking = false
  private turnReady = false
  private muted = false
  // A speech-start arrived while the assistant was thinking/speaking. We do NOT cancel the
  // reply yet (a cough/table-knock also fires speech-start); we wait for the ASR to confirm
  // real words before barging in, and a VAD misfire clears this without touching the reply.
  private pendingBargeIn = false
  // Rolling ~500 ms of the most recent mic audio, always on, used to seed each new
  // ASR stream so the first word is never clipped by VAD onset latency.
  private preBuffer: Float32Array[] = []
  private preBufferLen = 0
  // Debug: exactly the audio the ASR streamed for the current turn (preBuffer seed +
  // live frames), stored on finalize → window.aidekinSaveAudio() saves that one utterance.
  private turnDebug: Float32Array[] = []
  private lastUtteranceAudio: Float32Array | null = null
  private finalizeTimer: ReturnType<typeof setTimeout> | null = null
  private asrUtteranceId = 0
  private currentAsrId = -1
  private ttsId = 0
  private readonly liveTtsIds = new Set<number>()

  private readonly waiters = new Map<string, Waiter>()

  private readonly loadMap = new Map<string, ComponentLoad>([
    ['LLM', { label: 'LLM', title: 'Brain · Bonsai', status: 'pending', detail: '@aidekin/webgpu-llm', fraction: 0 }],
    ['ASR', { label: 'ASR', title: 'Hearing · Nemotron 3.5', status: 'pending', detail: 'onnxruntime-web', fraction: 0 }],
    ['TTS', { label: 'TTS', title: 'Voice · Supertonic-3', status: 'pending', detail: 'onnxruntime-web', fraction: 0 }],
    ['VAD', { label: 'VAD', title: 'Activity · Silero v5', status: 'pending', detail: 'vad-web', fraction: 0 }],
    ['Turn', { label: 'Turn', title: 'Turn-taking · Smart Turn v3', status: 'pending', detail: 'onnxruntime-web / WASM', fraction: 0 }],
  ])

  constructor(opts: OrchestratorOptions) {
    this.cb = opts.callbacks ?? {}
    this.device = opts.device
    // Auto-gain DEFAULT OFF: verified to drop the first words (the gain envelope ramps
    // over the onset, distorting it) and it never helped accuracy anyway - quiet audio
    // transcribes fine; the real lever is mic noise suppression, not loudness.
    this.useAutoGain = opts.micAutoGain ?? false
    if (opts.engine) {
      // Shared-brain (widget) mode: reuse the widget's already-loaded engine + LLM so
      // there is ONE model with continuous context across text and voice. We own only
      // the speech workers; clause→TTS routing is attached on startListening().
      this.engine = opts.engine
      this.ownsEngine = false
      const llm = this.loadMap.get('LLM')
      if (llm) {
        llm.status = 'ready'
        llm.fraction = 1
        llm.detail = 'shared with chat'
      }
    } else {
      // Standalone (voice app) mode: own the LLM worker + engine, as before. The engine
      // adopts the worker after load and drives TTS via onAssistantClause.
      this.llm = new Worker(new URL('../workers/llm.worker.ts', import.meta.url), { type: 'module' })
      this.ownsEngine = true
      this.engine = new ConversationEngine({
        systemPrompt: SYSTEM_PROMPT,
        device: 'webgpu',
        chunkClauses: true,
        callbacks: {
          onAssistantText: (text, done) => this.cb.onAssistantText?.(text, done),
          onAssistantClause: (clause) => this.speak(clause),
          onGenerationStart: () => this.setState('thinking'),
          onGenerationEnd: () => this.settleAfterGeneration(),
          onError: (where, message) => this.reportError(where, message),
        },
      })
    }
    this.playback.onPlayingChange = (playing) => {
      if (!playing && this.state === 'speaking') this.setState('idle')
    }
    this.wireWorkers()
    // Debug hook: window.aidekinSaveAudio() downloads the last ~30 s of exactly the
    // 16 kHz audio the ASR received, so it can be run through scripts/asr-spike.ts.
    if (typeof window !== 'undefined') {
      ;(window as unknown as { aidekinSaveAudio?: () => void }).aidekinSaveAudio = () => this.saveDebugAudio()
    }
  }

  /** Download the recent mic audio for offline ASR debugging: the RAW 16 kHz the ASR
   *  receives, plus a peak-normalized (audible) copy to also test the quiet-level theory. */
  private saveDebugAudio(): void {
    const pcm = this.lastUtteranceAudio
    if (!pcm || pcm.length === 0) {
      console.warn('[aidekin] no utterance captured yet - speak a sentence first, then call aidekinSaveAudio()')
      return
    }
    const n = pcm.length
    let peak = 0
    let sumSq = 0
    for (let i = 0; i < n; i++) {
      const a = Math.abs(pcm[i])
      if (a > peak) peak = a
      sumSq += pcm[i] * pcm[i]
    }
    const rms = Math.sqrt(sumSq / Math.max(1, n))
    console.info(
      `[aidekin] captured ${(n / 16000).toFixed(1)}s · peak=${peak.toFixed(3)} rms=${rms.toFixed(4)} → ` +
        `aidekin-utterance.wav (raw) + aidekin-utterance-loud.wav (normalized)`,
    )
    this.downloadWav(pcm, 'aidekin-utterance.wav')
    const g = peak > 1e-4 ? 0.7 / peak : 1
    const loud = new Float32Array(n)
    for (let i = 0; i < n; i++) loud[i] = pcm[i] * g
    this.downloadWav(loud, 'aidekin-utterance-loud.wav')
  }

  private downloadWav(pcm: Float32Array, name: string): void {
    const n = pcm.length
    const buf = new ArrayBuffer(44 + n * 2)
    const v = new DataView(buf)
    const wstr = (o: number, s: string) => {
      for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i))
    }
    wstr(0, 'RIFF')
    v.setUint32(4, 36 + n * 2, true)
    wstr(8, 'WAVE')
    wstr(12, 'fmt ')
    v.setUint32(16, 16, true)
    v.setUint16(20, 1, true)
    v.setUint16(22, 1, true)
    v.setUint32(24, 16000, true)
    v.setUint32(28, 32000, true)
    v.setUint16(32, 2, true)
    v.setUint16(34, 16, true)
    wstr(36, 'data')
    v.setUint32(40, n * 2, true)
    for (let i = 0; i < n; i++) {
      const s = Math.max(-1, Math.min(1, pcm[i]))
      v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true)
    }
    const url = URL.createObjectURL(new Blob([buf], { type: 'audio/wav' }))
    const a = document.createElement('a')
    a.href = url
    a.download = name
    a.click()
    URL.revokeObjectURL(url)
  }

  /** Download + initialize all models, one heavy model at a time. */
  async load(): Promise<void> {
    if (this.loaded) {
      this.setState('ready')
      return
    }
    this.setState('loading')
    this.emitLoad()

    // Self-heal: drop any partial weights from a previously interrupted download (tab
    // closed / worker killed mid-stream) so they can't accumulate as orphaned OPFS bytes.
    const pruned = await pruneIncompleteAssets().catch(() => 0)
    if (pruned > 0) console.info(`[aidekin] pruned ${pruned} incomplete model file(s) from a prior interrupted download`)

    // Phase 1 - DOWNLOAD the heavy speech weights (ASR ~690 MB + TTS ~360 MB) in PARALLEL.
    // Each worker streams its own files to the OPFS cache without creating any sessions, so the
    // two big downloads overlap (and within each, the files download concurrently) instead of
    // running ASR-then-TTS in series. Streaming to disk keeps peak memory low, so parallel
    // downloading is safe; the memory-heavy part (session creation) stays serial in phase 2.
    await Promise.all([
      this.prefetchWorker(this.asr, { kind: 'prefetch', modelBase: modelSource('asr') }, 'ASR'),
      this.prefetchWorker(this.tts, { kind: 'prefetch', modelBase: modelSource('tts') }, 'TTS'),
    ])

    // Phase 2 - INITIALIZE one model at a time (reads from the cache warmed above, so this is
    // CPU/GPU-bound, not network-bound). Serial bounds peak GPU/WASM memory to one model
    // (Safari's ceiling). VAD/Turn are tiny and the LLM uses its own cache, so they just init.
    await this.initWorker(this.vad, { kind: 'init', assetBase: modelSource('vad') }, 'VAD', false)
    await this.initWorker(this.turn, { kind: 'init', modelBase: '/models/turn' }, 'Turn', true)
    // ASR: the FP16 Nemotron, encoder on WebGPU (real-time). Single engine - WebGPU
    // is required (as it is for the LLM). Reads from the OPFS cache warmed in phase 1.
    await this.initWorker(this.asr, { kind: 'init', modelBase: modelSource('asr'), device: 'webgpu' }, 'ASR', false)
    await this.initWorker(this.tts, { kind: 'init', modelBase: modelSource('tts'), device: this.device }, 'TTS', false)
    // Brain: Bonsai on our @aidekin/webgpu-llm engine / WebGPU (data streams from the HF Hub, caches
    // to OPFS; manifest + aux served same-origin).
    // Brain: only in standalone mode. Shared mode reuses the widget engine's LLM.
    if (this.llm) {
      const u = llmModelUrls()
      const llmInit: LlmIn = { kind: 'init', manifestUrl: u.manifestUrl, dataUrl: u.dataUrl, auxUrl: u.auxUrl, tokenizerModelId: LLM.tokenizerModelId, eosTokenId: LLM.eosTokenId, maxSeqLen: LLM.maxSeqLen }
      await this.initWorker(this.llm, llmInit, 'LLM', false)
      // Worker is loaded + initialized - hand it to the engine, which now drives
      // generation. onLlm forwards token/done events to engine.handleLlmMessage().
      this.engine.adoptLlmWorker(this.llm)
    }

    this.loaded = true
    this.setState('ready')
  }

  get isLoaded(): boolean {
    return this.loaded
  }

  /** Begin a listening session (mic on). Models must already be loaded. */
  async startListening(): Promise<void> {
    if (!this.loaded) throw new Error('Models are not loaded yet')
    await this.playback.resume()
    if (this.ownsEngine) {
      this.engine.reset() // standalone: a fresh conversation each listening session
    } else {
      // Shared engine: keep the conversation context; route the stream to TTS while
      // listening, and re-enable clause chunking (text mode turned it off).
      this.engine.setChunkClauses(true)
      this.engine.setClauseSink((clause) => this.speak(clause))
    }
    this.mic = new MicCapture({
      frameSize: 512,
      noiseSuppression: true,
      onFrame: (f) => this.onMicFrame(f.samples),
      onLevel: (rms) => this.cb.onLevel?.(rms),
    })
    await this.mic.start()
    this.setState('idle')
  }

  /** Stop listening but KEEP the models loaded, so restart is instant. */
  async stopListening(): Promise<void> {
    await this.mic?.stop()
    this.mic = undefined
    this.clearFinalizeTimer()
    this.cancelInFlight()
    this.inUserTurn = false
    this.vadSpeaking = false
    this.preBuffer = []
    this.preBufferLen = 0
    // Shared engine: return it to text mode so typed replies don't get spoken.
    if (!this.ownsEngine) {
      this.engine.setChunkClauses(false)
      this.engine.setClauseSink(null)
    }
    this.setState('ready')
  }

  /** Mute/unmute the mic without ending the session. Muted = mic frames are dropped (no VAD /
   *  ASR), so the assistant stops listening; unmuting resumes instantly. Does not interrupt an
   *  in-progress reply. */
  setMuted(muted: boolean): void {
    this.muted = muted
    if (muted) {
      // Discard any half-captured utterance so it isn't finalized after unmuting.
      this.clearFinalizeTimer()
      this.inUserTurn = false
      this.vadSpeaking = false
      if (this.state === 'listening') this.setState('idle')
    }
  }

  async dispose(): Promise<void> {
    if (this.disposed) return // idempotent: abandon + unmount can both fire
    this.disposed = true
    this.clearFinalizeTimer()
    // Detach worker handlers FIRST so a message queued just before terminate() can't fire a
    // stale callback (setState / onUserTranscript / onState) on a now-disposed orchestrator.
    for (const w of [this.vad, this.asr, this.llm, this.tts, this.turn]) {
      if (w) {
        w.onmessage = null
        w.onerror = null
        w.onmessageerror = null
      }
    }
    // Reject any pending worker-init waiters so an in-flight load() rejects instead of hanging
    // (e.g. the user abandoned voice mid-download - see useTextController.toggleVoice). Snapshot
    // first: a reject microtask must not mutate the Map mid-iteration.
    const pending = [...this.waiters.entries()]
    this.waiters.clear()
    for (const [label, w] of pending) w.reject(new Error(`${label}: voice load cancelled`))
    await this.mic?.stop()
    this.playback.stop()
    if (this.ownsEngine) {
      this.engine.dispose()
    } else {
      // Shared engine belongs to the widget - just detach voice, don't tear it down.
      this.engine.abort()
      this.engine.setChunkClauses(false)
      this.engine.setClauseSink(null)
    }
    // Terminating the workers aborts any in-flight model download - so abandoning voice
    // mid-load stops the ~1.6 GB transfer immediately instead of draining in the background.
    for (const w of [this.vad, this.asr, this.llm, this.tts, this.turn]) w?.terminate()
    // Then sweep any partially-written weights the terminated workers left (no .done marker),
    // so an abandoned install leaves no garbage on the device. Best-effort; also runs on next load.
    await pruneIncompleteAssets().catch(() => 0)
    this.setState('cold')
  }

  private initWorker(w: Worker, msg: VadIn | AsrIn | TtsIn | TurnIn | LlmIn, label: string, optional: boolean): Promise<void> {
    return new Promise<void>((resolve, reject) => {
      this.waiters.set(label, { resolve, reject, optional })
      this.post(w, msg)
    })
  }

  /** Download a worker's model files to the OPFS cache WITHOUT creating sessions, so several
   *  workers can download in parallel before the serial init phase. Resolves on 'prefetched'.
   *  Reuses the per-label waiter (prefetch and init never overlap for one worker), so a failure
   *  during prefetch rejects through the same path as an init failure. */
  private prefetchWorker(w: Worker, msg: AsrIn | TtsIn, label: string): Promise<void> {
    return new Promise<void>((resolve, reject) => {
      this.waiters.set(label, { resolve, reject, optional: false })
      this.post(w, msg)
    })
  }

  // ── mic → VAD (+ live ASR stream) ─────────────────────────────────────────
  // The mic drives the VAD gate, fills the rolling pre-onset buffer, and - while a
  // turn is active - STREAMS frames to the ASR worker so transcription happens live.
  // Software echo handling: browser AEC is OFF (it hurts ASR), so we drop mic frames while our
  // TTS is audible OR a clause is still queued/synthesizing (liveTtsIds). Gating on liveTtsIds
  // (not just playback.playing) closes the gap BETWEEN clauses, where playback.playing briefly
  // flips false and the mic would otherwise hear the next clause and false-trigger the VAD.
  // Barge-in while THINKING still works (no TTS is pending then); mid-speech interrupt is the
  // mute button's job (true interrupt-during-speech needs acoustic echo cancellation).
  private onMicFrame(samples: Float32Array): void {
    if (this.playback.playing || this.liveTtsIds.size > 0) return
    const keep = samples.slice() // retained in the rolling pre-onset buffer
    this.pushPreBuffer(keep)
    if (this.muted) return // muted: keep the pre-onset buffer fresh, but do not listen
    if (this.inUserTurn) {
      // Auto-gain the ASR copy only (NOT the VAD frame - boosting room tone would
      // false-trigger the gate). process() may return `samples` itself when no boost is
      // needed, so slice() to get a transferable buffer that won't detach `samples`.
      const gained = this.useAutoGain ? this.autoGain.process(samples) : samples
      const forAsr = gained.slice()
      this.turnDebug.push(forAsr.slice()) // debug copy BEFORE we transfer forAsr.buffer
      this.post(this.asr, { kind: 'chunk', id: this.currentAsrId, samples: forAsr }, [forAsr.buffer])
    }
    this.post(this.vad, { kind: 'frame', samples }, [samples.buffer])
  }

  private pushPreBuffer(frame: Float32Array): void {
    this.preBuffer.push(frame)
    this.preBufferLen += frame.length
    while (this.preBufferLen > PRE_BUFFER_SAMPLES && this.preBuffer.length > 1) {
      this.preBufferLen -= (this.preBuffer.shift() as Float32Array).length
    }
  }

  private wireWorkers(): void {
    this.vad.onmessage = (e: MessageEvent<VadOut>) => this.onVad(e.data)
    this.asr.onmessage = (e: MessageEvent<AsrOut>) => this.onAsr(e.data)
    if (this.llm) this.llm.onmessage = (e: MessageEvent<LlmOut>) => this.onLlm(e.data)
    this.tts.onmessage = (e: MessageEvent<TtsOut>) => this.onTts(e.data)
    this.turn.onmessage = (e: MessageEvent<TurnOut>) => this.onTurn(e.data)

    const pairs: ReadonlyArray<readonly [Worker, string]> = [
      [this.vad, 'VAD'], [this.asr, 'ASR'], [this.tts, 'TTS'], [this.turn, 'Turn'],
      ...(this.llm ? ([[this.llm, 'LLM']] as ReadonlyArray<readonly [Worker, string]>) : []),
    ]
    for (const [w, label] of pairs) {
      w.onerror = (e: ErrorEvent) => this.reportError(label, e.message || 'worker crashed at load')
      w.onmessageerror = () => this.reportError(label, 'worker message (de)serialization error')
    }
  }

  private onVad(m: VadOut): void {
    if (this.lifecycle('VAD', m)) return
    if (m.kind === 'speech-start') {
      console.info('[aidekin] VAD speech-start')
      // The user (re)started talking - cancel any pending end-of-turn finalize.
      this.clearFinalizeTimer()
      // Defer barge-in: a transient noise (cough, table knock, door) also fires speech-start
      // and must NOT kill an in-flight reply. Mark it pending and start capturing; we only
      // actually cancel the reply once the ASR confirms real words (onAsr), and a VAD misfire
      // clears the pending flag without ever touching the reply.
      if (this.state === 'speaking' || this.state === 'thinking') this.pendingBargeIn = true
      if (!this.inUserTurn) this.beginUserTurn()
      this.vadSpeaking = true
    } else if (m.kind === 'speech-end') {
      console.info(`[aidekin] VAD speech-end · ${m.durationMs.toFixed(0)}ms of speech`)
      this.vadSpeaking = false
      if (!this.inUserTurn) return // no active turn
      // The full turn audio lives in this.turnAudio; m.audio is just the recent
      // speech segment, which is exactly what Smart Turn wants to judge the ending.
      if (this.turnReady) {
        this.post(this.turn, { kind: 'analyze', id: this.currentAsrId, samples: m.audio }, [m.audio.buffer])
        this.armFinalizeFallback()
      } else {
        this.finalizeUserTurn()
      }
    } else if (m.kind === 'misfire') {
      // Silero retracted a too-short blip (a cough/click/table-knock under minSpeechMs). It was
      // NOT real speech, so drop the provisional turn WITHOUT barging in - this is what stops a
      // stray noise from killing an in-flight reply (cancelProvisionalTurn clears pendingBargeIn
      // and never touches the assistant). The orb resyncs to whatever the reply is doing.
      console.info('[aidekin] VAD misfire (too-short blip) - cancelling provisional turn')
      this.cancelProvisionalTurn()
    }
  }

  private onTurn(m: TurnOut): void {
    if (this.lifecycle('Turn', m)) return
    if (m.kind === 'verdict') {
      console.info(`[aidekin] Smart Turn verdict · complete=${m.complete} p=${m.prob.toFixed(3)}`)
      if (this.inUserTurn && m.complete) this.finalizeUserTurn()
      // not-complete → keep listening; the fallback timer still guarantees finalize.
    }
  }

  private armFinalizeFallback(): void {
    this.clearFinalizeTimer()
    this.finalizeTimer = setTimeout(() => {
      this.finalizeTimer = null
      if (this.inUserTurn && !this.vadSpeaking) {
        console.info('[aidekin] turn fallback fired - finalizing without a complete verdict')
        this.finalizeUserTurn()
      }
    }, TURN_FALLBACK_MS)
  }

  private clearFinalizeTimer(): void {
    if (this.finalizeTimer !== null) {
      clearTimeout(this.finalizeTimer)
      this.finalizeTimer = null
    }
  }

  private finalizeUserTurn(): void {
    this.clearFinalizeTimer()
    if (!this.inUserTurn) return
    this.inUserTurn = false
    // Store exactly this turn's audio for debug capture (window.aidekinSaveAudio()).
    let dn = 0
    for (const c of this.turnDebug) dn += c.length
    const utter = new Float32Array(dn)
    let doff = 0
    for (const c of this.turnDebug) {
      utter.set(c, doff)
      doff += c.length
    }
    this.lastUtteranceAudio = utter
    this.turnDebug = []
    // The stream was fed live; just flush the sub-chunk tail for the final transcript.
    console.info(`[aidekin] finalize turn - flushing ASR (id=${this.currentAsrId})`)
    this.post(this.asr, { kind: 'flush', id: this.currentAsrId })
  }

  private onAsr(m: AsrOut): void {
    if (this.lifecycle('ASR', m)) return
    if (m.kind === 'partial') {
      // Drop partials from a superseded stream (a new turn already started).
      if (m.id !== undefined && m.id !== this.currentAsrId) return
      // First real words while a reply is in flight: NOW it is a genuine interruption, so
      // barge-in (stop the previous playback/TTS/generation). A noise never reaches here
      // because it produces no transcript - that is what makes barge-in robust to knocks.
      if (this.pendingBargeIn && m.text.trim()) {
        this.bargeIn()
        this.pendingBargeIn = false
      }
      this.cb.onUserTranscript?.(m.text, false)
    } else if (m.kind === 'final') {
      // Always apply a final (show the transcript) so the user's words are never lost, even
      // when turns arrive back-to-back. Latest-RESPONSE-wins is handled in the engine instead
      // (a new generation aborts the in-flight one in generate()), so rapid turns don't pile
      // up on the LLM - without dropping any spoken text.
      this.cb.onUserTranscript?.(m.text, true)
      const text = m.text.trim()
      console.info(`[aidekin] ASR final: "${text}"`)
      if (text) {
        // Real transcript: confirm the interruption (a fast one-word reply may not have
        // produced a partial first) so the previous reply's audio/generation is stopped.
        if (this.pendingBargeIn) {
          this.bargeIn()
          this.pendingBargeIn = false
        }
        if (this.ownsEngine) {
          void this.engine.sendUserMessage(text) // engine callbacks drive state
        } else {
          // Shared engine carries the widget's callbacks (not ours) - drive orb state here.
          this.setState('thinking')
          void this.engine
            .sendUserMessage(text)
            .then(() => this.settleAfterGeneration())
            .catch(() => undefined)
        }
      } else {
        // Empty transcript = a noise/false-trigger turn. Do NOT barge in; clear any pending
        // barge-in and resync the orb to what the assistant is actually doing, so a stray
        // noise never hides an in-flight reply behind "Listening". (The empty placeholder
        // bubble is dropped separately in the controller.)
        this.pendingBargeIn = false
        this.resyncOrbState()
      }
    }
  }

  private onLlm(m: LlmOut): void {
    // During load the orchestrator owns the lifecycle (waiters + load UI). After load, a
    // generation 'error' must reach the engine so the in-flight turn settles instead of
    // hanging on "thinking".
    if (m.kind === 'error' && this.loaded && this.engine.isGenerating) {
      this.engine.handleLlmMessage(m)
      return
    }
    if (this.lifecycle('LLM', m)) return
    // The shared engine owns generation state (ids, streaming, history) and drives
    // playback via the onAssistantClause callback wired in the constructor.
    this.engine.handleLlmMessage(m)
  }

  private onTts(m: TtsOut): void {
    if (this.lifecycle('TTS', m)) return
    if (m.kind === 'audio') {
      if (!this.liveTtsIds.has(m.id)) return
      console.info(`[aidekin] TTS audio id=${m.id} · ${m.pcm.length} samples @ ${m.sampleRate}Hz → playback`)
      this.setState('speaking')
      this.playback.enqueue(m.pcm, m.sampleRate)
    } else if (m.kind === 'done') {
      this.liveTtsIds.delete(m.id)
      // If generation already finished and nothing is left to play, settle to idle.
      if (!this.engine.isGenerating && this.liveTtsIds.size === 0 && !this.playback.playing && this.state === 'speaking') {
        this.setState('idle')
      }
    }
  }

  // ── turn lifecycle ────────────────────────────────────────────────────────
  private beginUserTurn(): void {
    this.inUserTurn = true
    this.currentAsrId = ++this.asrUtteranceId
    this.autoGain.reset() // fresh gain envelope per utterance
    // Start a fresh ASR stream and seed it with the rolling pre-onset audio (gained
    // the same way as live frames) so the first word isn't clipped; subsequent frames
    // stream live from onMicFrame.
    this.post(this.asr, { kind: 'reset' })
    const seeded: Float32Array[] = []
    for (const frame of this.preBuffer) {
      const seed = (this.useAutoGain ? this.autoGain.process(frame) : frame).slice()
      seeded.push(seed.slice())
      this.post(this.asr, { kind: 'chunk', id: this.currentAsrId, samples: seed }, [seed.buffer])
    }
    this.turnDebug = seeded // debug capture mirrors exactly what the ASR received
    this.cb.onUserTranscript?.('', false)
    this.setState('listening')
  }

  private settleAfterGeneration(): void {
    // A superseded turn resolves its promise too; don't settle to idle if a newer generation
    // is already running (it owns the state now).
    if (this.engine.isGenerating) return
    // Generation finished naturally: if nothing is queued or playing, settle to idle.
    if (this.liveTtsIds.size === 0 && !this.playback.playing && this.state !== 'cold' && this.state !== 'ready') {
      this.setState('idle')
    }
  }

  private speak(text: string): void {
    if (!text.trim()) return
    // The reply is about to talk, which echo-gates the mic (onMicFrame). A still-open
    // speculative provisional turn (waiting to confirm a barge-in) could no longer conclude
    // via the VAD, so resolve it as a non-interruption now to avoid a hung "..." placeholder.
    // True interrupt-during-speech remains the mute button's job.
    if (this.pendingBargeIn) this.cancelProvisionalTurn()
    const id = ++this.ttsId
    this.liveTtsIds.add(id)
    this.post(this.tts, { kind: 'speak', id, text })
  }

  /** Drop a provisional user turn that turned out not to be a real interruption (a VAD misfire,
   *  an empty transcript, or the reply starting to speak before words were confirmed): clear the
   *  pending barge-in, reset the ASR stream, drop the "..." placeholder, and resync the orb. The
   *  assistant's in-flight reply is deliberately left untouched. */
  private cancelProvisionalTurn(): void {
    this.pendingBargeIn = false
    this.clearFinalizeTimer()
    this.vadSpeaking = false
    if (this.inUserTurn) {
      this.inUserTurn = false
      this.post(this.asr, { kind: 'reset' })
      this.cb.onUserTranscript?.('', true) // empty final drops the placeholder bubble
    }
    this.resyncOrbState()
  }

  /** Set the orb to what the assistant is actually doing right now: thinking if the LLM is
   *  generating, speaking if audio is queued/playing, else idle. Used after a misfire or a noise
   *  turn so a speculative provisional turn doesn't leave the orb stuck on "Listening". */
  private resyncOrbState(): void {
    if (this.engine.isGenerating) {
      this.setState('thinking')
    } else if (this.playback.playing || this.liveTtsIds.size > 0) {
      this.setState('speaking')
    } else if (this.state !== 'cold' && this.state !== 'ready') {
      this.setState('idle')
    }
  }

  private bargeIn(): void {
    this.cancelInFlight()
  }

  private cancelInFlight(): void {
    this.playback.stop()
    this.engine.abort()
    for (const id of this.liveTtsIds) this.post(this.tts, { kind: 'abort', id })
    this.liveTtsIds.clear()
  }

  // ── load lifecycle helpers ────────────────────────────────────────────────
  private lifecycle(label: string, m: { kind: string }): boolean {
    const c = this.loadMap.get(label)
    if (m.kind === 'load') {
      const l = m as LoadProgress
      if (c) {
        c.status = 'loading'
        c.detail = l.detail
        c.fraction = l.total > 0 ? Math.min(1, l.loaded / l.total) : c.fraction
        this.emitLoad()
      }
      return true
    }
    if (m.kind === 'prefetched') {
      // Phase-1 download for this worker finished (no session created yet) - resolve its
      // prefetch waiter so load() can proceed to the serial init phase.
      const w = this.waiters.get(label)
      this.waiters.delete(label)
      w?.resolve()
      return true
    }
    if (m.kind === 'ready') {
      const r = m as Extract<VadOut, { kind: 'ready' }>
      if (c) {
        c.status = 'ready'
        c.fraction = 1
        c.detail = r.info ?? 'ready'
        this.emitLoad()
      }
      if (label === 'Turn') this.turnReady = true
      const w = this.waiters.get(label)
      this.waiters.delete(label)
      w?.resolve()
      return true
    }
    if (m.kind === 'error') {
      this.reportError(label, (m as Extract<VadOut, { kind: 'error' }>).message)
      return true
    }
    return false
  }

  private reportError(label: string, message: string): void {
    this.cb.onError?.(label, message)
    const c = this.loadMap.get(label)
    if (c && c.status !== 'ready') {
      c.status = 'error'
      c.detail = message
      this.emitLoad()
    }
    const w = this.waiters.get(label)
    if (w) {
      this.waiters.delete(label)
      if (w.optional) {
        // Smart Turn is optional - degrade to VAD-only turn ending (turnReady stays false).
        w.resolve()
      } else {
        w.reject(new Error(`${label} failed to load: ${message}`))
      }
    }
  }

  private emitLoad(): void {
    this.cb.onLoadStatus?.([...this.loadMap.values()])
  }

  private setState(s: AgentState): void {
    if (s !== this.state) {
      this.state = s
      this.cb.onState?.(s)
    }
  }

  private post(w: Worker, msg: VadIn | AsrIn | LlmIn | TtsIn | TurnIn, transfer: Transferable[] = []): void {
    w.postMessage(msg, transfer)
  }
}
