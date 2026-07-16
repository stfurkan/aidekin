// React hook wrapping the ConversationEngine for the TEXT widget. Owns the engine,
// projects its history into UI turns, streams the assistant reply, and gates model
// loading: the ~237 MB LLM downloads on first open (loadOnMount) or first send -
// nothing heavy loads before the widget is opened.

import { useCallback, useEffect, useRef, useState, type RefObject } from 'react'
import { ConversationEngine } from '@/engine/conversationEngine'
import type { AgentState, ComponentLoad, Orchestrator } from '@/pipeline/orchestrator'
import { requestPersist, type ClearResult } from '@/core/storage'
import { resolveSystemPrompt, type WidgetConfig } from './protocol'

export type WidgetStatus = 'cold' | 'loading' | 'ready' | 'thinking' | 'error'

export interface WidgetTurn {
  id: number
  role: 'user' | 'assistant' | 'error'
  text: string
}

export interface TextController {
  turns: WidgetTurn[]
  status: WidgetStatus
  loadPct: number
  loadDetail: string
  /** True once the model is cached locally → "Loading" instead of "Downloading". */
  cached: boolean
  error: string | null
  send: (text: string) => void
  /** Begin loading the brain now (call when the user enters a conversation, so opening to
   *  the mode picker does not download the model before a choice is made). */
  preload: () => void
  /** Abort the in-flight generation (the "stop" button while thinking). */
  stop: () => void
  /** Re-attempt model load after an error. */
  retry: () => void
  clear: () => void
  /** True once a reply MEASURED under reading speed (<6 tok/s) - the UI sets expectations. */
  slowDevice: boolean
  /** Unload the model from memory, clear its on-disk cache, and reset to 'cold' so the
   *  next message re-downloads. */
  forgetModel: () => Promise<ClearResult>
  // ── voice (lazy: the speech pipeline loads on the first toggle) ──────────────
  /** Orb state while voice is active ('listening' | 'thinking' | 'speaking' | …). */
  voiceState: AgentState
  voiceActive: boolean
  /** Speech-model (ASR/TTS/VAD/Turn) download progress, 0-1. */
  voiceLoadPct: number
  /** Whether the speech weights are already on this device. null = still checking (shows a
   *  neutral message); true = fast read from cache; false = needs the ~1.6 GB download. Drives
   *  the loading copy, with the null state avoiding a "downloading" flash before the check. */
  voiceCached: boolean | null
  levelRef: RefObject<number>
  toggleVoice: () => void
  /** Voice mic muted: frames are dropped so the assistant stops listening. */
  muted: boolean
  toggleMute: () => void
}

interface Options {
  /** Begin downloading the LLM as soon as the widget mounts (= opens). Default true. */
  loadOnMount?: boolean
  /** localStorage key for per-site history persistence (omit = no persistence). */
  persistKey?: string
  /** Notified for each user/assistant message (host analytics bridge). */
  onMessage?: (role: 'user' | 'assistant', text: string) => void
  onError?: (where: string, message: string) => void
}

export function useTextController(config: WidgetConfig, opts: Options = {}): TextController {
  const [turns, setTurns] = useState<WidgetTurn[]>([])
  const [status, setStatus] = useState<WidgetStatus>('cold')
  const [loadPct, setLoadPct] = useState(0)
  const [loadDetail, setLoadDetail] = useState('')
  const [cached, setCached] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [voiceState, setVoiceState] = useState<AgentState>('cold')
  const [voiceActive, setVoiceActive] = useState(false)
  const [voiceLoadPct, setVoiceLoadPct] = useState(0)
  const [voiceCached, setVoiceCached] = useState<boolean | null>(null)
  const [slowDevice, setSlowDevice] = useState(false)
  const [muted, setMuted] = useState(false)

  // Has the LLM been downloaded before? Check the SAME OPFS cache + key the worker writes
  // (modelStore.getModelAssetStream('llm-bonsai-1.7b-q1_0-gguf')). The old opfsModelCache.hasLlmCache()
  // looked in a different dir (aidekin-llm-cache) that nothing writes to, so it always returned false -
  // which made repeat visits say "Downloading" and disabled "Remove downloaded model" before a mode pick.
  useEffect(() => {
    void import('@/core/modelStore')
      .then(({ hasModelAsset }) => hasModelAsset('llm-bonsai-1.7b-q1_0-gguf'))
      .then(setCached)
      .catch(() => undefined)
  }, [])

  const engineRef = useRef<ConversationEngine | null>(null)
  const loadingRef = useRef<Promise<void> | null>(null)
  const streamingId = useRef<number | null>(null)
  const seq = useRef(0)
  const pendingDispose = useRef<ReturnType<typeof setTimeout> | null>(null)
  const orchRef = useRef<Orchestrator | null>(null)
  // Voice activation generation: bumped on every toggle. An in-flight activation checks
  // it after each await, so toggling off mid-load cancels instead of resurrecting voice.
  const voiceGen = useRef(0)
  const userStreamingId = useRef<number | null>(null)
  const levelRef = useRef(0)
  // Device-speed tracking for the "slower side" banner: the best decode rate seen and how many
  // replies have completed, so the cold first turn can't mislabel a healthy device (see onGenerationEnd).
  const bestTps = useRef(0)
  const perfTurns = useRef(0)
  // Latest callbacks, read by the (mount-once) engine so it never goes stale. Written in an
  // effect (the sanctioned latest-ref pattern): the engine only reads it asynchronously.
  const cbRef = useRef(opts)
  useEffect(() => {
    cbRef.current = opts
  })

  const nextId = () => ++seq.current

  // ── create the engine once; survive StrictMode's dev double-mount ────────────
  // StrictMode runs setup → cleanup → setup. If we disposed the worker on the first
  // cleanup and made a new engine on the second setup, we'd run TWO model downloads
  // and the worker could answer one engine while send() targets another (tokens then
  // get dropped on an id mismatch → reply generated but never shown). So we defer
  // disposal: a synchronous remount cancels it and reuses the SAME single engine.
  useEffect(() => {
    if (pendingDispose.current) {
      clearTimeout(pendingDispose.current)
      pendingDispose.current = null
    }

    if (!engineRef.current) {
      const engine = new ConversationEngine({
        systemPrompt: resolveSystemPrompt(config),
        brandName: config.title,
        retriever: null, // RAG attaches later via setRetriever when a knowledge URL is set
        ragTopK: config.ragTopK,
        chunkClauses: false,
        reasoning: config.reasoning,
        persistKey: opts.persistKey,
        callbacks: {
          onLoadStatus: (pct, detail) => {
            setLoadPct(pct)
            setLoadDetail(detail)
          },
          onAssistantText: (text, done) => {
            // Side effects (id allocation, ref writes) MUST stay OUT of the setState
            // updater: StrictMode double-invokes updaters, so an impure one silently
            // drops the append on its second pass - that was the missing-reply bug.
            let id = streamingId.current
            if (id == null) {
              id = nextId()
              streamingId.current = id
              const tid = id
              setTurns((prev) => [...prev, { id: tid, role: 'assistant', text }])
            } else {
              const tid = id
              setTurns((prev) => prev.map((t) => (t.id === tid ? { ...t, text } : t)))
            }
            if (done) {
              streamingId.current = null
              cbRef.current.onMessage?.('assistant', text)
            }
          },
          onGenerationStart: () => setStatus('thinking'),
          onGenerationEnd: () => {
            // A reply that ended WITHOUT done (aborted/superseded) must not leave its bubble id
            // armed, or the NEXT reply's stream overwrites the old bubble in place (a reply then
            // appears ABOVE the question it answers).
            streamingId.current = null
            setStatus('ready')
            // Performance-based expectation setting: the static pre-download heuristic can't tell
            // a flagship phone (~15-19 tok/s, fine) from a budget one (~2-5 tok/s, painful). But the
            // FIRST reply is a cold sample - GPU pipelines and a cold prefill warm up on it, and a
            // short reply has a high fixed-overhead ratio - so it reads slow on a device that is
            // actually fine (a common false "slower side" on capable machines). Judge steady state:
            // track the best rate seen, clear the flag the instant any reply clears reading speed, and
            // only raise it once the device has had a warm turn and still cannot keep up.
            const tps = engineRef.current?.lastGenStats?.tps
            if (tps !== undefined && tps > 0) {
              perfTurns.current++
              bestTps.current = Math.max(bestTps.current, tps)
              if (bestTps.current >= 6) setSlowDevice(false)
              else if (perfTurns.current >= 2) setSlowDevice(true)
            }
          },
          onError: (where, message) => {
            setError(`${where}: ${message}`)
            setStatus('error')
            const eid = nextId()
            setTurns((prev) => [...prev, { id: eid, role: 'error', text: `${where}: ${message}` }])
            cbRef.current.onError?.(where, message)
          },
        },
      })
      engineRef.current = engine

      // Project any persisted history into UI turns.
      const restored: WidgetTurn[] = []
      for (const m of engine.history) {
        if (m.role === 'system') continue
        restored.push({ id: nextId(), role: m.role === 'assistant' ? 'assistant' : 'user', text: m.content })
      }
      // eslint-disable-next-line react-hooks/set-state-in-effect -- one-shot restore of persisted history; the engine (its source) is created inside this same mount effect
      if (restored.length) setTurns(restored)
      // NOTE: we no longer auto-load the brain on mount. ChatPanel calls preload() once the
      // user enters a conversation (picks a mode / lands in the text view), so just opening to
      // the Type/Talk picker does not pull the ~237 MB model before a choice is made.
    }

    return () => {
      // Voice (if active) tears down immediately on a real unmount.
      orchRef.current?.dispose().catch(() => undefined)
      orchRef.current = null
      const engine = engineRef.current
      pendingDispose.current = setTimeout(() => {
        engine?.dispose()
        engineRef.current = null
        loadingRef.current = null
        streamingId.current = null
        pendingDispose.current = null
      }, 0)
    }
    // Mount-once: config/opts changes are handled by the effects below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Live-update the persona (configurator preview): custom prompt or title-derived name.
  useEffect(() => {
    engineRef.current?.setSystemPrompt(resolveSystemPrompt(config))
    // eslint-disable-next-line react-hooks/exhaustive-deps -- deliberately narrowed: only the two prompt-affecting fields should re-derive the persona
  }, [config.systemPrompt, config.title])

  useEffect(() => {
    engineRef.current?.setReasoning(config.reasoning ?? false)
  }, [config.reasoning])

  // Load the RAG retriever when a knowledge URL is set (lazy - pulls in the embedder +
  // index only then). Clearing the URL detaches RAG so the widget stays a plain chat.
  useEffect(() => {
    const url = config.knowledgeUrl
    if (!url) {
      engineRef.current?.setRetriever(null)
      return
    }
    let alive = true
    // Debounce: in the configurator the URL changes on every keystroke - only load once
    // the field settles, so we don't fire a fetch (and a console warning) per character.
    const timer = setTimeout(() => {
      void import('@/rag/retriever')
        .then(({ createRetriever }) => createRetriever(url))
        .then(({ retriever }) => {
          if (!alive) return
          engineRef.current?.setRetriever(retriever)
          // Prewarm the embedder's ORT session NOW, in the background (overlapping the LLM load),
          // so the FIRST query doesn't pay the cold load. Building the bge wasm session is several
          // seconds and otherwise lands on the user's first message - and the LLM's ttft metric
          // does not include it, which is why a "2s" reply can feel like 10s+ the first time.
          void import('@/rag/embedder').then(({ loadEmbedder }) => loadEmbedder()).catch(() => {})
        })
        .catch((e: unknown) => {
          if (alive) console.warn('[aidekin] knowledge load failed:', e)
        })
    }, 500)
    return () => {
      alive = false
      clearTimeout(timer)
    }
  }, [config.knowledgeUrl])

  const ensureLoaded = useCallback((engine: ConversationEngine): Promise<void> => {
    if (loadingRef.current) return loadingRef.current
    // Don't clobber an optimistic 'thinking' (send() sets it when the model is cached/ready and
    // only needs a quick init) with 'loading' - that would flip the dots back to a load state.
    setStatus((s) => (s === 'thinking' ? s : 'loading'))
    const p = engine
      .loadLlm()
      .then(() => {
        setStatus((s) => (s === 'loading' ? 'ready' : s))
        // Ask the browser to make this site's cached model eviction-resistant. Caches are
        // partitioned per top-level site, so this only helps repeat visits to THIS site -
        // it can't share across the different sites that embed the widget.
        void requestPersist().catch(() => false)
      })
      .catch((e: unknown) => {
        loadingRef.current = null
        setStatus('error')
        setError(e instanceof Error ? e.message : String(e))
        throw e
      })
    loadingRef.current = p
    return p
  }, [])

  // Start loading the brain at the moment the user commits to a conversation (mode picked /
  // text view shown), so the download overlaps with them typing instead of firing at the picker.
  const preload = useCallback(() => {
    const engine = engineRef.current
    if (engine) void ensureLoaded(engine).catch(() => undefined)
  }, [ensureLoaded])

  const send = useCallback(
    (raw: string) => {
      const text = raw.trim()
      const engine = engineRef.current
      if (!text || !engine || status === 'thinking') return
      const uid = nextId()
      setTurns((prev) => [...prev, { id: uid, role: 'user', text }])
      streamingId.current = null
      cbRef.current.onMessage?.('user', text)
      // Optimistic feedback: show the thinking dots the instant the message is sent, so there is
      // no dead air before onGenerationStart fires. Only when the model is already on disk
      // (cached) or loaded (ready) - i.e. just "preparing" - not a fresh ~237 MB download, where
      // the progress bar is the right feedback instead.
      if (cached || status === 'ready') setStatus('thinking')
      void ensureLoaded(engine)
        .then(() => engine.sendUserMessage(text))
        .catch(() => {
          /* onError already surfaced it */
        })
    },
    [ensureLoaded, status, cached],
  )

  const stop = useCallback(() => {
    engineRef.current?.abort()
    streamingId.current = null
    setStatus((s) => (s === 'thinking' ? 'ready' : s))
  }, [])

  const retry = useCallback(() => {
    const engine = engineRef.current
    if (!engine) return
    setError(null)
    loadingRef.current = null
    void ensureLoaded(engine)
  }, [ensureLoaded])

  const clear = useCallback(() => {
    engineRef.current?.clearHistory()
    streamingId.current = null
    setTurns([])
    setError(null)
    setStatus((s) => (s === 'cold' || s === 'loading' ? s : 'ready'))
  }, [])

  const forgetModel = useCallback(async (): Promise<ClearResult> => {
    // Voice shares the engine + uses OPFS-locked speech weights - tear it down (and
    // await it) BEFORE clearing caches, so the OPFS locks are released first.
    const orch = orchRef.current
    orchRef.current = null
    setVoiceActive(false)
    setVoiceState('cold')
    if (orch) await orch.dispose().catch(() => undefined)
    engineRef.current?.unloadLlm()
    loadingRef.current = null
    streamingId.current = null
    setStatus('cold')
    setLoadPct(0)
    const { clearModelCaches } = await import('@/core/storage')
    const result = await clearModelCaches()
    setCached(false)
    return result
  }, [])

  // Show the user's live speech transcript as a streaming user turn (pure updaters).
  const upsertUserTranscript = useCallback((text: string, final: boolean) => {
    // Final with nothing recognized → drop the placeholder turn instead of leaving a
    // dangling "…" bubble. (It's never added to the model context either - the
    // orchestrator only sends non-empty finals to the engine.)
    if (final && !text.trim()) {
      const id = userStreamingId.current
      userStreamingId.current = null
      if (id != null) setTurns((prev) => prev.filter((t) => t.id !== id))
      return
    }
    const shown = text || '…'
    let id = userStreamingId.current
    if (id == null) {
      id = nextId()
      userStreamingId.current = id
      const tid = id
      setTurns((prev) => [...prev, { id: tid, role: 'user', text: shown }])
    } else {
      const tid = id
      setTurns((prev) => prev.map((t) => (t.id === tid ? { ...t, text: shown } : t)))
    }
    if (final) userStreamingId.current = null
  }, [])

  const toggleVoice = useCallback(() => {
    const orch = orchRef.current
    if (voiceActive) {
      voiceGen.current++ // cancel any in-flight activation (checked after each await below)
      setVoiceActive(false)
      setMuted(false)
      if (orch && orch.isLoaded) {
        // Loaded → just turn the mic off; KEEP the speech models resident so re-entry is instant.
        void orch.stopListening().catch(() => undefined)
      } else {
        // Still downloading → ABANDON: dispose() terminates the workers (stopping the in-flight
        // ~1.6 GB download) and sweeps the partial weights, so nothing is left half-installed. A
        // later re-tap starts a fresh load (resuming from whatever fully downloaded + cached).
        orchRef.current = null
        setVoiceState('cold')
        void orch?.dispose().catch(() => undefined)
      }
      return
    }
    setVoiceActive(true)
    // Already loaded this session → resume listening instantly, no reload.
    if (orch) {
      const gen = ++voiceGen.current
      void orch
        .startListening()
        .then(() => {
          // Toggled off while the mic was opening: undo the listen that just landed.
          if (voiceGen.current !== gen) void orch.stopListening().catch(() => undefined)
        })
        .catch(() => undefined)
      return
    }
    // First activation: ensure the shared LLM is loaded, then lazy-load the orchestrator
    // (speech models) and start listening - one model, continuous context with text.
    const engine = engineRef.current
    if (!engine) {
      setVoiceActive(false)
      return
    }
    const gen = ++voiceGen.current
    setVoiceState('loading')
    setVoiceLoadPct(0)
    // Are the speech weights already on disk? Drives "Loading" vs "~1.6 GB download" copy.
    void import('@/core/modelStore')
      .then(({ hasModelAsset }) => hasModelAsset('asr/encoder.onnx.data'))
      .then(setVoiceCached)
      .catch(() => undefined)
    setError(null)
    void (async () => {
      let created: Orchestrator | null = null
      try {
        // Pre-check storage: voice adds ~1.6 GB. Fail fast with a clear, actionable message
        // instead of a QuotaExceededError mid-download (best practice for large downloads).
        // BUT only when there is a real, persistent storage grant to run out of. An ephemeral
        // session (Safari Private Browsing) reports a SMALL quota; there voice still runs - the
        // weights stream/buffer uncached and re-download next visit, which private mode implies -
        // so a small quota must NOT block it. Blocking only makes sense on a large-quota device
        // that is genuinely full. (A small quota that's nearly full = private/ephemeral → proceed.)
        const { estimateStorage } = await import('@/core/storage')
        const est = await estimateStorage().catch(() => null)
        if (voiceGen.current !== gen) return
        const PERSISTENT_QUOTA = 4_000_000_000 // below this the session is ephemeral (private) - never block
        if (est && est.quotaBytes >= PERSISTENT_QUOTA && est.quotaBytes - est.usageBytes < 1_700_000_000) {
          throw new Error('Not enough free storage for voice (about 1.6 GB needed). Try text instead, or free up space.')
        }
        await ensureLoaded(engine)
        // Toggled off mid-activation: before the orchestrator exists there is nothing
        // for the toggle-off path to dispose, so the cancel happens here.
        if (voiceGen.current !== gen) return
        const { Orchestrator } = await import('@/pipeline/orchestrator')
        if (voiceGen.current !== gen) return
        created = new Orchestrator({
          device: 'webgpu',
          engine,
          callbacks: {
            onState: (s) => setVoiceState(s),
            onUserTranscript: (t, f) => upsertUserTranscript(t, f),
            onLoadStatus: (components: ComponentLoad[]) => {
              const speech = components.filter((c) => c.label !== 'LLM')
              const avg = speech.length ? speech.reduce((s, c) => s + c.fraction, 0) / speech.length : 0
              setVoiceLoadPct(avg)
            },
            onLevel: (rms) => {
              levelRef.current = rms
            },
            onError: (where, message) => {
              const eid = nextId()
              setError(`${where}: ${message}`)
              setTurns((prev) => [...prev, { id: eid, role: 'error', text: `${where}: ${message}` }])
            },
          },
        })
        orchRef.current = created
        await created.load()
        if (voiceGen.current !== gen) return // toggle-off already disposed it via orchRef
        await created.startListening()
        // Toggled off while the mic was opening: undo the listen that just landed.
        if (voiceGen.current !== gen) void created.stopListening().catch(() => undefined)
      } catch (e) {
        // If the user abandoned voice mid-load (toggleVoice cleared/replaced orchRef and already
        // disposed), this rejection is the intentional cancel - clean up quietly, no error shown.
        if (voiceGen.current !== gen || orchRef.current !== created) {
          void created?.dispose().catch(() => undefined)
          return
        }
        orchRef.current = null
        setVoiceActive(false)
        setVoiceState('cold')
        setError(e instanceof Error ? e.message : String(e))
        void created?.dispose().catch(() => undefined)
      }
    })()
  }, [voiceActive, ensureLoaded, upsertUserTranscript])

  const toggleMute = useCallback(() => {
    setMuted((m) => {
      const next = !m
      orchRef.current?.setMuted(next)
      return next
    })
  }, [])

  return {
    turns,
    status,
    loadPct,
    loadDetail,
    cached,
    error,
    send,
    preload,
    stop,
    retry,
    clear,
    slowDevice,
    forgetModel,
    voiceState,
    voiceActive,
    voiceLoadPct,
    voiceCached,
    levelRef,
    toggleVoice,
    muted,
    toggleMute,
  }
}
