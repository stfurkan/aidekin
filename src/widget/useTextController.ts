// React hook wrapping the ConversationEngine for the TEXT widget. Owns the engine,
// projects its history into UI turns, streams the assistant reply, and gates model
// loading: the ~290 MB LLM downloads on first open (loadOnMount) or first send —
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
  /** Abort the in-flight generation (the "stop" button while thinking). */
  stop: () => void
  /** Re-attempt model load after an error. */
  retry: () => void
  clear: () => void
  /** True once the sliding window has dropped old turns (UI shows a subtle marker). */
  trimmed: boolean
  /** Unload the model from memory, clear its on-disk cache, and reset to 'cold' so the
   *  next message re-downloads. */
  forgetModel: () => Promise<ClearResult>
  // ── voice (lazy: the speech pipeline loads on the first toggle) ──────────────
  /** Orb state while voice is active ('listening' | 'thinking' | 'speaking' | …). */
  voiceState: AgentState
  voiceActive: boolean
  /** Speech-model (ASR/TTS/VAD/Turn) download progress, 0–1. */
  voiceLoadPct: number
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
  const [trimmed, setTrimmed] = useState(false)
  const [muted, setMuted] = useState(false)

  // Has the LLM been downloaded before? It caches to OPFS (see opfsModelCache.ts), so check
  // there — NOT Cache Storage, which only holds the optional Smart-Turn/embedder configs and
  // is absent for a plain text widget (which would make repeat visits always say "Downloading").
  useEffect(() => {
    void import('@/core/opfsModelCache')
      .then(({ hasLlmCache }) => hasLlmCache())
      .then(setCached)
      .catch(() => undefined)
  }, [])

  const engineRef = useRef<ConversationEngine | null>(null)
  const loadingRef = useRef<Promise<void> | null>(null)
  const streamingId = useRef<number | null>(null)
  const seq = useRef(0)
  const pendingDispose = useRef<ReturnType<typeof setTimeout> | null>(null)
  const orchRef = useRef<Orchestrator | null>(null)
  const userStreamingId = useRef<number | null>(null)
  const levelRef = useRef(0)
  // Latest callbacks, read by the (mount-once) engine so it never goes stale.
  const cbRef = useRef(opts)
  cbRef.current = opts

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
            // drops the append on its second pass — that was the missing-reply bug.
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
          onGenerationEnd: () => setStatus('ready'),
          onHistoryTrimmed: () => setTrimmed(true),
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
      if (restored.length) setTurns(restored)

      if (opts.loadOnMount !== false) void ensureLoaded(engine)
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
  }, [config.systemPrompt, config.title])

  useEffect(() => {
    engineRef.current?.setReasoning(config.reasoning ?? false)
  }, [config.reasoning])

  // Load the RAG retriever when a knowledge URL is set (lazy — pulls in the embedder +
  // index only then). Clearing the URL detaches RAG so the widget stays a plain chat.
  useEffect(() => {
    const url = config.knowledgeUrl
    if (!url) {
      engineRef.current?.setRetriever(null)
      return
    }
    let alive = true
    // Debounce: in the configurator the URL changes on every keystroke — only load once
    // the field settles, so we don't fire a fetch (and a console warning) per character.
    const timer = setTimeout(() => {
      void import('@/rag/retriever')
        .then(({ createRetriever }) => createRetriever(url))
        .then(({ retriever }) => {
          if (alive) engineRef.current?.setRetriever(retriever)
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
    setStatus('loading')
    const p = engine
      .loadLlm()
      .then(() => {
        setStatus((s) => (s === 'loading' ? 'ready' : s))
        // Ask the browser to make this site's cached model eviction-resistant. Caches are
        // partitioned per top-level site, so this only helps repeat visits to THIS site —
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

  const send = useCallback(
    (raw: string) => {
      const text = raw.trim()
      const engine = engineRef.current
      if (!text || !engine || status === 'thinking') return
      const uid = nextId()
      setTurns((prev) => [...prev, { id: uid, role: 'user', text }])
      streamingId.current = null
      cbRef.current.onMessage?.('user', text)
      void ensureLoaded(engine)
        .then(() => engine.sendUserMessage(text))
        .catch(() => {
          /* onError already surfaced it */
        })
    },
    [ensureLoaded, status],
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
    setTrimmed(false)
    setStatus((s) => (s === 'cold' || s === 'loading' ? s : 'ready'))
  }, [])

  const forgetModel = useCallback(async (): Promise<ClearResult> => {
    // Voice shares the engine + uses OPFS-locked speech weights — tear it down (and
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
    // dangling "…" bubble. (It's never added to the model context either — the
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
      void orch.startListening().catch(() => undefined)
      return
    }
    // First activation: ensure the shared LLM is loaded, then lazy-load the orchestrator
    // (speech models) and start listening — one model, continuous context with text.
    const engine = engineRef.current
    if (!engine) {
      setVoiceActive(false)
      return
    }
    setVoiceState('loading')
    setVoiceLoadPct(0)
    setError(null)
    void (async () => {
      let created: Orchestrator | null = null
      try {
        // Pre-check storage: voice adds ~1.6 GB. Fail fast with a clear, actionable message
        // instead of a QuotaExceededError mid-download (best practice for large downloads).
        const { estimateStorage } = await import('@/core/storage')
        const est = await estimateStorage().catch(() => null)
        if (est && est.quotaBytes > 0 && est.quotaBytes - est.usageBytes < 1_700_000_000) {
          throw new Error('Not enough free storage for voice (about 1.6 GB needed). Try text instead, or free up space.')
        }
        await ensureLoaded(engine)
        const { Orchestrator } = await import('@/pipeline/orchestrator')
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
        await created.startListening()
      } catch (e) {
        // If the user abandoned voice mid-load (toggleVoice cleared/replaced orchRef and already
        // disposed), this rejection is the intentional cancel — clean up quietly, no error shown.
        if (orchRef.current !== created) {
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
    stop,
    retry,
    clear,
    trimmed,
    forgetModel,
    voiceState,
    voiceActive,
    voiceLoadPct,
    levelRef,
    toggleVoice,
    muted,
    toggleMute,
  }
}
