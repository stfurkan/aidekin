// ConversationEngine - the shared brain for BOTH the text widget and the voice
// orchestrator. It owns the conversation state (messages[] + system prompt),
// optional RAG retrieval, history persistence + trimming, and the streaming LLM
// turn. It knows NOTHING about mic / VAD / ASR / TTS - voice wraps it with
// ASR-in (sendUserMessage) and TTS-out (the onAssistantClause callback).
//
// Two ownership modes for the LLM worker, so we never break the voice path:
//   • loadLlm()          → the engine CREATES + initializes its own worker
//                          (the text widget: no speech workers ever spin up).
//   • adoptLlmWorker(w)  → the engine REUSES the orchestrator's already-loaded
//                          worker; the orchestrator keeps routing the worker's
//                          messages and forwards token/done via handleLlmMessage().
// In adopted mode the orchestrator stays in charge of load/ready/error lifecycle,
// so its existing load UI + error handling are byte-identical.

import { LLM } from '../models/registry'
import type { ChatMessage, Device, LlmIn, LlmOut } from '../protocol/messages'
import { SentenceChunker } from '../pipeline/sentenceChunker'

/** One retrieved chunk of grounding context returned by the RAG retriever. */
export interface RetrievedChunk {
  readonly text: string
  readonly score?: number
  readonly source?: string
}

/** Pluggable retriever - null = no RAG, and then nothing RAG-related ever loads. */
export interface Retriever {
  retrieve(query: string, k: number): Promise<RetrievedChunk[]>
}

export interface EngineCallbacks {
  /** Cumulative assistant text as it streams (done=true on the final, recorded reply). */
  onAssistantText?: (text: string, done: boolean) => void
  /** A completed clause - voice routes this to TTS. Only emitted when chunkClauses=true. */
  onAssistantClause?: (clause: string) => void
  /** Fired when a generation starts (voice → setState('thinking')). */
  onGenerationStart?: () => void
  /** Fired when a generation ends or aborts (voice → settle to idle). */
  onGenerationEnd?: () => void
  /** Owned-worker load progress: pct in [0,1] + a human detail (e.g. "142 / 290 MB"). */
  onLoadStatus?: (pct: number, detail: string) => void
  onError?: (where: string, message: string) => void
  /** RAG telemetry: how many chunks were injected and how long retrieval took. */
  onRetrieval?: (info: { used: number; tookMs: number }) => void
  /** Fired when the sliding window dropped old turns so the UI can show a subtle
   *  "earlier messages trimmed" marker instead of forgetting silently. */
  onHistoryTrimmed?: (info: { dropped: number }) => void
}

export interface EngineOptions {
  systemPrompt: string
  /** LLM device for the owned-worker path (default 'webgpu'). */
  device?: Device
  /** null disables RAG entirely (no embedder, no query embedding). */
  retriever?: Retriever | null
  /** Top-k chunks to retrieve when RAG is on (kept small for a 1.7B model). */
  ragTopK?: number
  /** Hard char budget for the injected context block. */
  ragCharBudget?: number
  /** Minimum cosine score for a chunk to be injected. Below this, the chunk is dropped
   *  - so an off-topic message doesn't pull (and recite) unrelated content. */
  ragMinScore?: number
  /** Voice splits the stream into clauses for low-latency TTS; text does not. */
  chunkClauses?: boolean
  /** Let the model think on EVERY turn (slower, more accurate). RAG turns always think
   *  regardless; this controls plain (non-RAG) turns. Default false (fast). */
  reasoning?: boolean
  /** localStorage key to persist history across reloads (per host origin). Unset = no persistence. */
  persistKey?: string
  /** Approximate token budget for retained history (system prompt is never dropped). */
  maxHistoryTokens?: number
  callbacks?: EngineCallbacks
}

const DEFAULT_MAX_HISTORY_TOKENS = 6000
// Rough token estimate (about 4 chars/token), good enough for a sliding-window trim.
const approxTokens = (s: string): number => Math.ceil(s.length / 4)

// Static answering rules for grounded (RAG) turns. Kept in the SYSTEM PROMPT (the cached
// prefix), NOT re-injected into every user turn, so they are prefilled once and only the
// per-turn <info> block + question land in the KV-cache delta. Phrased to be a no-op on turns
// that carry no <info> block (greetings, small talk). Also forbids HTML/links so the small
// model stops emitting mangled hrefs from grounded text.
const RAG_INSTRUCTION =
  'When the user message includes an <info> block, answer using ONLY the information inside it, ' +
  'in your own words as if you already knew it, in 1-2 sentences. Do not mention the block or use ' +
  'phrases like "the reference", "the text says", "based on the context", or "according to", and do ' +
  'not add unrelated details. If the answer is not in the block, say you do not have that information. ' +
  'Never output HTML or markdown; refer to pages by name and write a URL or email only if it appears ' +
  'verbatim in the <info> block.'

/** One conversation turn. `content` is the plain text (what the user actually said /
 *  the assistant replied) - used for display, persistence and hydration. `model` is the
 *  OPTIONAL augmented form sent to the LLM: a grounded (RAG) user turn carries its
 *  context-injected prompt here. Keeping the augmented form on the turn - instead of
 *  rebuilding it per request and discarding it - is what makes the model-prompt PREFIX
 *  stable across turns, so the worker's KV cache reuses it (prefill only the new turn)
 *  rather than re-prefilling the whole transcript every grounded turn. */
interface Turn {
  role: ChatMessage['role']
  content: string
  model?: string
}
/** The exact form sent to the model (augmented where present), used for both the
 *  generate request and the KV-cache prefix. */
const toModel = (m: Turn): ChatMessage => ({ role: m.role, content: m.model ?? m.content })

export class ConversationEngine {
  private readonly cb: EngineCallbacks
  private readonly device: Device
  private retriever: Retriever | null
  private readonly ragTopK: number
  private readonly ragCharBudget: number
  private readonly ragMinScore: number
  private chunkClauses: boolean
  private alwaysThink: boolean
  private clauseSink: ((clause: string) => void) | null = null
  private readonly persistKey?: string
  private readonly maxHistoryTokens: number

  private systemPrompt: string
  private messages: Turn[]
  private readonly chunker = new SentenceChunker()

  /** The conversation as the model sees it (augmented grounded turns), for the request
   *  and the cache prefix. A fresh array each call - callers consume it immediately. */
  private modelView(): ChatMessage[] {
    return this.messages.map(toModel)
  }

  /** The system prompt as the model sees it: the owner's persona plus, ONLY while RAG is
   *  active, the static grounding rules. Keeping the rules in the cached prefix (not in every
   *  user turn) means they are prefilled once, so each grounded turn's delta is just the
   *  retrieved context + the question. */
  private composedSystem(): string {
    return this.retriever ? `${this.systemPrompt}\n\n${RAG_INSTRUCTION}` : this.systemPrompt
  }

  private llm: Worker | null = null
  private ownsWorker = false
  // Track the largest file being downloaded (the model weights dominate) → one smooth
  // 0→100 bar, without per-file resets or the bogus totals from summing reused keys.
  private dlTotal = 0
  private dlLoaded = 0

  private genId = 0
  private currentId = -1
  // When the history prefix changes non-append (reset / clear / system-prompt change /
  // sliding-window trim), the worker's KV cache is stale - flag the next generate to
  // rebuild it. Set here, sent once, then cleared.
  private cacheDirty = false
  private assistant = ''
  private pending: { id: number; resolve: (text: string) => void } | null = null
  private ready: { resolve: () => void; reject: (e: Error) => void } | null = null

  constructor(opts: EngineOptions) {
    this.cb = opts.callbacks ?? {}
    this.device = opts.device ?? 'webgpu'
    this.retriever = opts.retriever ?? null
    this.ragTopK = opts.ragTopK ?? 3
    this.ragCharBudget = opts.ragCharBudget ?? 1500
    // Gate so greetings / off-topic messages do NOT pull doc chunks and get recited back.
    // Calibrated from real queries on our corpus (a greeting scored ~0.51, a real question
    // ~0.65), so 0.55 sits in the gap. Absolute cosine cutoffs are not portable across models
    // or corpora, so treat this as a tunable DEFAULT, not a universal truth; the retrieval log
    // prints each query's score so a deployment can recalibrate.
    this.ragMinScore = opts.ragMinScore ?? 0.55
    this.chunkClauses = opts.chunkClauses ?? false
    this.alwaysThink = opts.reasoning ?? false
    this.persistKey = opts.persistKey
    this.maxHistoryTokens = opts.maxHistoryTokens ?? DEFAULT_MAX_HISTORY_TOKENS
    this.systemPrompt = opts.systemPrompt
    this.messages = this.hydrate()
  }

  // ── worker ownership ────────────────────────────────────────────────────────

  /** Text path: create + initialize a dedicated LLM worker. */
  async loadLlm(): Promise<void> {
    if (this.llm) return
    this.dlTotal = 0
    this.dlLoaded = 0
    // Literal `new Worker(new URL(...), {type:'module'})` - the exact form Vite bundles.
    const w = new Worker(new URL('../workers/llm.worker.ts', import.meta.url), { type: 'module' })
    this.llm = w
    this.ownsWorker = true
    w.onmessage = (e: MessageEvent<LlmOut>) => this.onOwnedMessage(e.data)
    w.onerror = (e: ErrorEvent) => {
      this.cb.onError?.('LLM', e.message || 'worker crashed at load')
      this.ready?.reject(new Error(e.message || 'LLM worker crashed'))
      this.ready = null
    }
    const init: LlmIn = {
      kind: 'init',
      model: LLM.hfModelId,
      dtype: LLM.dtype,
      device: this.device,
      eosTokenId: LLM.eosTokenId,
    }
    try {
      await new Promise<void>((resolve, reject) => {
        this.ready = { resolve, reject }
        w.postMessage(init)
      })
    } catch (err) {
      // Init failed (download/OOM/quota) - tear down so a retry recreates a fresh worker.
      this.ready = null
      this.ownsWorker = false
      this.llm = null
      try {
        w.terminate()
      } catch {
        /* already gone */
      }
      throw err
    }
  }

  /** Voice path: reuse the orchestrator's already-initialized worker. */
  adoptLlmWorker(w: Worker): void {
    this.llm = w
    this.ownsWorker = false
  }

  /** Voice path: the orchestrator forwards the worker's token/done events here. */
  handleLlmMessage(m: LlmOut): void {
    this.processGeneration(m)
  }

  private onOwnedMessage(m: LlmOut): void {
    if (m.kind === 'load') {
      // Follow the largest file (the weights). Tiny tokenizer/config files finish in a
      // blink and must not reset the bar; we never sum reused keys (that inflated totals).
      if (m.total > this.dlTotal) {
        this.dlTotal = m.total
        this.dlLoaded = m.loaded
      } else if (m.total === this.dlTotal) {
        this.dlLoaded = Math.max(this.dlLoaded, m.loaded)
      }
      const pct = this.dlTotal > 0 ? Math.min(1, this.dlLoaded / this.dlTotal) : 0
      this.cb.onLoadStatus?.(pct, this.dlTotal > 0 ? '' : m.detail || 'Preparing…')
      return
    }
    if (m.kind === 'ready') {
      this.ready?.resolve()
      this.ready = null
      return
    }
    if (m.kind === 'error') {
      if (this.ready) {
        // Load-phase error → reject the load promise (existing behaviour).
        this.cb.onError?.('LLM', m.message)
        this.ready.reject(new Error(m.message))
        this.ready = null
      } else {
        // Generation-phase error → settle the in-flight turn so it doesn't hang on "thinking".
        this.settleGenerationError(m.message)
      }
      return
    }
    this.processGeneration(m)
  }

  // ── conversation turn ───────────────────────────────────────────────────────

  /** Run one user turn: record it, (optionally) retrieve context, stream a reply. */
  sendUserMessage(text: string): Promise<string> {
    const clean = text.trim()
    if (!clean) return Promise.resolve('')
    this.pushUser(clean)
    // No retriever → build the request synchronously so voice timing is unchanged.
    if (!this.retriever) return this.generate(this.modelView(), this.alwaysThink)
    return this.withRag(clean)
  }

  private async withRag(userText: string): Promise<string> {
    try {
      const t0 = performance.now()
      const hits = await this.retriever!.retrieve(userText, this.ragTopK)
      // Ground only on chunks that clear the relevance gate. The query is embedded with the
      // bge retrieval instruction (see embedder.embedQuery), which sharpens the on-topic vs
      // off-topic score gap, so an off-topic message (a greeting, small-talk) scores below the
      // gate and is dropped here rather than pulled in and recited.
      const relevant = hits.filter((h) => (h.score ?? 0) >= this.ragMinScore)
      // Log the scores so the gate can be calibrated against real greetings vs questions.
      console.info(
        `[aidekin] RAG "${userText.slice(0, 40)}" top=${(hits[0]?.score ?? 0).toFixed(3)} ` +
          `used=${relevant.length}/${hits.length} (gate ${this.ragMinScore})`,
      )
      this.cb.onRetrieval?.({ used: relevant.length, tookMs: performance.now() - t0 })
      if (relevant.length) this.applyContext(userText, relevant)
    } catch (err) {
      this.cb.onError?.('RAG', (err as Error).message)
    }
    // RAG turns run NON-thinking: a <think> block is stripped from the stored reply, so it
    // wouldn't round-trip and the worker would have to drop the KV cache every grounded turn.
    // Non-thinking keeps the cache reusable (the whole point of the augmented model-view).
    // Reasoning, if the owner explicitly enabled it, still applies (and rebuilds the cache).
    return this.generate(this.modelView(), this.alwaysThink)
  }

  /** Fold retrieved context into the CURRENT user turn's `model` field. Its plain `content`
   *  is untouched (display/persist), but the LLM - and the KV-cache prefix - see the
   *  augmented prompt, which is frozen on the turn so the prefix stays stable next turn. */
  private applyContext(userText: string, hits: RetrievedChunk[]): void {
    let used = ''
    for (const h of hits) {
      const next = used ? `${used}\n\n---\n\n${h.text}` : h.text
      if (next.length > this.ragCharBudget) break
      used = next
    }
    // The static answering rules now live in the system prompt (RAG_INSTRUCTION), cached once.
    // Only the per-turn context + question go here, so the KV-cache delta stays small.
    const augmented = `<info>\n${used}\n</info>\n\nQuestion: ${userText}`
    const last = this.messages[this.messages.length - 1]
    if (last && last.role === 'user') last.model = augmented
  }

  private generate(request: ChatMessage[], think: boolean): Promise<string> {
    if (!this.llm) {
      const e = new Error('LLM worker not ready')
      this.cb.onError?.('LLM', e.message)
      return Promise.reject(e)
    }
    // Latest-response-wins: if a generation is still in flight (a newer user turn arrived
    // before the previous reply finished), abort it on the worker and settle its promise so
    // turns don't queue and balloon ttft. No transcript is lost - only the superseded reply.
    if (this.currentId >= 0) {
      const abortMsg: LlmIn = { kind: 'abort', id: this.currentId }
      this.llm.postMessage(abortMsg)
      this.pending?.resolve(this.assistant)
      this.pending = null
    }
    this.genId++
    const id = this.genId
    this.currentId = id
    this.chunker.reset()
    this.assistant = ''
    this.cb.onGenerationStart?.()
    const msg: LlmIn = { kind: 'generate', id, messages: request, think, resetCache: this.cacheDirty }
    this.cacheDirty = false
    this.llm.postMessage(msg)
    return new Promise<string>((resolve) => {
      this.pending = { id, resolve }
    })
  }

  private processGeneration(m: LlmOut): void {
    if (m.kind === 'token') {
      if (m.id !== this.currentId) return
      this.assistant += m.text
      this.cb.onAssistantText?.(this.assistant, false)
      if (this.chunkClauses) {
        const sink = this.clauseSink ?? this.cb.onAssistantClause
        for (const clause of this.chunker.push(m.text)) sink?.(clause)
      }
    } else if (m.kind === 'done') {
      if (m.id !== this.currentId) return // stale generation (superseded/aborted)
      this.finish(m.text)
    } else if (m.kind === 'error') {
      // Generation error reaching the adopted (voice) path - settle so it doesn't hang.
      this.settleGenerationError(m.message)
    }
  }

  /** Settle an in-flight generation that errored: report it, drop the turn, and resolve the
   *  pending promise + fire onGenerationEnd so the UI leaves "thinking" instead of hanging. */
  private settleGenerationError(message: string): void {
    this.cb.onError?.('LLM', message)
    if (this.currentId < 0) return
    this.currentId = -1
    this.chunker.reset()
    this.cb.onGenerationEnd?.()
    this.pending?.resolve(this.assistant)
    this.pending = null
  }

  private finish(doneText?: string): void {
    // Prefer the streamed text; fall back to the final 'done' payload if streaming
    // produced nothing - the reply still shows even if token messages were missed.
    const text = this.assistant.trim() ? this.assistant : (doneText ?? '').trim()
    if (this.chunkClauses) {
      const sink = this.clauseSink ?? this.cb.onAssistantClause
      const rest = this.chunker.flush()
      if (rest) sink?.(rest)
    }
    // Skip an empty reply (e.g. all tokens spent in a <think> block) - don't record it.
    if (text) {
      this.assistant = text
      this.pushAssistant(text)
      this.cb.onAssistantText?.(text, true)
    }
    this.currentId = -1
    this.cb.onGenerationEnd?.()
    this.pending?.resolve(this.assistant)
    this.pending = null
  }

  /** Barge-in / stop: abort the in-flight generation. Does NOT fire onGenerationEnd -
   *  the caller (voice barge-in) drives its own state, so we avoid a spurious idle. */
  abort(): void {
    if (this.currentId >= 0 && this.llm) {
      const msg: LlmIn = { kind: 'abort', id: this.currentId }
      this.llm.postMessage(msg)
    }
    this.currentId = -1
    this.chunker.reset()
    this.pending?.resolve(this.assistant)
    this.pending = null
  }

  // ── state ───────────────────────────────────────────────────────────────────

  get isGenerating(): boolean {
    return this.currentId >= 0
  }

  get history(): readonly ChatMessage[] {
    // Expose the PLAIN conversation (no injected RAG context) - for display/hydration.
    return this.messages.map((m) => ({ role: m.role, content: m.content }))
  }

  /** New session: clear back to just the system prompt (keeps it in storage). */
  reset(): void {
    this.abort()
    this.messages = [{ role: 'system', content: this.composedSystem() }]
    this.cacheDirty = true
    this.persist()
  }

  /** Wipe persisted history too (the widget's "clear chat" control). */
  clearHistory(): void {
    this.reset()
    if (this.persistKey) {
      try {
        localStorage.removeItem(this.persistKey)
      } catch {
        /* storage may be unavailable (private mode / partitioning) */
      }
    }
  }

  /** Attach (or clear) RAG after construction - the index loads asynchronously. */
  setRetriever(retriever: Retriever | null): void {
    const had = !!this.retriever
    this.retriever = retriever
    // The grounding rules live in the system prompt only while RAG is active; toggling RAG
    // changes the cached prefix, so refresh the system message and invalidate the cache once.
    if (had !== !!retriever && this.messages[0]?.role === 'system') {
      this.messages[0] = { role: 'system', content: this.composedSystem() }
      this.cacheDirty = true
    }
  }

  /** Toggle internal reasoning on plain (non-RAG) turns. */
  setReasoning(on: boolean): void {
    this.alwaysThink = on
  }

  /** Voice attaches here at runtime: split the stream into clauses and route them to a
   *  TTS sink. Setting chunkClauses off (text mode) restores plain streaming. */
  setChunkClauses(on: boolean): void {
    this.chunkClauses = on
  }

  setClauseSink(sink: ((clause: string) => void) | null): void {
    this.clauseSink = sink
  }

  setSystemPrompt(prompt: string): void {
    this.systemPrompt = prompt
    if (this.messages.length && this.messages[0].role === 'system') {
      this.messages[0] = { role: 'system', content: this.composedSystem() }
    } else {
      this.messages.unshift({ role: 'system', content: this.composedSystem() })
    }
    this.cacheDirty = true // the cached prefix starts with the old system prompt
    this.persist()
  }

  dispose(): void {
    this.abort()
    if (this.ownsWorker) this.llm?.terminate()
    this.llm = null
  }

  /** Free the LLM worker + VRAM but keep the engine reusable - a later loadLlm()
   *  re-creates the worker (and re-downloads, if the on-disk cache was cleared). */
  unloadLlm(): void {
    this.abort()
    if (this.ownsWorker) this.llm?.terminate()
    this.llm = null
    this.ownsWorker = false
    this.ready = null
    this.dlTotal = 0
    this.dlLoaded = 0
  }

  // ── history helpers ─────────────────────────────────────────────────────────

  private pushUser(text: string): void {
    this.messages.push({ role: 'user', content: text })
    this.trim()
    this.persist()
  }

  private pushAssistant(text: string): void {
    this.messages.push({ role: 'assistant', content: text })
    this.trim()
    this.persist()
  }

  /** Sliding window. When over budget, keep three anchors - the system prompt (0)
   *  and the FIRST user+assistant exchange (1,2) - and drop the OLDEST middle turns
   *  in user/assistant pairs, always preserving the most recent exchanges. Pinning
   *  the opening keeps the "attention sink" + the conversation's framing; evicting
   *  the middle (not the head) is what good local-LLM chats do. */
  private trim(): void {
    const budget = this.maxHistoryTokens
    // Count the MODEL form (augmented grounded turns are larger) - that's what actually
    // fills the context window the worker prefills.
    const tok = (m: Turn): number => approxTokens(m.model ?? m.content)
    let total = this.messages.reduce((n, m) => n + tok(m), 0)
    if (total <= budget) return
    const head = Math.min(3, this.messages.length) // system + first exchange
    const keepTail = 4 // always keep the last couple of exchanges verbatim
    let dropped = 0
    while (total > budget && this.messages.length - dropped > head + keepTail + 1) {
      // Drop the oldest middle pair (user+assistant) to keep history coherent.
      const a = this.messages[head]
      const b = this.messages[head + 1]
      if (!a) break
      total -= tok(a) + (b ? tok(b) : 0)
      this.messages.splice(head, b ? 2 : 1)
      dropped += b ? 2 : 1
    }
    if (dropped > 0) {
      this.cacheDirty = true // the prefix changed → worker must re-prefill
      this.cb.onHistoryTrimmed?.({ dropped })
    }
  }

  private hydrate(): Turn[] {
    const fresh: Turn[] = [{ role: 'system', content: this.composedSystem() }]
    if (!this.persistKey) return fresh
    try {
      const raw = localStorage.getItem(this.persistKey)
      if (!raw) return fresh
      const parsed = JSON.parse(raw) as ChatMessage[]
      if (!Array.isArray(parsed) || !parsed.length) return fresh
      // Restored turns are plain (no `model`); the cache simply rebuilds once on the first
      // grounded turn after a reload, then reuses from there. Always re-assert the current
      // system prompt as the head (config may have changed).
      const rest = parsed.filter((m) => m.role !== 'system').map((m) => ({ role: m.role, content: m.content }))
      return [{ role: 'system', content: this.composedSystem() }, ...rest]
    } catch {
      return fresh
    }
  }

  private persist(): void {
    if (!this.persistKey) return
    try {
      // Persist the PLAIN conversation only - never the injected RAG context (it's large,
      // re-derived each turn, and would resurface as visitor-visible text on reload).
      const plain: ChatMessage[] = this.messages.map((m) => ({ role: m.role, content: m.content }))
      localStorage.setItem(this.persistKey, JSON.stringify(plain))
    } catch {
      /* quota / unavailable - history is best-effort, never fatal */
    }
  }
}
