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

import { LLM, llmMaxSeqLen, llmModelUrls } from '../models/registry'
import type { ChatMessage, Device, LlmConfidence, LlmIn, LlmOut } from '../protocol/messages'
import { SentenceChunker, speakable } from '../pipeline/sentenceChunker'
import { debugEnabled, dlog } from '../core/log'

/** One retrieved chunk of grounding context returned by the RAG retriever. */
export interface RetrievedChunk {
  readonly text: string
  readonly score?: number
  readonly source?: string
  /** Hybrid-retrieval signal: the chunk contains every distinctive term of the query. The gate
   *  accepts this even below the cosine threshold, so exact-term queries ground reliably. */
  readonly lexMatch?: boolean
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
  /** Owned-worker load progress: pct in [0,1] + a human detail (e.g. "142 / 237 MB"). */
  onLoadStatus?: (pct: number, detail: string) => void
  onError?: (where: string, message: string) => void
  /** RAG telemetry: how many chunks were injected and how long retrieval took. */
  onRetrieval?: (info: { used: number; tookMs: number }) => void
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
  /** Ask the worker for per-token confidence (bitgpu logprobs) on every turn, exposed via
   *  {@link lastGenStats}. Costs a few % (disables speculation for the turn), so default false;
   *  turn on when a caller wants the "not sure about this" signal. */
  measureConfidence?: boolean
  /** localStorage key to persist history across reloads (per host origin). Unset = no persistence.
   *  Also namespaces the worker's cross-reload KV-cache snapshot (bitgpu chat.save/restore), so a
   *  returning visitor's first turn is a cache-append with no re-prefill of the whole history. */
  persistKey?: string
  /** Fixed sampler seed for deterministic replies. Eval-only; leave unset in production. */
  samplerSeed?: number
  /** Prompt-lookup speculative decoding (bitgpu experimental; identical output, speed varies
   *  by workload). Default off; measure with the eval before enabling anywhere. */
  promptLookup?: boolean
  /** The assistant's brand name (the widget title). When set to a single word of 6+ letters,
   *  near-miss spellings in replies are corrected to it deterministically: a small model spells
   *  a coined name unreliably (measured at every sampling setting), and a mangled brand name is
   *  the single most visible quality defect. Words the user or the retrieved context actually
   *  used are never touched. */
  brandName?: string
  callbacks?: EngineCallbacks
}

// Static answering rules for grounded (RAG) turns. Kept in the SYSTEM PROMPT (the cached
// prefix), NOT re-injected into every user turn, so they are prefilled once and only the
// per-turn <info> block + question land in the KV-cache delta. Phrased to be a no-op on turns
// that carry no <info> block (greetings, small talk). Also forbids HTML/links so the small
// model stops emitting mangled hrefs from grounded text.
const RAG_INSTRUCTION =
  'When the user message includes an <info> block, answer using ONLY the facts inside it, staying close ' +
  'to its wording, in 1-2 sentences. Do NOT add, guess, or infer anything not stated there - never ' +
  'invent URLs, emails, commands, file names, numbers, or features. If the block does not answer the ' +
  'question, say you do not have that information instead of filling in. Refer to pages and tools by ' +
  'NAME; do not write URLs, code, HTML, or markdown. Do not mention the block or use phrases like "the ' +
  'reference", "the context", "the text", "based on", or "according to".'

// Always-on grounding discipline (added whenever a retriever is configured). The rule the model
// applies to EVERY turn, in any language: the harm we prevent is a wrong claim about the OWNER'S
// site/business, not the use of world knowledge. So it may greet, discuss itself, recall what the
// user said, and answer clearly general questions - but it must never assert a site/business fact
// it was not given. No keywords, no hard-coded phrases: the multilingual model does the judging.
const SITE_GROUNDING =
  'Facts about this specific site, business, or product - whether it HAS or offers a feature, ' +
  'service, plan, app, integration, or payment option, and its prices, hours, or policies - may ' +
  'ONLY come from an <info> block or from what the user told you. If you are asked whether it offers ' +
  'or supports something and were not given the answer, do NOT answer yes or no from assumption and ' +
  'do NOT invent details: say in one sentence that you do not have that information. This overrides ' +
  'your general knowledge for anything specific to this site or business. Greetings, small talk, ' +
  'questions about yourself, recalling what the user told you, and clearly general questions ' +
  'unrelated to this site are answered normally.'

// Conversational mechanics, appended to EVERY system prompt (custom or default). A small model
// primed by info-answering instructions treats any input as a query to answer, so a bare
// "hello again" gets a dictionary-style definition of the phrase instead of a greeting back.
// Kept separate from the persona (the owner's prompt) and from RAG grounding rules.
const CHAT_INSTRUCTION =
  'When the user greets you or makes small talk, reply with one short, friendly sentence and offer ' +
  'to help; never explain, define, or analyze what the user said. If an earlier question in the ' +
  'conversation was never answered, address it in your reply. You are open-source software running ' +
  'entirely in this browser: you have no location, workplace, team, or creator organization, and ' +
  'you must never invent one.'

// Hybrid retrieval: a strong lexical match (all distinctive query terms present) can rescue a chunk
// that scored just under the cosine gate - but only if it is still semantically PLAUSIBLE, i.e. within
// this margin of the gate. Without the floor an incidental term hit (the greeting "hello" matching a
// "hello@..." email) would ground an off-topic turn; with it, only near-miss chunks are rescued.
const RAG_LEX_MARGIN = 0.1

/** Preserve the original word's leading capitalization when substituting the brand spelling. */
const matchCase = (firstChar: string, brand: string): string =>
  firstChar === firstChar.toUpperCase() ? brand[0].toUpperCase() + brand.slice(1) : brand

/** True when `b` is within one edit (sub/ins/del) of `a`, case-insensitive. */
function withinOneEdit(a: string, b: string): boolean {
  if (a === b) return true
  const la = a.length
  const lb = b.length
  if (Math.abs(la - lb) > 1) return false
  let i = 0
  while (i < la && i < lb && a[i] === b[i]) i++
  if (la === lb) return a.slice(i + 1) === b.slice(i + 1) // one substitution
  const [shorter, longer] = la < lb ? [a, b] : [b, a]
  return shorter.slice(i) === longer.slice(i + 1) // one insertion/deletion
}

// The product promises plain text. A model that emits markup under pressure ("write me the
// exact html embed code") gets it stripped deterministically - enforcement in code, not one
// more prompt rule. Partial tags at a streaming edge ("<scr") are stripped too and the full
// tag is caught on the next update. Markdown code fences get the same treatment: with the
// HTML inside them stripped they render as bare ``` noise (the widget's renderer is
// intentionally fence-free), so drop the fence markers themselves.
const stripHtmlTags = (s: string): string => s.replace(/<\/?[a-z][^<>]*>?/gi, '').replace(/^[ \t]*```[a-z]*[ \t]*$\n?|```/gim, '')

// A URL or email in an answer. Used by the fabrication guard below.
const URL_OR_EMAIL = /(?:https?:\/\/|www\.)[^\s)<>"']+|[^\s@<>"']+@[^\s@<>"']+\.[^\s)<>"']+/gi

/** Deterministic anti-fabrication guard for GROUNDED answers. A small model sometimes emits a URL or
 *  email that is not in the retrieved context - either invented, or a real one it mangled
 *  ("https:\/\/aide kin. com"). Remove any URL/email from the answer that does NOT appear verbatim in
 *  `context`. While the answer is still streaming, also hold back a trailing in-progress URL/email so a
 *  bad one never flashes into view; once it is complete (or at finish) it is kept only if it matches
 *  the context. Pure string ops, no model, runs once per streamed update - so it stays generic and
 *  cheap for every embed, not just ours. */
function guardFabrications(text: string, context: string, streaming: boolean): string {
  let out = text.replace(/\\(?=[/_*`~])/g, '') // drop stray JSON-escape backslashes ("https:\/\/" -> "https://")
  if (streaming) {
    const tail = /\S*(?:https?:\/\/|www\.|@)\S*$/i.exec(out) // a link/email still being typed at the end
    if (tail && tail.index >= 0) out = out.slice(0, tail.index)
  }
  out = out.replace(URL_OR_EMAIL, (m) => {
    if (context.includes(m)) return m
    // Sentence punctuation glued to the match ("see aidekin.com.") is not part of the URL/email;
    // don't let it turn a legitimate quote into a false positive.
    const bare = m.replace(/[.,!?;:]+$/, '')
    return bare && context.includes(bare) ? m : ''
  })
  return out.replace(/[^\S\n]{2,}/g, ' ').replace(/[^\S\n]+([.,!?;:])/g, '$1')
}

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
  private retriever: Retriever | null
  private readonly ragTopK: number
  private readonly ragCharBudget: number
  private readonly ragMinScore: number
  private chunkClauses: boolean
  private alwaysThink: boolean
  private clauseSink: ((clause: string) => void) | null = null
  private supersededSink: (() => void) | null = null
  private readonly persistKey?: string
  private readonly samplerSeed?: number
  private promptLookup: boolean
  private readonly measureConfidence: boolean
  private readonly brandName: string | null
  /** tok/s + speculation + confidence + cache-reuse of the last completed generation (measurement/eval/UI). */
  lastGenStats: { tps?: number; reusedCache?: boolean; speculation?: { steps: number; drafted: number; accepted: number }; confidence?: LlmConfidence } | null = null

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
    const parts = [this.systemPrompt, CHAT_INSTRUCTION]
    if (this.retriever) parts.push(RAG_INSTRUCTION, SITE_GROUNDING)
    return parts.join('\n\n')
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
  // The retrieved chunk text backing the CURRENT turn (empty when the turn wasn't grounded). The
  // fabrication guard checks the answer's URLs/emails against this, so nothing not in the context
  // reaches the screen. Set in applyContext, reset at the start of each user message.
  private turnContext = ''
  private pending: { id: number; resolve: (text: string) => void } | null = null
  private ready: { resolve: () => void; reject: (e: Error) => void } | null = null
  private prewarmTimer: ReturnType<typeof setTimeout> | null = null

  constructor(opts: EngineOptions) {
    this.cb = opts.callbacks ?? {}
    this.retriever = opts.retriever ?? null
    this.ragTopK = opts.ragTopK ?? 3
    this.ragCharBudget = opts.ragCharBudget ?? 1500
    // Gate so greetings / off-topic messages do NOT pull doc chunks and get recited back.
    // Calibrated from real queries on our corpus (a greeting scored ~0.51, a real question
    // ~0.65), so 0.55 sits in the gap. Absolute cosine cutoffs are not portable across models
    // or corpora, so treat this as a tunable DEFAULT, not a universal truth; the retrieval log
    // prints each query's score so a deployment can recalibrate.
    // Calibrated against the section-scoped chunker: honest matches score ~0.68-0.78, while
    // greetings/small talk peak ~0.51 on their best (irrelevant) chunk - 0.55 separates them.
    this.ragMinScore = opts.ragMinScore ?? 0.55
    this.chunkClauses = opts.chunkClauses ?? false
    this.alwaysThink = opts.reasoning ?? false
    this.persistKey = opts.persistKey
    this.samplerSeed = opts.samplerSeed
    this.promptLookup = opts.promptLookup ?? false
    this.measureConfidence = opts.measureConfidence ?? false
    const brand = opts.brandName?.trim() ?? ''
    this.brandName = /^[a-z]{6,}$/i.test(brand) ? brand : null
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
    const u = llmModelUrls()
    const init: LlmIn = {
      kind: 'init',
      manifestUrl: u.manifestUrl,
      dataUrl: u.dataUrl,
      auxUrl: u.auxUrl,
      tokenizerModelId: LLM.tokenizerModelId,
      eosTokenId: LLM.eosTokenId,
      maxSeqLen: llmMaxSeqLen(),
      kvCache: LLM.kvCache,
      overflow: LLM.overflow, // 'sinks': the conversation grows unbounded; the engine rolls the window
      // Persist the KV cache across reloads only when text persistence is on. Mirroring persistKey
      // keeps both gated together; the worker namespaces the snapshot with this + model + mode.
      persistSession: this.persistKey,
      debug: debugEnabled(),
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
    this.schedulePrewarm() // warm the system prompt into the cache while the user reads the greeting
  }

  /** Voice path: reuse the orchestrator's already-initialized worker. */
  adoptLlmWorker(w: Worker): void {
    this.llm = w
    this.ownsWorker = false
    this.schedulePrewarm()
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
      } else if (m.id === undefined || m.id === this.currentId) {
        // Generation-phase error → settle the in-flight turn so it doesn't hang on "thinking".
        // A SUPERSEDED turn's error (its unwind after an abort) must not settle its replacement.
        this.settleGenerationError(m.message)
      }
      return
    }
    this.processGeneration(m)
  }

  // ── conversation turn ───────────────────────────────────────────────────────

  /** Correct near-miss spellings of the brand name ("Aidkin", "aideskin", "aide kin") to the
   *  configured spelling. Deterministic and tightly scoped: single alphabetic words sharing the
   *  brand's first three letters, within ONE edit, and not literally present in the user's
   *  message or the retrieved context (a word they actually used is never rewritten). */
  private fixBrandSpelling(text: string): string {
    const brand = this.brandName
    if (!brand) return text
    const lower = brand.toLowerCase()
    const protectedText = (this.turnContext + ' ' + (this.messages[this.messages.length - 1]?.content ?? '')).toLowerCase()
    // The split form first ("aide kin"): join it when the two halves spell the brand.
    let out = text.replace(/\b([a-z]{2,})[ ]([a-z]{2,})\b/gi, (m, a: string, b: string) => ((a + b).toLowerCase() === lower ? matchCase(m[0], brand) : m))
    out = out.replace(/\b[a-z]{3,}\b/gi, (w) => {
      const wl = w.toLowerCase()
      if (wl === lower || wl.slice(0, 3) !== lower.slice(0, 3)) return w
      if (!withinOneEdit(wl, lower)) return w
      if (protectedText.includes(wl)) return w // a real word from the user/context stays
      return matchCase(w[0], brand)
    })
    return out
  }

  /** A new turn is superseding a reply that is still streaming. Whatever already streamed was
   *  SEEN - and in voice mode already HEARD - so vanishing it desyncs the transcript from
   *  reality. Commit the partial as the reply, in order, before the new user turn is recorded. */
  private commitPartialReply(): void {
    if (this.currentId < 0) return
    const partial = this.fixBrandSpelling(stripHtmlTags(this.retriever ? guardFabrications(this.assistant, this.turnContext, false) : this.assistant)).trim()
    if (!partial) return
    this.assistant = partial
    this.pushAssistant(partial)
    this.cb.onAssistantText?.(partial, true)
  }

  /** Run one user turn: record it, (optionally) retrieve context, stream a reply. */
  sendUserMessage(text: string): Promise<string> {
    const clean = text.trim()
    if (!clean) return Promise.resolve('')
    this.commitPartialReply() // must precede pushUser (order) and the turnContext reset (guarding)
    this.turnContext = '' // reset per turn; applyContext refills it if this turn is grounded
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
      // Hybrid gate: clear the cosine threshold, OR be a strong lexical match (all distinctive query
      // terms present) so an exact menu-term query grounds even when the embedding underscores it.
      // Greetings have no distinctive terms, so lexMatch is never set for them and abstention holds.
      const relevant = hits.filter((h) => (h.score ?? 0) >= this.ragMinScore || (h.lexMatch && (h.score ?? 0) >= this.ragMinScore - RAG_LEX_MARGIN))
      // Log the scores so the gate can be calibrated against real greetings vs questions.
      dlog(
        `[aidekin] RAG "${userText.slice(0, 40)}" top=${(hits[0]?.score ?? 0).toFixed(3)} ` +
          `used=${relevant.length}/${hits.length} (gate ${this.ragMinScore}, lex ${hits.filter((h) => h.lexMatch).length})`,
      )
      this.cb.onRetrieval?.({ used: relevant.length, tookMs: performance.now() - t0 })
      if (relevant.length) this.applyContext(userText, relevant)
      // Nothing cleared the gate: no per-turn injection. The always-on SITE_GROUNDING rule (in the
      // system prompt) tells the model to abstain on site questions it wasn't given, while still
      // greeting, recalling what the user said, and answering general questions - in any language.
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
    // A single hit larger than the whole budget would otherwise inject an EMPTY <info> block and
    // force a wrong "I don't have that information"; truncate the top hit instead.
    if (!used && hits.length > 0) used = hits[0].text.slice(0, this.ragCharBudget)
    this.turnContext = used // the fabrication guard keeps only URLs/emails that appear verbatim here
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
    // turns don't queue and balloon ttft. Text already streamed was committed by
    // commitPartialReply; the supersededSink lets voice stop SPEAKING the stale reply too
    // (its clauses were already queued to TTS and would otherwise play out to the end).
    if (this.currentId >= 0) {
      const abortMsg: LlmIn = { kind: 'abort', id: this.currentId }
      this.llm.postMessage(abortMsg)
      this.pending?.resolve(this.assistant)
      this.pending = null
      this.supersededSink?.()
    }
    this.genId++
    const id = this.genId
    this.currentId = id
    this.chunker.reset()
    this.assistant = ''
    this.cb.onGenerationStart?.()
    const msg: LlmIn = { kind: 'generate', id, messages: request, think, resetCache: this.cacheDirty, seed: this.samplerSeed, promptLookup: this.promptLookup, wantConfidence: this.measureConfidence || undefined }
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
      const shown = this.fixBrandSpelling(stripHtmlTags(this.retriever ? guardFabrications(this.assistant, this.turnContext, true) : this.assistant))
      this.cb.onAssistantText?.(shown, false)
      if (this.chunkClauses) {
        const sink = this.clauseSink ?? this.cb.onAssistantClause
        for (const clause of this.chunker.push(m.text)) {
          const spoken = speakable(clause)
          if (spoken) sink?.(spoken)
        }
      }
    } else if (m.kind === 'done') {
      if (m.id !== this.currentId) return // stale generation (superseded/aborted)
      this.lastGenStats = { tps: m.tps, reusedCache: m.reusedCache, speculation: m.speculation, confidence: m.confidence }
      this.finish(m.text)
    } else if (m.kind === 'error') {
      // A superseded turn's error (e.g. its unwind after an abort) must not settle the turn
      // that replaced it; only errors for the CURRENT generation (or turnless ones) count.
      if (m.id !== undefined && m.id !== this.currentId) return
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
    const raw = this.assistant.trim() ? this.assistant : (doneText ?? '').trim()
    // Grounded turns: strip any URL/email the model invented or mangled (not verbatim in the
    // retrieved context) from the STORED + displayed answer, so it can't resurface on reload either.
    const text = this.fixBrandSpelling(stripHtmlTags(this.retriever ? guardFabrications(raw, this.turnContext, false) : raw))
    if (this.chunkClauses) {
      const sink = this.clauseSink ?? this.cb.onAssistantClause
      const rest = this.chunker.flush()
      if (rest) {
        const spoken = speakable(rest)
        if (spoken) sink?.(spoken)
      }
    }
    // Skip an empty reply (e.g. all tokens spent in a <think> block) - don't record it.
    if (text) {
      this.assistant = text
      // Keep the RAW reply as the turn's model form when the guard edited it: the worker's KV
      // cache holds the raw tokens, so sending the edited text would silently kill reuse.
      this.pushAssistant(text, raw !== text ? raw : undefined)
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
    // Keep whatever already streamed (the visitor read and, in voice, HEARD it) - this covers
    // BOTH interrupt paths: the barge-in (orchestrator calls abort() before the new transcript
    // reaches sendUserMessage, so the commit there would find no in-flight turn) and the text
    // widget's Stop button. Must run while currentId is still live.
    this.commitPartialReply()
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

  /** Invalidate the worker's KV-cache prefix, with a logged reason: a spurious dirty here is
   *  exactly what turns a warm first turn into a multi-second cold prefill, so make each
   *  occurrence attributable from the console. */
  private markDirty(reason: string): void {
    this.cacheDirty = true
    dlog(`[aidekin] LLM cache dirtied (${reason})`)
  }

  /** New session: clear back to just the system prompt (keeps it in storage). */
  reset(): void {
    this.abort()
    this.messages = [{ role: 'system', content: this.composedSystem() }]
    this.markDirty('chat reset')
    // Forget the worker's committed transcript AND delete its persisted KV snapshot, so a reload
    // right after "new chat" starts fresh instead of restoring the conversation we just cleared.
    this.llm?.postMessage({ kind: 'reset-session' } satisfies LlmIn)
    this.schedulePrewarm() // re-warm the system prefix so the first turn of the NEW chat is a cache-append
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

  /** Warm the worker's KV cache with the CURRENT system prompt so the user's first turn is a cheap
   *  cache-append, not a cold full prefill (the otherwise-hidden seconds before the first token).
   *  Debounced, so loading + the async RAG attach coalesce into ONE prewarm with the final system
   *  view. Best-effort; the worker serializes the prefill before any generation. Clears cacheDirty
   *  because the prewarm rebuilds the cache to match exactly this system view. */
  private schedulePrewarm(): void {
    if (!this.llm) return
    if (this.prewarmTimer) clearTimeout(this.prewarmTimer)
    this.prewarmTimer = setTimeout(() => {
      this.prewarmTimer = null
      if (!this.llm) return
      const system: ChatMessage = { role: 'system', content: this.composedSystem() }
      // Send the full model view: for a restored conversation the worker warms the whole
      // transcript (exactly what the next request will send), not just the system prompt.
      this.llm.postMessage({ kind: 'prewarm', system, messages: this.modelView() })
      this.cacheDirty = false
    }, 300)
  }

  /** Attach (or clear) RAG after construction - the index loads asynchronously. */
  setRetriever(retriever: Retriever | null): void {
    const had = !!this.retriever
    this.retriever = retriever
    // The grounding rules live in the system prompt only while RAG is active; toggling RAG
    // changes the cached prefix, so refresh the system message and invalidate the cache once.
    if (had !== !!retriever && this.messages[0]?.role === 'system') {
      this.messages[0] = { role: 'system', content: this.composedSystem() }
      this.markDirty('retriever toggled')
      this.schedulePrewarm() // re-warm the cache with the new (RAG) system view, off the first turn
    }
  }

  /** Toggle prompt-lookup speculative decoding at runtime (measurement/eval). */
  setPromptLookup(on: boolean): void {
    this.promptLookup = on
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

  /** Voice attaches here: called when a new turn supersedes an in-flight reply, so the
   *  orchestrator can stop the stale reply's audio (playback + queued TTS synths). */
  setSupersededSink(sink: (() => void) | null): void {
    this.supersededSink = sink
  }

  setSystemPrompt(prompt: string): void {
    // No-op when unchanged. The widget re-applies the configured prompt on mount (after the
    // constructor already set it), and an unconditional dirty there would invalidate the load-time
    // prewarm and force the FIRST turn into a cold full prefill. Same prompt → keep the warm cache.
    if (prompt === this.systemPrompt && this.messages[0]?.role === 'system') return
    this.systemPrompt = prompt
    if (this.messages.length && this.messages[0].role === 'system') {
      this.messages[0] = { role: 'system', content: this.composedSystem() }
    } else {
      this.messages.unshift({ role: 'system', content: this.composedSystem() })
    }
    this.markDirty('system prompt changed') // the cached prefix starts with the old system prompt
    this.schedulePrewarm() // re-warm with the new system prompt so the next turn still reuses the cache
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
    this.persist()
  }

  private pushAssistant(text: string, model?: string): void {
    this.messages.push(model ? { role: 'assistant', content: text, model } : { role: 'assistant', content: text })
    this.persist()
  }

  // No conversation-side sliding window: the LLM runs kvCache 'q8' with overflow 'sinks', so the
  // engine keeps a fixed window (attention sinks + the recent turns) and evicts the middle itself
  // as the chat grows - the transcript stays whole for display and cache-appends never break on a
  // trim. The one remaining overflow guard is prompt-side, in the worker: a SINGLE rendered prompt
  // larger than the window (e.g. a huge paste, or a cold re-prefill after the cache is dirtied)
  // still throws, and chat.send's onOverflow trims that one prompt and retries.

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
