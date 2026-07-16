/// <reference lib="webworker" />
// LLM worker: the "brain" - PrismML Bonsai (Qwen3-architecture, 1-bit/binary weights) run on WebGPU
// via our own bitgpu engine. The text boundary - tokenizer, chat-template rendering, <think>
// stripping, UTF-8-safe streaming, and cross-turn KV-cache reuse with exact token bookkeeping - is
// owned by bitgpu/chat's createChat (the same @huggingface tokenizer/jinja libs we used, inlined into
// bitgpu and verified byte-exact, see scripts/verify-tokenizer.ts). This worker keeps only the
// message protocol and the latest-wins / barge-in coordinator; chat.send does the rest, and it fixes
// the reuse-delta <|im_end|> bookkeeping our hand-rolled version got subtly wrong on turn 2+.
import { createEngine, type Engine, type TokenLogprobs } from 'bitgpu'
import { createChat, type Chat, type ChatSnapshot } from 'bitgpu/chat'
import type { ChatMessage, LlmConfidence, LlmIn, LlmOut } from '../protocol/messages'
import { getModelAsset, getModelAssetStream, getSmallAsset } from '../core/modelStore'
import { sessionDelete, sessionGet, sessionPut } from '../core/sessionStore'
import { withRetry } from '../core/retry'
import { dlog, setDebug } from '../core/log'

const ctx = self as unknown as DedicatedWorkerGlobalScope
const post = (m: LlmOut): void => ctx.postMessage(m)

// ── worker state ─────────────────────────────────────────────────────────────
let currentId: number | null = null
let engine: Engine | null = null
let chat: Chat | null = null
let abortController: AbortController | null = null

// ── cross-reload session (chat.save/restore) ─────────────────────────────────
// Non-null only when persistence is opted in (init.persistSession set). Keyed by session
// namespace + model + kvCache mode + window, so a snapshot is never restored across a model or
// mode change (chat.restore would throw on mismatch anyway; the key avoids even attempting it).
let sessionKey: string | null = null
// The load restored a conversation from its snapshot, so the cache is already warm with the whole
// transcript: the main thread's first prewarm (which would re-prefill it) is skipped once.
let restoredThisLoad = false
// Coalesced, best-effort snapshot writes: at most one save runs at a time; a save requested while
// one is in flight sets `saveAgain` so a trailing save captures the latest turn. `sessionEpoch`
// bumps on reset-session so a save that completes after a "new chat" never rewrites the deleted snapshot.
let saving = false
let saveAgain = false
let sessionEpoch = 0

/** onOverflow trim policy: keep the system prompt plus as many of the most recent messages as fit the
 *  model's KV window, measured EXACTLY with chat.countTokens (not a char estimate), reserving `reserve`
 *  tokens of the window for the answer. bitgpu calls this only when a prompt overruns maxSeqLen (e.g. a
 *  long persisted transcript restored on reload) and retries once with a clean full prefill of the result. */
function trimToWindow(c: Chat, messages: readonly ChatMessage[], window: number, reserve: number, think: boolean): ChatMessage[] {
  const budget = Math.max(256, window - reserve)
  const out = [...messages]
  const head = out[0]?.role === 'system' ? 1 : 0
  while (out.length - head > 1 && c.countTokens(out, { addGenerationPrompt: true, think }) > budget) {
    out.splice(head, 1) // drop the oldest non-system message; the latest user turn is always kept
  }
  return out
}

// ── single-flight generation queue (latest-wins) ──────────────────────────────
// Only one generation may run on the GPU at a time; a new request aborts the running one and is
// stashed as `queued`; the loop picks up only the LATEST queued request once the current run unwinds.
interface GenJob {
  id: number
  messages: readonly ChatMessage[]
  allowThink: boolean
  resetCache: boolean
  seed?: number
  promptLookup?: boolean
  wantConfidence?: boolean
}
let running = false
let queued: GenJob | null = null
// A background system-prompt prewarm (chat.prewarm). runGeneration awaits it before generating so a
// prewarm prefill never overlaps a decode and the first turn sees the warm cache.
let prewarmPromise: Promise<void> | null = null

ctx.onmessage = (ev: MessageEvent<LlmIn>) => {
  void handle(ev.data)
}

async function handle(msg: LlmIn): Promise<void> {
  try {
    if (msg.kind === 'init') {
      await init(msg)
    } else if (msg.kind === 'generate') {
      await generate(msg.id, msg.messages, msg.think ?? false, msg.resetCache ?? false, msg.seed, msg.promptLookup, msg.wantConfidence)
    } else if (msg.kind === 'abort') {
      if (queued && msg.id === queued.id) queued = null // cancel a queued turn before it ever starts
      if (msg.id === currentId) abortController?.abort() // chat.send owns the cache bookkeeping on abort
    } else if (msg.kind === 'reset-session') {
      await resetSession()
    } else if (msg.kind === 'prewarm') {
      await prewarm(msg.system, msg.messages)
    }
  } catch (err) {
    post({ kind: 'error', message: `LLM: ${(err as Error).message}` })
  }
}

async function init(msg: Extract<LlmIn, { kind: 'init' }>): Promise<void> {
  setDebug(msg.debug ?? false)
  chat?.reset()
  chat = null
  engine?.dispose() // re-init must not leak the previous GPUDevice (~300 MB VRAM)
  engine = null
  restoredThisLoad = false
  // Snapshot persistence is opt-in (persistSession set) and namespaced so a snapshot is only ever
  // restored into the SAME model + kvCache mode + window it was saved from.
  const kvMode = msg.kvCache ?? 'f32'
  sessionKey = msg.persistSession ? `${msg.persistSession}::${msg.dataUrl}::${kvMode}/${msg.overflow ?? 'error'}/${msg.maxSeqLen ?? 2048}` : null
  const onRetry = (n: number, _e: unknown, ms: number): void => console.warn(`[aidekin] LLM load failed (transient); retry ${n} in ${ms}ms`)

  // Tokenizer JSON, routed through the OPFS model cache so a fully cached model boots offline (the
  // ~7MB tokenizer.json is fetched once). createChat is given the preloaded JSON, so bitgpu/chat never
  // fetches it itself and our caching + offline behaviour is preserved. Fetched with the engine below.
  const base = `https://huggingface.co/${msg.tokenizerModelId}/resolve/main`
  const fetchTokJson = (file: string): Promise<unknown> =>
    withRetry(async (): Promise<unknown> => {
      const bytes = await getModelAsset(`llm-tokenizer/${msg.tokenizerModelId}/${file}`, `${base}/${file}`)
      return JSON.parse(new TextDecoder().decode(bytes)) as unknown
    }, { onRetry })

  // STREAM the ~237MB GGUF data file (OPFS-cached; chunks flow straight into GPU buffers, so the whole
  // file never sits in the worker heap - the peak that got the tab killed on phones). The cache key
  // carries the container ('...q1_0-gguf'): it changed from the old ONNX key so a returning visitor
  // re-streams the GGUF instead of the marker serving stale ONNX bytes for the new URL.
  const fetchStream = async (url: string): Promise<ReadableStream<Uint8Array>> => {
    if (url === msg.dataUrl) {
      return getModelAssetStream('llm-bonsai-1.7b-q1_0-gguf', url, (p) =>
        post({ kind: 'load', label: 'LLM', file: 'weights', detail: `weights ${Math.round((100 * p.loaded) / (p.total || 1))}%`, loaded: p.loaded, total: p.total || 0 }),
      )
    }
    const res = await fetch(url)
    if (!res.ok) throw new Error(`fetch ${url} failed: HTTP ${res.status}`)
    return res.body as ReadableStream<Uint8Array>
  }
  // Manifest + aux are small but must ALSO come from the OPFS cache, or a warm-cache boot dies offline
  // on them (bitgpu routes the manifest through fetchJson and the aux through fetchArrayBuffer, neither
  // of which touched the store before). getSmallAsset keeps the SPA-fallback guard so a bad deploy that
  // serves index.html is never cached as model data.
  // Keys carry the '-gguf' container tag for the same reason as the weights key: the manifest + aux
  // content differs from the old ONNX pair, so a returning visitor must fetch the GGUF ones afresh.
  const fetchJson = (url: string): Promise<unknown> =>
    withRetry(async (): Promise<unknown> => {
      const bytes = await getSmallAsset(`llm-manifest-gguf/${msg.tokenizerModelId}`, url)
      return JSON.parse(new TextDecoder().decode(bytes)) as unknown
    }, { onRetry })
  const fetchArrayBuffer = async (url: string): Promise<ArrayBuffer> => {
    if (url === msg.auxUrl) return getSmallAsset(`llm-aux-gguf/${msg.tokenizerModelId}`, url)
    const res = await fetch(url)
    if (!res.ok) throw new Error(`fetch ${url} failed: HTTP ${res.status}`)
    if ((res.headers.get('content-type') ?? '').includes('text/html')) throw new Error(`${url} returned HTML (SPA fallback), not model data`)
    return res.arrayBuffer()
  }

  const [tokJson, tokCfg, eng] = await Promise.all([
    fetchTokJson('tokenizer.json'),
    fetchTokJson('tokenizer_config.json'),
    withRetry(
      () =>
        createEngine({
          manifestUrl: msg.manifestUrl,
          dataUrl: msg.dataUrl,
          auxUrl: msg.auxUrl,
          maxSeqLen: msg.maxSeqLen ?? 2048,
          kvCache: msg.kvCache,
          overflow: msg.overflow, // 'sinks' = unbounded chat in a fixed window (the engine evicts the middle)
          fetchJson,
          fetchStream,
          fetchArrayBuffer,
          onProgress: (p) => post({ kind: 'load', label: 'LLM', detail: p.phase, loaded: 0, total: 0 }),
        }),
      { onRetry },
    ),
  ])
  engine = eng

  // Surface an unexpected GPU device loss (driver reset, OS reclaim) instead of hanging "thinking".
  const current = engine
  void current.lost.then((info) => {
    if (info.reason === 'destroyed' || engine !== current) return // dispose/re-init, not a failure
    engine = null
    chat = null
    post({ kind: 'error', message: `LLM: GPU device lost (${info.message || info.reason}); close and reopen the chat to reload` })
  })

  chat = await createChat(engine, { tokenizer: { json: tokJson, config: tokCfg as Record<string, unknown> } })
  if (!chat.tokenizer.hasChatTemplate) console.warn('[aidekin] LLM: non-ChatML template - cross-turn KV reuse disabled')

  await warmup() // exercises the decode path on a throwaway turn and resets to empty...
  await restoreSession() // ...then brings the persisted conversation back into a warm cache (no re-prefill)
  const cap = engine.capabilities
  post({ kind: 'ready', info: `bitgpu (${cap.useSubgroups ? 'subgroups SG=' + cap.subgroupSize : 'workgroup fallback'}, kv ${cap.kvCache})` })
}

/** Bring back the persisted conversation so the next turn extends the cache with no re-prefill.
 *  Best-effort: a missing snapshot is a fresh start; a stale one (model / kvCache mode / window
 *  changed) throws in chat.restore and is discarded. Runs AFTER warmup so warmup's reset can't wipe it. */
async function restoreSession(): Promise<void> {
  if (!sessionKey || !chat) return
  try {
    const snap = await sessionGet<ChatSnapshot>(sessionKey)
    if (!snap) return
    await chat.restore(snap) // validates model + kvCache mode; throws on mismatch
    restoredThisLoad = true
    dlog(`[aidekin] LLM session restored (${snap.committed?.length ?? 0} msg) - first turn is a cache-append`)
  } catch (e) {
    console.warn('[aidekin] LLM session restore skipped:', (e as Error).message)
    void sessionDelete(sessionKey) // a snapshot that no longer fits this model/mode is dead weight
  }
}

/** New conversation: forget the transcript + KV cache and delete the persisted snapshot, so a
 *  reload after "new chat" starts fresh instead of restoring the old conversation. The epoch bump
 *  makes any save still in flight (from the just-ended conversation) a no-op. */
async function resetSession(): Promise<void> {
  sessionEpoch++
  saveAgain = false
  restoredThisLoad = false
  chat?.reset()
  if (sessionKey) await sessionDelete(sessionKey)
}

/** Persist the conversation snapshot after a completed turn. Coalesced (one write at a time, a
 *  trailing write captures the latest turn) and best-effort - a quota/write failure is swallowed.
 *  Guarded by sessionEpoch so a save that finishes after a reset-session never rewrites the deleted
 *  snapshot. chat.save() queues behind in-flight turns and returns null when nothing is committed. */
function scheduleSave(): void {
  if (!sessionKey || !chat) return
  if (saving) {
    saveAgain = true
    return
  }
  saving = true
  const key = sessionKey
  void (async () => {
    try {
      do {
        saveAgain = false
        const epoch = sessionEpoch
        const snap = await chat?.save()
        if (snap && sessionEpoch === epoch && sessionKey === key) {
          await sessionPut(key, snap)
          // A reset-session ('new chat') can land while chat.save()/sessionPut are in flight, racing
          // its own delete against this write. Re-check after the PUT: if the epoch moved, that clear
          // must win, so undo the snapshot we just wrote (a cleared conversation leaves nothing to restore).
          if (sessionEpoch !== epoch) await sessionDelete(key)
        }
      } while (saveAgain && chat && sessionKey === key)
    } catch (e) {
      dlog(`[aidekin] LLM session save skipped: ${(e as Error).message}`)
    } finally {
      saving = false
    }
  })()
}

/** Warm the decode path once so the user's FIRST message isn't a cold start. Best-effort; resets the
 *  chat afterward so nothing leaks into the conversation. */
async function warmup(): Promise<void> {
  if (!chat) return
  try {
    const t0 = performance.now()
    await chat.send([{ role: 'user', content: 'Hi' }], { maxTokens: 1, ...SAMPLING })
    chat.reset()
    dlog(`[aidekin] LLM warmup ${(performance.now() - t0).toFixed(0)}ms`)
  } catch (e) {
    console.warn('[aidekin] LLM warmup skipped:', (e as Error).message)
  }
}

/** Prefill the static system prompt (or a restored transcript) into the KV cache at load, so the
 *  user's FIRST turn is a cheap cache-append instead of a cold full prefill. chat.prewarm renders the
 *  prefix so it ends exactly at <|im_end|>; the next turn's reuse delta continues token-for-token.
 *  Best-effort; skipped mid-generation and on non-ChatML templates. */
async function prewarm(system: ChatMessage, messages?: readonly ChatMessage[]): Promise<void> {
  if (restoredThisLoad) {
    // A snapshot restore already warmed the cache with the whole transcript; re-prefilling it here
    // would be redundant work (prewarm always prefills). Consume the flag so a later prewarm (e.g.
    // after a system-prompt change) still runs.
    restoredThisLoad = false
    dlog('[aidekin] LLM prewarm skipped (session restored)')
    return
  }
  if (!chat || running || !chat.tokenizer.hasChatTemplate) return // never disturb a live turn
  const c = chat
  const job = (async () => {
    try {
      // Prefer the full transcript (a restored conversation); fall back to just the system prompt when
      // it is trivial or too long to leave append room. Length estimated with the chat's own tokenizer.
      let transcript: readonly ChatMessage[] = messages && messages.length > 1 ? messages : [system]
      // Length measured exactly with the chat's own tokenizer (chat.countTokens); fall back to just the
      // system prompt when a restored transcript is too long to leave the next turn append room.
      const window = engine?.capabilities.maxSeqLen ?? 2048
      if (transcript.length > 1 && c.countTokens([...transcript], { addGenerationPrompt: false }) > window - 600) transcript = [system]
      const t0 = performance.now()
      await c.prewarm([...transcript])
      dlog(`[aidekin] LLM prewarm ${(performance.now() - t0).toFixed(0)}ms (${transcript.length} msg)`)
    } catch (e) {
      console.warn('[aidekin] LLM prewarm skipped:', (e as Error).message)
    }
  })()
  prewarmPromise = job
  await job
}

// Sampling params for the bitgpu engine's do_sample. Temperature is 0.3: this is a knowledge assistant,
// so we bias toward the highest-probability, context-faithful continuation and away from the "creative"
// tail that invents details. topP is accepted but not applied (a no-op) - bitgpu's sampler is bit-exact
// with the transformers.js v4.2.0 reference we validate against, where top_p is also disabled.
const SAMPLING = { temperature: 0.3, topK: 20, topP: 0.85, repetitionPenalty: 1.15, noRepeatNgramSize: 0 } as const

/** Coordinator: enqueue this turn as the latest, abort anything running, and drain the queue ONE
 *  generation at a time. */
async function generate(id: number, messages: readonly ChatMessage[], allowThink: boolean, resetCache: boolean, seed?: number, promptLookup?: boolean, wantConfidence?: boolean): Promise<void> {
  queued = { id, messages, allowThink, resetCache, seed, promptLookup, wantConfidence } // latest-wins
  if (currentId !== null && currentId >= 0) abortController?.abort()
  if (running) return
  running = true
  try {
    while (queued) {
      const job = queued
      queued = null
      try {
        await runGeneration(job.id, job.messages, job.allowThink, job.resetCache, job.seed, job.promptLookup, job.wantConfidence)
      } catch (err) {
        post({ kind: 'error', id: job.id, message: `LLM: ${(err as Error).message}` })
      }
    }
  } finally {
    running = false
  }
}

/** Summarize per-token logprobs (bitgpu's true full-vocab log-softmax) into a compact confidence
 *  signal: the geometric-mean token probability, and the fraction of tokens emitted below p=0.3. */
const LOW_CONF_LOGPROB = Math.log(0.3)
function summarizeConfidence(logprobs: TokenLogprobs[]): LlmConfidence {
  const n = logprobs.length
  if (!n) return { meanProb: 1, lowConfFrac: 0, tokens: 0 }
  let sum = 0
  let low = 0
  for (const lp of logprobs) {
    sum += lp.logprob
    if (lp.logprob < LOW_CONF_LOGPROB) low++
  }
  return { meanProb: Math.exp(sum / n), lowConfFrac: low / n, tokens: n }
}

async function runGeneration(id: number, messages: readonly ChatMessage[], allowThink: boolean, resetCache: boolean, seed?: number, promptLookup?: boolean, wantConfidence?: boolean): Promise<void> {
  if (!chat) throw new Error('LLM not initialized')
  const c = chat
  currentId = id
  abortController = new AbortController()

  // If a system-prompt prewarm is still in flight, let it finish first: it populates the cache this
  // turn may reuse, and the engine must never run a prefill and a decode concurrently.
  if (prewarmPromise) {
    try { await prewarmPromise } catch { /* prewarm is best-effort */ }
    prewarmPromise = null
  }

  const maxTokens = allowThink ? 1024 : 512 // room for the (stripped) <think> block + answer

  // chat.send owns the reuse decision (clean append, non-think, ChatML), the reuse delta (with correct
  // <|im_end|> re-insertion), <think> stripping, and the cross-turn cache bookkeeping. reuseCache=false
  // forces a full prefill when the caller resets (new session / cleared chat / system change). onOverflow
  // is bitgpu's window-recovery hook: only the PROMPT can overrun maxSeqLen (a long restored transcript),
  // and bitgpu hands us the count + window, takes our trimmed list, and retries ONCE with a clean prefill.
  // (chat.send also offers stopSequences and format:'json' / {json:{schema}} for stops + schema-valid
  //  JSON; this knowledge-assistant flow needs neither.)
  const result = await c.send([...messages], {
    maxTokens,
    ...SAMPLING,
    seed, // undefined in production (entropy); fixed by the behavioral eval for determinism
    promptLookup: promptLookup ?? 'auto', // measure draft acceptance on probation, drop speculation when it doesn't pay; output is bit-identical either way. Explicit false still disables it.
    reuseCache: !resetCache,
    think: allowThink,
    // Opt-in per turn: ask bitgpu for the emitted tokens' true logprobs so we can report confidence.
    // Costs a few % (it disables promptLookup and routes greedy turns through the sampler path), so it
    // is only set when the caller asked; hot paths leave it off.
    ...(wantConfidence ? { logprobs: 5 } : {}),
    signal: abortController.signal,
    onText: (delta: string): void => {
      if (currentId === id) post({ kind: 'token', id, text: delta })
    },
    onOverflow: ({ maxSeqLen: window }): ChatMessage[] => {
      const trimmed = trimToWindow(c, messages, window, maxTokens, allowThink)
      console.warn(`[aidekin] LLM transcript exceeded the ${window}-token window; trimmed ${messages.length - trimmed.length} old message(s) and retried`)
      return trimmed
    },
  })

  // logprobs align 1:1 with the emitted tokens; summarize into a confidence signal for the caller.
  const confidence = result.logprobs ? summarizeConfidence(result.logprobs) : undefined
  dlog(
    `[aidekin] LLM(bonsai) done id=${id} ${result.reusedCache ? 'cache-reuse' : 'full-prefill'} ` +
      `${result.tokensPerSecond.toFixed(1)} tok/s (ttft ${result.prefillMs.toFixed(0)}ms, ${result.tokens.length} tok)` +
      (confidence ? ` conf=${confidence.meanProb.toFixed(2)} low=${(confidence.lowConfFrac * 100).toFixed(0)}%` : ''),
  )
  post({ kind: 'done', id, text: result.text, tps: result.tokensPerSecond, reusedCache: result.reusedCache, confidence })
  if (currentId === id) currentId = null
  scheduleSave() // persist the extended conversation off the critical path (after the reply is delivered)
}
