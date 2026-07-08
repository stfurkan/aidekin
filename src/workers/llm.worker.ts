/// <reference lib="webworker" />
// LLM worker: the "brain" - PrismML Bonsai (Qwen3-architecture, 1-bit/binary weights) run on WebGPU
// via our own bitgpu engine. The text boundary - tokenizer, chat-template rendering, <think>
// stripping, UTF-8-safe streaming, and cross-turn KV-cache reuse with exact token bookkeeping - is
// owned by bitgpu/chat's createChat (the same @huggingface tokenizer/jinja libs we used, inlined into
// bitgpu and verified byte-exact, see scripts/verify-tokenizer.ts). This worker keeps only the
// message protocol and the latest-wins / barge-in coordinator; chat.send does the rest, and it fixes
// the reuse-delta <|im_end|> bookkeeping our hand-rolled version got subtly wrong on turn 2+.
import { createEngine, type Engine } from 'bitgpu'
import { createChat, type Chat } from 'bitgpu/chat'
import type { ChatMessage, LlmIn, LlmOut } from '../protocol/messages'
import { getModelAsset, getModelAssetStream } from '../core/modelStore'
import { withRetry } from '../core/retry'
import { dlog, setDebug } from '../core/log'

const ctx = self as unknown as DedicatedWorkerGlobalScope
const post = (m: LlmOut): void => ctx.postMessage(m)

// ── worker state ─────────────────────────────────────────────────────────────
let currentId: number | null = null
let engine: Engine | null = null
let chat: Chat | null = null
let maxSeqLen = 2048
let abortController: AbortController | null = null

/** Keep the system prompt plus as many of the most recent messages as plausibly fit the model's KV
 *  window (~4 chars/token, generous slack for the template + generation room). Recovery path for
 *  transcripts that outgrow maxSeqLen (e.g. persisted history restored on reload). */
function fitToWindow(messages: readonly ChatMessage[]): ChatMessage[] {
  const budgetChars = Math.max(512, maxSeqLen - 700) * 4
  const out = [...messages]
  const head = out[0]?.role === 'system' ? 1 : 0
  let chars = out.reduce((n, m) => n + m.content.length, 0)
  while (chars > budgetChars && out.length - head > 1) {
    chars -= out[head].content.length
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
      await generate(msg.id, msg.messages, msg.think ?? false, msg.resetCache ?? false, msg.seed, msg.promptLookup)
    } else if (msg.kind === 'abort') {
      if (queued && msg.id === queued.id) queued = null // cancel a queued turn before it ever starts
      if (msg.id === currentId) abortController?.abort() // chat.send owns the cache bookkeeping on abort
    } else if (msg.kind === 'prewarm') {
      await prewarm(msg.system, msg.messages)
    }
  } catch (err) {
    post({ kind: 'error', message: `LLM: ${(err as Error).message}` })
  }
}

async function init(msg: Extract<LlmIn, { kind: 'init' }>): Promise<void> {
  setDebug(msg.debug ?? false)
  maxSeqLen = msg.maxSeqLen ?? 2048
  chat?.reset()
  chat = null
  engine?.dispose() // re-init must not leak the previous GPUDevice (~300 MB VRAM)
  engine = null
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

  // STREAM the ~290MB data file (OPFS-cached; chunks flow straight into GPU buffers, so the whole file
  // never sits in the worker heap - the peak that got the tab killed on phones). Manifest + aux are small.
  const fetchStream = async (url: string): Promise<ReadableStream<Uint8Array>> => {
    if (url === msg.dataUrl) {
      return getModelAssetStream('llm-bonsai-1.7b-q1', url, (p) =>
        post({ kind: 'load', label: 'LLM', file: 'weights', detail: `weights ${Math.round((100 * p.loaded) / (p.total || 1))}%`, loaded: p.loaded, total: p.total || 0 }),
      )
    }
    const res = await fetch(url)
    if (!res.ok) throw new Error(`fetch ${url} failed: HTTP ${res.status}`)
    return res.body as ReadableStream<Uint8Array>
  }
  const fetchArrayBuffer = async (url: string): Promise<ArrayBuffer> => {
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

  await warmup()
  const cap = engine.capabilities
  post({ kind: 'ready', info: `bitgpu (${cap.useSubgroups ? 'subgroups SG=' + cap.subgroupSize : 'workgroup fallback'}, kv ${cap.kvCache})` })
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
  if (!chat || running || !chat.tokenizer.hasChatTemplate) return // never disturb a live turn
  const c = chat
  const job = (async () => {
    try {
      // Prefer the full transcript (a restored conversation); fall back to just the system prompt when
      // it is trivial or too long to leave append room. Length estimated with the chat's own tokenizer.
      let transcript: readonly ChatMessage[] = messages && messages.length > 1 ? messages : [system]
      const tokenCount = (msgs: readonly ChatMessage[]): number =>
        c.tokenizer.encode(c.tokenizer.applyChatTemplate([...msgs], { addGenerationPrompt: false }), false).length
      if (transcript.length > 1 && tokenCount(transcript) > maxSeqLen - 600) transcript = [system]
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
async function generate(id: number, messages: readonly ChatMessage[], allowThink: boolean, resetCache: boolean, seed?: number, promptLookup?: boolean): Promise<void> {
  queued = { id, messages, allowThink, resetCache, seed, promptLookup } // latest-wins
  if (currentId !== null && currentId >= 0) abortController?.abort()
  if (running) return
  running = true
  try {
    while (queued) {
      const job = queued
      queued = null
      try {
        await runGeneration(job.id, job.messages, job.allowThink, job.resetCache, job.seed, job.promptLookup)
      } catch (err) {
        post({ kind: 'error', id: job.id, message: `LLM: ${(err as Error).message}` })
      }
    }
  } finally {
    running = false
  }
}

async function runGeneration(id: number, messages: readonly ChatMessage[], allowThink: boolean, resetCache: boolean, seed?: number, promptLookup?: boolean): Promise<void> {
  if (!chat) throw new Error('LLM not initialized')
  currentId = id
  abortController = new AbortController()

  // If a system-prompt prewarm is still in flight, let it finish first: it populates the cache this
  // turn may reuse, and the engine must never run a prefill and a decode concurrently.
  if (prewarmPromise) {
    try { await prewarmPromise } catch { /* prewarm is best-effort */ }
    prewarmPromise = null
  }

  // chat.send owns the reuse decision (clean append, non-think, ChatML), the reuse delta (with correct
  // <|im_end|> re-insertion), <think> stripping, and the cross-turn cache bookkeeping. We pass
  // reuseCache=false when the caller forces a reset (new session / cleared chat / system change).
  const opts = {
    maxTokens: allowThink ? 1024 : 512, // room for the (stripped) <think> block + answer
    temperature: SAMPLING.temperature,
    topK: SAMPLING.topK,
    topP: SAMPLING.topP,
    repetitionPenalty: SAMPLING.repetitionPenalty,
    noRepeatNgramSize: SAMPLING.noRepeatNgramSize,
    seed, // undefined in production (entropy); fixed by the behavioral eval for determinism
    promptLookup: promptLookup ?? false,
    reuseCache: !resetCache,
    think: allowThink,
    signal: abortController.signal,
    onText: (delta: string): void => {
      if (currentId === id) post({ kind: 'token', id, text: delta })
    },
  }

  let result
  try {
    result = await chat.send([...messages], opts)
  } catch (err) {
    if (!/maxSeqLen/.test((err as Error).message)) throw err
    // The transcript outgrew the model's KV window (only the PROMPT can trigger this; maxTokens is
    // clamped by the engine). Recover instead of bricking the chat: keep the system prompt + the most
    // recent turns and full-prefill once (a trimmed list is not a clean append, so chat.send rebuilds).
    const trimmed = fitToWindow(messages)
    console.warn(`[aidekin] LLM transcript exceeded the ${maxSeqLen}-token window; trimmed ${messages.length - trimmed.length} old message(s) and retried`)
    result = await chat.send(trimmed, { ...opts, reuseCache: false })
  }

  dlog(
    `[aidekin] LLM(bonsai) done id=${id} ${result.reusedCache ? 'cache-reuse' : 'full-prefill'} ` +
      `${result.tokensPerSecond.toFixed(1)} tok/s (ttft ${result.prefillMs.toFixed(0)}ms, ${result.tokens.length} tok)`,
  )
  post({ kind: 'done', id, text: result.text, tps: result.tokensPerSecond })
  if (currentId === id) currentId = null
}
