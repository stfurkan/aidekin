/// <reference lib="webworker" />
// LLM worker: the "brain" - PrismML Bonsai (Qwen3-architecture, 1-bit/binary weights) run on WebGPU
// via our own bitgpu engine (NO transformers.js / onnxruntime). The tokenizer is the
// standalone LlmTokenizer (@huggingface/tokenizers + @huggingface/jinja, byte-exact with HF). Streams
// tokens, strips Qwen3 <think> blocks, toggles reasoning via the chat template's enable_thinking flag,
// reuses the engine's cross-turn KV cache (prefill only the new turn), and logs tokens/sec.
import { createEngine, type Engine, type GenerateResult } from 'bitgpu'
import type { ChatMessage, LlmIn, LlmOut } from '../protocol/messages'
import { LlmTokenizer } from '../core/tokenizer'
import { getModelAsset } from '../core/modelStore'
import { withRetry } from '../core/retry'

const ctx = self as unknown as DedicatedWorkerGlobalScope
const post = (m: LlmOut): void => ctx.postMessage(m)

// ── Qwen3 <think> stripping ──────────────────────────────────────────────────
// Bonsai keeps Qwen3's chat template and can emit <think>...</think> blocks we must never show or
// speak. This strips them from the token STREAM (tags can straddle token boundaries), holding back
// only a possible partial tag at each chunk's edge.
const THINK_OPEN = '<think>'
const THINK_CLOSE = '</think>'
function holdback(s: string, tag: string): number {
  const max = Math.min(s.length, tag.length - 1)
  for (let k = max; k > 0; k--) if (tag.startsWith(s.slice(s.length - k))) return s.length - k
  return s.length
}
class ThinkFilter {
  private inThink = false
  private hold = ''
  push(text: string): string {
    let s = this.hold + text
    this.hold = ''
    let out = ''
    for (;;) {
      if (!this.inThink) {
        const i = s.indexOf(THINK_OPEN)
        if (i === -1) {
          const safe = holdback(s, THINK_OPEN)
          out += s.slice(0, safe)
          this.hold = s.slice(safe)
          return out
        }
        out += s.slice(0, i)
        s = s.slice(i + THINK_OPEN.length)
        this.inThink = true
      } else {
        const i = s.indexOf(THINK_CLOSE)
        if (i === -1) {
          this.hold = s.slice(holdback(s, THINK_CLOSE))
          return out
        }
        s = s.slice(i + THINK_CLOSE.length)
        this.inThink = false
      }
    }
  }
  flush(): string {
    const r = this.inThink ? '' : this.hold
    this.hold = ''
    this.inThink = false
    return r
  }
}

/** True iff `next` is exactly `cached` plus one new trailing user turn - a clean append, so the engine
 *  can extend its KV cache with just that turn instead of re-prefilling. We track committed MESSAGES
 *  (not token ids) because Bonsai's template renders past assistant turns differently from the live one
 *  (empty <think> block), so a re-tokenized history is never a token-prefix of what's cached. */
function isCleanAppend(cached: readonly ChatMessage[] | null, next: readonly ChatMessage[]): boolean {
  if (!cached || next.length !== cached.length + 1) return false
  if (next[next.length - 1].role !== 'user') return false
  for (let i = 0; i < cached.length; i++) {
    if (next[i].role !== cached[i].role || next[i].content !== cached[i].content) return false
  }
  return true
}

/** Chat-template wrappers, derived from the tokenizer at init (never hardcoded). `genPrompt` is what
 *  add_generation_prompt appends; `userPrefix`/`userSuffix` wrap a user turn. null when the model isn't
 *  standard ChatML, in which case cross-turn reuse is disabled (full-prefill each turn - correct, slower). */
let chatWrap: { genPrompt: string; userPrefix: string; userSuffix: string } | null = null

function deriveChatWrap(tk: LlmTokenizer): typeof chatWrap {
  try {
    const render = (msgs: ChatMessage[], agp: boolean): string => tk.applyChatTemplate(msgs, { addGenerationPrompt: agp, enableThinking: false })
    const SENT = 'SENT'
    const userOnly = render([{ role: 'user', content: SENT }], false)
    const userGen = render([{ role: 'user', content: SENT }], true)
    const genPrompt = userGen.slice(userOnly.length) // e.g. "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    const i = userOnly.indexOf(SENT)
    if (i < 0 || !genPrompt.includes('assistant')) return null
    const userPrefix = userOnly.slice(0, i) // "<|im_start|>user\n"
    const userSuffix = userOnly.slice(i + SENT.length) // "<|im_end|>\n"
    if (!userPrefix.includes('<|im_start|>') || !userSuffix.includes('<|im_end|>')) return null
    return { genPrompt, userPrefix, userSuffix }
  } catch {
    return null
  }
}

// ── worker state ─────────────────────────────────────────────────────────────
let currentId: number | null = null
let engine: Engine | null = null
let tokenizer: LlmTokenizer | null = null
let eosTokenId = 151645
let maxSeqLen = 2048
let abortController: AbortController | null = null

/** Keep the system prompt plus as many of the most recent messages as plausibly fit the model's
 *  KV window (rough 4 chars/token, generous slack for the template + generation room). Recovery
 *  path for transcripts that outgrow maxSeqLen (e.g. persisted history restored on reload). */
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
// A background system-prompt prewarm (engine.prefill). runGeneration awaits it before deciding
// cache reuse, so a prewarm prefill can never overlap a decode and the first turn sees the warm cache.
let prewarmPromise: Promise<void> | null = null

// ── cross-turn cache bookkeeping ───────────────────────────────────────────────
// The engine owns the physical KV cache (generate(delta, {reuseCache}) / resetCache). We track the
// committed conversation it represents so the next clean append can extend it. `invalidateCache` is
// set on abort: the partial cache no longer matches a clean prefix and must be reset.
let cachedMessages: ChatMessage[] | null = null
let invalidateCache = false

function dropCache(): void {
  engine?.resetCache()
  cachedMessages = null
}

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
      if (msg.id === currentId) {
        invalidateCache = true // partial generation -> cache no longer matches a clean prefix
        abortController?.abort()
      }
    } else if (msg.kind === 'prewarm') {
      await prewarm(msg.system)
    }
  } catch (err) {
    post({ kind: 'error', message: `LLM: ${(err as Error).message}` })
  }
}

async function init(msg: Extract<LlmIn, { kind: 'init' }>): Promise<void> {
  eosTokenId = msg.eosTokenId ?? 151645
  maxSeqLen = msg.maxSeqLen ?? 2048
  dropCache()
  engine?.dispose() // re-init must not leak the previous GPUDevice (~300 MB VRAM)
  engine = null
  const onRetry = (n: number, _e: unknown, ms: number): void => console.warn(`[aidekin] LLM load failed (transient); retry ${n} in ${ms}ms`)

  // Tokenizer (standalone, byte-exact with transformers.js). The ~7MB tokenizer.json is fetched once.
  tokenizer = await withRetry(() => LlmTokenizer.load({ modelId: msg.tokenizerModelId }), { onRetry })
  chatWrap = deriveChatWrap(tokenizer)
  if (!chatWrap) console.warn('[aidekin] LLM: non-ChatML template - cross-turn KV reuse disabled')

  // Engine: OPFS-cache the ~290MB data file (so it never re-downloads); manifest + aux are tiny.
  const fetchArrayBuffer = async (url: string): Promise<ArrayBuffer> => {
    if (url === msg.dataUrl) {
      return getModelAsset('llm-bonsai-1.7b-q1', url, (p) =>
        post({ kind: 'load', label: 'LLM', file: 'weights', detail: `weights ${Math.round((100 * p.loaded) / (p.total || 1))}%`, loaded: p.loaded, total: p.total || 0 }),
      )
    }
    const res = await fetch(url)
    if (!res.ok) throw new Error(`fetch ${url} failed: HTTP ${res.status}`)
    if ((res.headers.get('content-type') ?? '').includes('text/html')) throw new Error(`${url} returned HTML (SPA fallback), not model data`)
    return res.arrayBuffer()
  }
  engine = await withRetry(
    () =>
      createEngine({
        manifestUrl: msg.manifestUrl,
        dataUrl: msg.dataUrl,
        auxUrl: msg.auxUrl,
        maxSeqLen: msg.maxSeqLen ?? 2048,
        fetchArrayBuffer,
        onProgress: (p) => post({ kind: 'load', label: 'LLM', detail: p.phase, loaded: 0, total: 0 }),
      }),
    { onRetry },
  )

  // Surface an unexpected GPU device loss (driver reset, OS reclaim) instead of hanging "thinking".
  const current = engine
  void current.lost.then((info) => {
    if (info.reason === 'destroyed' || engine !== current) return // dispose/re-init, not a failure
    engine = null
    post({ kind: 'error', message: `LLM: GPU device lost (${info.message || info.reason}); close and reopen the chat to reload` })
  })

  await warmup()
  const cap = engine.capabilities
  post({ kind: 'ready', info: `bitgpu (${cap.useSubgroups ? 'subgroups SG=' + cap.subgroupSize : 'workgroup fallback'})` })
}

/** Warm the decode path once so the user's FIRST message isn't a cold start. Best-effort; resets the
 *  cache afterward so nothing leaks into the conversation. */
async function warmup(): Promise<void> {
  if (!engine || !tokenizer) return
  try {
    const t0 = performance.now()
    const ids = tokenizer.encode(tokenizer.applyChatTemplate([{ role: 'user', content: 'Hi' }], { addGenerationPrompt: true, enableThinking: false }), false)
    await engine.generate(ids, { maxTokens: 1, ...SAMPLING, stopTokens: [eosTokenId] })
    engine.resetCache()
    console.info(`[aidekin] LLM warmup ${(performance.now() - t0).toFixed(0)}ms`)
  } catch (e) {
    console.warn('[aidekin] LLM warmup skipped:', (e as Error).message)
  }
}

/** Prefill the static system prompt into the KV cache at load, so the user's FIRST turn is a cheap
 *  cache-append instead of a cold full prefill (the otherwise-hidden few seconds before the first
 *  token). We render the system block and drop its trailing newline so the cache ends exactly at
 *  <|im_end|>; the standard cache-reuse delta (which begins with "\n", see runGeneration) then
 *  reconstructs the first [system,user] prompt token-for-token. Best-effort; skipped mid-generation. */
async function prewarm(system: ChatMessage): Promise<void> {
  if (!engine || !tokenizer || !chatWrap || running) return // need ChatML wrappers; never disturb a live turn
  const job = (async () => {
    try {
      const sysStr = tokenizer!.applyChatTemplate([system], { addGenerationPrompt: false, enableThinking: false }).replace(/\n$/, '')
      const ids = tokenizer!.encode(sysStr, false)
      const t0 = performance.now()
      await engine!.prefill(ids)
      cachedMessages = [system] // the cache now represents exactly [system]; the next clean append reuses it
      console.info(`[aidekin] LLM system prewarm ${(performance.now() - t0).toFixed(0)}ms (${ids.length} tok)`)
    } catch (e) {
      console.warn('[aidekin] LLM prewarm skipped:', (e as Error).message)
    }
  })()
  prewarmPromise = job
  await job
}

// Sampling params for the bitgpu engine's do_sample. Temperature is 0.3 (down from 0.5): this is a
// knowledge assistant, so we bias toward the highest-probability, context-faithful continuation and
// away from the "creative" tail that invents details. topP is accepted but not applied (a no-op) -
// bitgpu's sampler is bit-exact with the transformers.js v4.2.0 reference we validate against, where
// top_p is also disabled.
const SAMPLING = { temperature: 0.3, topK: 20, topP: 0.85, repetitionPenalty: 1.15, noRepeatNgramSize: 3 } as const

/** Coordinator: enqueue this turn as the latest, abort anything running, and drain the queue ONE
 *  generation at a time. */
async function generate(id: number, messages: readonly ChatMessage[], allowThink: boolean, resetCache: boolean, seed?: number, promptLookup?: boolean): Promise<void> {
  queued = { id, messages, allowThink, resetCache, seed, promptLookup } // latest-wins
  if (currentId !== null && currentId >= 0) {
    invalidateCache = true
    abortController?.abort()
  }
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
  if (!engine || !tokenizer) throw new Error('LLM not initialized')
  currentId = id
  invalidateCache = false
  abortController = new AbortController()

  // If a system-prompt prewarm is still in flight, let it finish first: it populates the cache this
  // turn may reuse, and the engine must never run a prefill and a decode concurrently.
  if (prewarmPromise) {
    try { await prewarmPromise } catch { /* prewarm is best-effort */ }
    prewarmPromise = null
  }

  // Reuse the engine's cache only on a clean append (committed conversation + one new user turn), in
  // non-thinking mode (a stripped <think> block isn't reflected in the cached tokens), with ChatML
  // wrappers available. Then feed ONLY the new turn's delta; otherwise rebuild the whole prompt.
  const canReuse = !resetCache && !allowThink && chatWrap !== null && cachedMessages !== null && isCleanAppend(cachedMessages, messages)

  // Diagnostic: when a turn does NOT reuse the cache, log which condition blocked it. Pinpoints why a
  // first turn pays a full prefill despite the load-time prewarm (e.g. resetCache set, a longer-than
  // -expected message list from resumed history, or a system-content mismatch).
  if (!canReuse) {
    console.info(
      `[aidekin] LLM full-prefill (no reuse) · resetCache=${resetCache} think=${allowThink} ` +
        `chatWrap=${chatWrap !== null} cached=${cachedMessages ? cachedMessages.length : 'null'} msgs=${messages.length} ` +
        `cleanAppend=${cachedMessages ? isCleanAppend(cachedMessages, messages) : 'n/a'}`,
    )
  }

  let inputIds: number[]
  if (canReuse) {
    const userText = messages[messages.length - 1].content
    const w = chatWrap as NonNullable<typeof chatWrap>
    const deltaStr = `\n${w.userPrefix}${userText}${w.userSuffix}${w.genPrompt}`
    inputIds = tokenizer.encode(deltaStr, false)
  } else {
    dropCache() // engine.resetCache() so the full prefill starts a fresh sequence
    inputIds = tokenizer.encode(tokenizer.applyChatTemplate(messages as ChatMessage[], { addGenerationPrompt: true, enableThinking: allowThink }), false)
  }

  const think = new ThinkFilter()
  const stream = tokenizer.createDecoderStream(true)
  let full = ''
  let nTokens = 0
  const onToken = (tokenId: number): void => {
    if (currentId !== id) return
    nTokens++
    const text = stream.push(tokenId)
    if (!text) return
    const clean = think.push(text)
    if (clean) {
      full += clean
      post({ kind: 'token', id, text: clean })
    }
  }

  const genOpts = {
    maxTokens: allowThink ? 1024 : 512, // room for the (stripped) <think> block + answer
    ...SAMPLING,
    seed, // undefined in production (entropy); fixed by the behavioral eval for determinism
    promptLookup: promptLookup ?? false,
    stopTokens: [eosTokenId],
    reuseCache: canReuse,
    onToken,
    signal: abortController.signal,
  }
  let msgsUsed = messages
  let result: GenerateResult
  try {
    result = await engine.generate(inputIds, genOpts)
  } catch (err) {
    if (!/maxSeqLen/.test((err as Error).message)) throw err
    // The transcript outgrew the model's KV window (the engine clamps maxTokens, so this only
    // happens when the PROMPT alone no longer fits). Recover instead of bricking the chat: drop
    // the cache, keep the system prompt + the most recent turns, and full-prefill once.
    dropCache()
    msgsUsed = fitToWindow(messages)
    console.warn(`[aidekin] LLM transcript exceeded the ${maxSeqLen}-token window; trimmed ${messages.length - msgsUsed.length} old message(s) and retried`)
    inputIds = tokenizer.encode(tokenizer.applyChatTemplate(msgsUsed as ChatMessage[], { addGenerationPrompt: true, enableThinking: allowThink }), false)
    result = await engine.generate(inputIds, { ...genOpts, reuseCache: false })
  }

  // flush any buffered decode + think tail
  let tail = think.push(stream.flush())
  tail += think.flush()
  if (tail && currentId === id) {
    full += tail
    post({ kind: 'token', id, text: tail })
  }

  // ── cache bookkeeping ──
  if (invalidateCache || abortController.signal.aborted) {
    invalidateCache = false
    dropCache() // barge-in mid-generation -> cache unreliable
  } else if (allowThink) {
    dropCache() // think turns emit reasoning the stored (stripped) reply won't reproduce
  } else if (full.trim()) {
    cachedMessages = [...msgsUsed, { role: 'assistant', content: full }] // engine cache now holds [prompt + reply] (msgsUsed may be a trimmed transcript)
  } else {
    dropCache() // empty reply -> nothing committed; don't risk a stale reuse
  }

  const tps = result.tokensPerSecond
  const nd = Math.max(1, result.tokens.length - 1)
  const per = result.decodeMs / nd
  const tm = result.timing
  const other = Math.max(0, per - tm.gpuMs - tm.recordMs - tm.readbackMs) // CPU sample + onToken decode/stream + writeBuffer
  console.info(
    `[aidekin] LLM(bonsai) done id=${id} ${canReuse ? 'cache-reuse' : 'full-prefill'} tokens=${nTokens} ${tps.toFixed(1)} tok/s (ttft ${result.prefillMs.toFixed(0)}ms) | per-token ${per.toFixed(1)}ms = gpu ${tm.gpuMs.toFixed(1)} + record ${tm.recordMs.toFixed(1)} + readback ${tm.readbackMs.toFixed(1)} + other ${other.toFixed(1)}` +
      (result.speculation ? ` | pld ${result.speculation.accepted}/${result.speculation.drafted} accepted in ${result.speculation.steps} steps` : ''),
  )
  post({ kind: 'done', id, text: full, tps, speculation: result.speculation })
  if (currentId === id) currentId = null
}
