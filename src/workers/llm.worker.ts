/// <reference lib="webworker" />
// LLM worker: the "brain" - PrismML Bonsai (Qwen3-architecture, ternary→ONNX) run on
// WebGPU via @huggingface/transformers. Streams tokens, strips Qwen3 <think> blocks,
// toggles reasoning via the chat template's enable_thinking flag, reuses a cross-turn
// KV cache (see below), and logs tokens/sec.

import {
  AutoModelForCausalLM,
  AutoTokenizer,
  TextStreamer,
  InterruptableStoppingCriteria,
  DynamicCache,
  Tensor,
  env,
  type PreTrainedModel,
  type PreTrainedTokenizer,
} from '@huggingface/transformers'
import type { ChatMessage, LlmIn, LlmOut } from '../protocol/messages'
import { installOpfsModelCache } from '../core/opfsModelCache'
import { withRetry } from '../core/retry'

const ctx = self as unknown as DedicatedWorkerGlobalScope
const post = (m: LlmOut): void => ctx.postMessage(m)

env.allowRemoteModels = true // stream model files from the HF Hub
// ORT wasm: leave transformers.js's default CDN (jsDelivr, pinned to the exact immutable
// onnxruntime-web version it bundles). That's the supported out-of-the-box path; the JS glue
// and wasm MUST be the same build, so we can NOT substitute our root onnxruntime-web here. The
// same jsDelivr origin already serves the speech workers' wasm under COEP in production.
// Cache the ~290 MB model in OPFS, not Cache Storage (which errors on an entry this large, so
// the weights would otherwise re-download every visit). Best-effort: see opfsModelCache.ts.
installOpfsModelCache(env)

// ── Qwen3 <think> stripping ──────────────────────────────────────────────────
// Bonsai keeps Qwen3's chat template and can emit <think>…</think> blocks we must
// never show or speak. This strips them from the token STREAM (tags can straddle token
// boundaries), holding back only a possible partial tag at each chunk's edge.
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

/** True iff `next` is exactly `cached` plus one new trailing user turn - i.e. a clean
 *  append, so we can extend the KV cache with just that turn instead of re-prefilling.
 *  (We can't compare re-tokenized prompts: Bonsai's template renders an assistant turn
 *  WITH an empty <think> block when it's last but STRIPS it once a newer turn follows,
 *  so a re-tokenized history is never a token-prefix of the cached sequence. We track the
 *  committed MESSAGES instead and append the new turn's delta tokens - see runGeneration.) */
function isCleanAppend(cached: readonly ChatMessage[] | null, next: readonly ChatMessage[]): boolean {
  if (!cached || next.length !== cached.length + 1) return false
  if (next[next.length - 1].role !== 'user') return false
  for (let i = 0; i < cached.length; i++) {
    if (next[i].role !== cached[i].role || next[i].content !== cached[i].content) return false
  }
  return true
}

/** Chat-template wrappers, derived from the tokenizer at init so the cache-append never
 *  hardcodes a template. `genPrompt` is what add_generation_prompt appends; `userPrefix`/
 *  `userSuffix` wrap a user turn's content. null when the model isn't standard ChatML, in
 *  which case cache-append is disabled (we fall back to full-prefill - correct, just slower). */
let chatWrap: { genPrompt: string; userPrefix: string; userSuffix: string } | null = null

function deriveChatWrap(tk: PreTrainedTokenizer): typeof chatWrap {
  try {
    const render = (msgs: Array<{ role: string; content: string }>, agp: boolean): string =>
      tk.apply_chat_template(msgs, { add_generation_prompt: agp, tokenize: false, enable_thinking: false } as never) as unknown as string
    const SENT = 'SENT'
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
let model: PreTrainedModel | null = null
let tokenizer: PreTrainedTokenizer | null = null
let eosTokenId = 151645
let stopper: InterruptableStoppingCriteria | null = null

// ── single-flight generation queue (latest-wins) ──────────────────────────────
// model.generate() must NEVER run twice at once: a second call reassigns the shared
// `stopper`, orphaning the first run so it keeps chewing the GPU forever - N rapid
// turns then stack N zombie generations that split the GPU N ways (the "stuck for
// minutes, stops responding" spiral). So we serialize: a new request interrupts the
// running one and is stashed as `queued`; the in-flight loop picks up only the LATEST
// queued request once the current run fully unwinds. One generation on the GPU at a time.
interface GenJob {
  id: number
  messages: readonly ChatMessage[]
  allowThink: boolean
  resetCache: boolean
}
let running = false
let queued: GenJob | null = null

// ── cross-turn KV cache ───────────────────────────────────────────────────────
// A persistent DynamicCache so each new turn only prefills the NEW turn instead of the
// whole transcript (the cause of "replies get slower every message").
//   • kvCache         - the live cache (key/value tensors), kept alive across turns.
//   • cachedMessages  - the committed conversation (incl. assistant replies) the cache
//                       physically represents. The cache is reused ONLY when the next
//                       request is exactly this + one new user turn (isCleanAppend); then
//                       we feed just that turn's delta tokens. Anything else (think turn,
//                       history trim, system-prompt change, barge-in abort) rebuilds.
// We track MESSAGES, not token ids, because Bonsai's template renders past assistant turns
// differently from the live one (empty <think> block), so a re-tokenized history is never a
// token-prefix of what's cached. Appending the delta to the physical cache sidesteps that.
let kvCache: DynamicCache | null = null
let cachedMessages: ChatMessage[] | null = null
let invalidateCache = false // set on abort: the partial cache is unusable

function disposeCache(): void {
  if (kvCache) void kvCache.dispose()
  kvCache = null
  cachedMessages = null
}

ctx.onmessage = (ev: MessageEvent<LlmIn>) => {
  void handle(ev.data)
}

async function handle(msg: LlmIn): Promise<void> {
  try {
    if (msg.kind === 'init') {
      await init(msg.model, msg.dtype ?? 'q1', msg.device ?? 'webgpu', msg.eosTokenId ?? 151645)
    } else if (msg.kind === 'generate') {
      await generate(msg.id, msg.messages, msg.think ?? false, msg.resetCache ?? false)
    } else if (msg.kind === 'abort') {
      if (msg.id === currentId) {
        invalidateCache = true // partial generation → cache no longer matches a clean prefix
        stopper?.interrupt()
      }
    }
  } catch (err) {
    post({ kind: 'error', message: `LLM: ${(err as Error).message}` })
  }
}

async function init(id: string, dtype: string, device: string, eos: number): Promise<void> {
  eosTokenId = eos
  disposeCache() // a fresh model means any prior KV cache is meaningless
  const onProgress = (p: { status?: string; file?: string; progress?: number; loaded?: number; total?: number }): void => {
    if (p.loaded != null && p.total) {
      post({ kind: 'load', label: 'LLM', file: p.file, detail: `${p.file ?? 'model'} · ${Math.round(p.progress ?? 0)}%`, loaded: p.loaded, total: p.total })
    } else if (p.status) {
      post({ kind: 'load', label: 'LLM', file: p.file, detail: `${p.status} ${p.file ?? ''}`.trim(), loaded: 0, total: 0 })
    }
  }
  // Retry transient CDN resets / rate limits while downloading the weights.
  const onRetry = (n: number, _e: unknown, ms: number): void =>
    console.warn(`[aidekin] LLM load failed (transient); retry ${n} in ${ms}ms`)
  tokenizer = await withRetry(
    () => AutoTokenizer.from_pretrained(id, { progress_callback: onProgress as never }),
    { onRetry },
  )
  // Derive the ChatML wrappers now so cross-turn cache-append can extend the KV cache with
  // just the new turn. null (non-ChatML template) → cache-append disabled, full-prefill each turn.
  chatWrap = deriveChatWrap(tokenizer)
  if (!chatWrap) console.warn('[aidekin] LLM: non-ChatML template - cross-turn KV cache disabled')
  model = await withRetry(
    () =>
      AutoModelForCausalLM.from_pretrained(id, {
        dtype: dtype as never,
        device: device as never,
        progress_callback: onProgress as never,
      }),
    { onRetry },
  )
  await warmup()
  post({ kind: 'ready', info: `transformers.js ${id} (${dtype}·${device})` })
}

/** Compile the WebGPU prefill/decode shaders at load time so the user's FIRST message isn't a
 *  ~10s cold start (the ASR/TTS workers warm up for the same reason). Throwaway: a SEPARATE
 *  cache, one token, and NO module state written - kvCache/cachedMessages stay null, so the
 *  first real turn still does a clean prefill and nothing from the warmup leaks into the
 *  conversation. Mirrors the real sampling params so the whole generation path is compiled.
 *  Best-effort: a warmup hiccup never blocks readiness. */
async function warmup(): Promise<void> {
  if (!model || !tokenizer) return
  const throwaway = new DynamicCache()
  try {
    const t0 = performance.now()
    const inputs = tokenizer.apply_chat_template(
      [{ role: 'user', content: 'Hi' }],
      { add_generation_prompt: true, return_dict: true, enable_thinking: false } as never,
    ) as Record<string, unknown>
    await model.generate({
      ...inputs,
      past_key_values: throwaway,
      max_new_tokens: 1,
      do_sample: true,
      temperature: 0.5,
      top_k: 20,
      top_p: 0.85,
      repetition_penalty: 1.15,
      no_repeat_ngram_size: 3,
      eos_token_id: eosTokenId,
    } as never)
    console.info(`[aidekin] LLM warmup ${(performance.now() - t0).toFixed(0)}ms`)
  } catch (e) {
    console.warn('[aidekin] LLM warmup skipped:', (e as Error).message)
  } finally {
    void throwaway.dispose()
  }
}

/** Coordinator: enqueue this turn as the latest, interrupt anything running, and drain
 *  the queue ONE generation at a time (see the single-flight note above). */
async function generate(
  id: number,
  messages: readonly ChatMessage[],
  allowThink: boolean,
  resetCache: boolean,
): Promise<void> {
  queued = { id, messages, allowThink, resetCache } // latest-wins
  // Stop whatever is mid-flight so the loop can advance to this newest request. A prefill
  // can't be interrupted (the stopper is only checked between decode steps), so the current
  // run may take a moment to unwind - but it will, and no two runs overlap on the GPU.
  if (currentId !== null && currentId >= 0) {
    invalidateCache = true
    stopper?.interrupt()
  }
  if (running) return
  running = true
  try {
    while (queued) {
      const job = queued
      queued = null
      try {
        await runGeneration(job.id, job.messages, job.allowThink, job.resetCache)
      } catch (err) {
        post({ kind: 'error', message: `LLM: ${(err as Error).message}` })
      }
    }
  } finally {
    running = false
  }
}

async function runGeneration(
  id: number,
  messages: readonly ChatMessage[],
  allowThink: boolean,
  resetCache: boolean,
): Promise<void> {
  if (!model || !tokenizer) throw new Error('LLM not initialized')
  currentId = id
  invalidateCache = false

  // Reuse the cache only on a clean append (committed conversation + one new user turn),
  // and only in non-thinking mode (a stripped <think> block wouldn't be reflected in the
  // cached tokens) with the ChatML wrappers available. Then feed ONLY the new turn's delta;
  // otherwise rebuild the whole prompt.
  const canReuse =
    !resetCache && !allowThink && chatWrap !== null && kvCache !== null && isCleanAppend(cachedMessages, messages)

  // Diagnostic: when we HAVE a cache but fall back to a full prefill (the slow ~13s path),
  // log WHY. Cross-turn reuse is what keeps ttft low (prefill only the new turn's delta, not
  // the whole transcript), so a silent fallback to full-prefill is the latency bug to hunt.
  if (kvCache !== null && !canReuse) {
    let why: string
    if (resetCache) why = 'resetCache flag'
    else if (allowThink) why = 'thinking mode'
    else if (chatWrap === null) why = 'no ChatML wrap'
    else {
      why = 'prefix changed'
      const c = cachedMessages
      if (c && messages.length !== c.length + 1) why += ` (length cached=${c.length} next=${messages.length})`
      else if (c) {
        for (let i = 0; i < c.length; i++) {
          if (c[i].role !== messages[i].role || c[i].content !== messages[i].content) {
            why += ` @${i} ${c[i].role}: cached="${c[i].content.slice(0, 30)}" next="${messages[i].content.slice(0, 30)}"`
            break
          }
        }
      }
    }
    console.warn(`[aidekin] LLM full-prefill, cache not reused: ${why}`)
  }

  let modelInputs: Record<string, unknown>
  if (canReuse) {
    // Append the new user turn + generation prompt directly to the live cache. We build the
    // delta from the new turn alone (not by re-tokenizing the history), so the empty-<think>
    // mismatch never matters - the physical cache stays a self-consistent ChatML transcript.
    const userText = messages[messages.length - 1].content
    const w = chatWrap as NonNullable<typeof chatWrap>
    const deltaStr = `\n${w.userPrefix}${userText}${w.userSuffix}${w.genPrompt}`
    const deltaIds = (tokenizer.encode(deltaStr, { add_special_tokens: false } as never) as number[]).map(Number)
    const input_ids = new Tensor('int64', BigInt64Array.from(deltaIds.map((x) => BigInt(x))), [1, deltaIds.length])
    modelInputs = { input_ids, past_key_values: kvCache } // no attention_mask → ones(past+delta)
  } else {
    disposeCache()
    kvCache = new DynamicCache()
    // Toggle reasoning via the template's `enable_thinking` flag (a structural switch the
    // template understands), NOT by appending "/no_think" to the user text - a small model
    // can quote that literal string back in its reply (the visible-"/no_think" bug).
    const tplOpts = { add_generation_prompt: true, return_dict: true, enable_thinking: allowThink }
    const inputs = tokenizer.apply_chat_template(
      messages as unknown as Array<{ role: string; content: string }>,
      tplOpts as never,
    ) as { input_ids: Tensor } & Record<string, unknown>
    modelInputs = { ...inputs, past_key_values: kvCache }
  }

  const think = new ThinkFilter()
  let full = ''
  let nTokens = 0
  const t0 = performance.now()
  let tFirst = 0
  stopper = new InterruptableStoppingCriteria()

  const streamer = new TextStreamer(tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    token_callback_function: (() => {
      if (!tFirst) tFirst = performance.now()
      nTokens++
    }) as never,
    callback_function: ((text: string) => {
      if (currentId !== id) return
      const clean = think.push(text)
      if (clean) {
        full += clean
        post({ kind: 'token', id, text: clean })
      }
    }) as never,
  })

  // We pass past_key_values, so generate() leaves the cache alive (extended by this turn's
  // tokens) for the next turn. The returned sequence isn't needed - we track the committed
  // MESSAGES, not token ids (see the cache note above).
  await model.generate({
    ...modelInputs,
    max_new_tokens: allowThink ? 1024 : 512, // room for the (stripped) <think> block + answer
    do_sample: true,
    temperature: 0.5,
    top_k: 20,
    top_p: 0.85,
    // Small ternary models degenerate into phrase loops on out-of-distribution input. 1.05 was
    // too weak; bump the penalty and hard-block any repeated 3-gram so a loop is impossible.
    repetition_penalty: 1.15,
    no_repeat_ngram_size: 3,
    eos_token_id: eosTokenId,
    streamer,
    stopping_criteria: stopper,
  } as never)

  const tail = think.flush()
  if (tail) {
    full += tail
    post({ kind: 'token', id, text: tail })
  }

  // ── cache bookkeeping ──
  if (invalidateCache) {
    invalidateCache = false
    disposeCache() // barge-in interrupted mid-generation → cache is unreliable
  } else if (allowThink) {
    // think turns emit reasoning the stored (stripped) reply won't reproduce - drop the cache.
    disposeCache()
  } else if (full.trim()) {
    // The KV cache now physically holds [prompt(this turn) + reply]. Record the committed
    // conversation it represents so the NEXT clean append can extend it.
    cachedMessages = [...messages, { role: 'assistant', content: full }]
  } else {
    disposeCache() // empty reply → nothing committed; don't risk a stale reuse
  }

  const tps = tps_(nTokens, tFirst, t0)
  console.info(
    `[aidekin] LLM(bonsai) done id=${id} ${canReuse ? 'cache-reuse' : 'full-prefill'} tokens=${nTokens} ${tps.toFixed(1)} tok/s (ttft ${(tFirst - t0).toFixed(0)}ms, visible=${full.length})`,
  )
  post({ kind: 'done', id, text: full, tps })
  if (currentId === id) currentId = null
}

/** tokens / decode-seconds (decode excludes the time-to-first-token / prefill). */
function tps_(nTokens: number, tFirst: number, t0: number): number {
  const decodeMs = performance.now() - (tFirst || t0)
  return nTokens > 1 && decodeMs > 0 ? (nTokens - 1) / (decodeMs / 1000) : 0
}
