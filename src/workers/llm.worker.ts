/// <reference lib="webworker" />
// LLM worker: the "brain" — PrismML Bonsai (Qwen3-architecture, ternary→ONNX) run on
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

/** True iff `a` is a strict prefix of `b` (used to confirm the KV cache's token
 *  sequence still leads the new prompt before we reuse it). */
function isPrefix(a: readonly number[], b: readonly number[]): boolean {
  if (a.length === 0 || a.length >= b.length) return false
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false
  return true
}

// ── worker state ─────────────────────────────────────────────────────────────
let currentId: number | null = null
let model: PreTrainedModel | null = null
let tokenizer: PreTrainedTokenizer | null = null
let eosTokenId = 151645
let stopper: InterruptableStoppingCriteria | null = null

// ── single-flight generation queue (latest-wins) ──────────────────────────────
// model.generate() must NEVER run twice at once: a second call reassigns the shared
// `stopper`, orphaning the first run so it keeps chewing the GPU forever — N rapid
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
// A persistent DynamicCache so each new turn only prefills the NEW tokens instead
// of the whole transcript (the cause of "replies get slower every message").
//   • kvCache    — the live cache (key/value tensors), kept alive across turns.
//   • cachedIds  — the exact token-id sequence the cache represents.
// DynamicCache can't be cropped, so we reuse ONLY when cachedIds is a verified
// prefix of the new prompt (fast mode); anything else (think/RAG turns, history
// trim, system-prompt change, barge-in abort, a non-round-tripping reply) rebuilds
// from scratch. Always correct, fast on the common multi-turn path.
let kvCache: DynamicCache | null = null
let cachedIds: number[] = []
let invalidateCache = false // set on abort: the partial cache is unusable

function disposeCache(): void {
  if (kvCache) void kvCache.dispose()
  kvCache = null
  cachedIds = []
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
  model = await withRetry(
    () =>
      AutoModelForCausalLM.from_pretrained(id, {
        dtype: dtype as never,
        device: device as never,
        progress_callback: onProgress as never,
      }),
    { onRetry },
  )
  post({ kind: 'ready', info: `transformers.js ${id} (${dtype}·${device})` })
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
  // run may take a moment to unwind — but it will, and no two runs overlap on the GPU.
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

  // Toggle reasoning via the chat template's `enable_thinking` flag (a structural switch
  // the template understands), NOT by appending "/no_think" to the user text — a small
  // model can quote that literal string back in its reply (the visible-"/no_think" bug).
  // This keeps user content clean AND prefix-stable across turns for the KV cache below.
  const tplOpts = { add_generation_prompt: true, return_dict: true, enable_thinking: allowThink }
  const inputs = tokenizer.apply_chat_template(
    messages as unknown as Array<{ role: string; content: string }>,
    tplOpts as never,
  ) as { input_ids: Tensor } & Record<string, unknown>
  const fullIds = (inputs.input_ids.tolist() as Array<Array<number | bigint>>)[0].map((x) => Number(x))

  // Reuse the cache only if its tokens are a genuine prefix of this prompt (fast
  // mode, not signalled dirty). Then prefill ONLY the new tail; otherwise rebuild.
  const canReuse =
    !resetCache && !allowThink && kvCache !== null && isPrefix(cachedIds, fullIds)

  let modelInputs: Record<string, unknown>
  if (canReuse) {
    const deltaIds = fullIds.slice(cachedIds.length)
    const input_ids = new Tensor('int64', BigInt64Array.from(deltaIds.map(BigInt)), [1, deltaIds.length])
    modelInputs = { input_ids, past_key_values: kvCache } // no attention_mask → ones(past+delta)
  } else {
    disposeCache()
    kvCache = new DynamicCache()
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

  // generate() returns the full token sequence it processed (delta when reusing, or
  // the whole prompt when fresh) PLUS the generated tokens, and — because we pass
  // past_key_values — leaves the cache alive for next turn.
  const out = (await model.generate({
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
  } as never)) as unknown as Tensor

  const tail = think.flush()
  if (tail) {
    full += tail
    post({ kind: 'token', id, text: tail })
  }

  // ── cache bookkeeping ──
  if (invalidateCache) {
    invalidateCache = false
    disposeCache() // barge-in interrupted mid-generation → cache prefix is unreliable
  } else if (allowThink) {
    // think/RAG turns generate reasoning + reference text that the stored plain
    // history won't reproduce next turn — drop the cache so the next turn rebuilds.
    disposeCache()
  } else {
    try {
      // out = the tokens we fed (delta when reusing, full prompt when fresh) + the
      // generated reply. Either way cachedIds becomes fullIds ++ generated, exactly
      // what the cache now holds.
      const rows = out.tolist() as Array<Array<number | bigint>>
      const ids = rows[0].map((x) => Number(x))
      cachedIds = canReuse ? cachedIds.concat(ids) : ids
    } catch {
      disposeCache() // couldn't read the sequence → don't risk a stale reuse
    }
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
