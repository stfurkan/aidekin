// Typed message protocol between the main-thread orchestrator and the model
// workers. Each worker has an `*In` (commands it receives) and `*Out` (events it
// emits) discriminated union, keyed on `kind`. Keep payloads transferable
// (Float32Array buffers are transferred, not copied). English-only.

// ── shared lifecycle events (every worker emits these) ───────────────────────
export interface LoadProgress {
  readonly kind: 'load'
  readonly label: string
  /** Human-readable detail: a filename + size, or a stage description. */
  readonly detail: string
  /** The file this progress refers to (lets consumers aggregate multi-file downloads). */
  readonly file?: string
  /** Bytes loaded / total (downloads), or a scaled fraction (loaded/total = progress). */
  readonly loaded: number
  readonly total: number
}
export interface ReadyEvent {
  readonly kind: 'ready'
  readonly info?: string
}
export interface WorkerErrorEvent {
  readonly kind: 'error'
  readonly message: string
  /** Generation id when the error belongs to a specific turn - a superseded turn's error
   *  must not settle the CURRENT turn. Absent for load/lifecycle errors. */
  readonly id?: number
}
type Lifecycle = LoadProgress | ReadyEvent | WorkerErrorEvent

export type Device = 'webgpu' | 'wasm'

/** A compact summary of the model's per-token certainty for one answer, from bitgpu's true logprobs.
 *  `meanProb` is the geometric-mean token probability (exp of the mean logprob), a single 0..1
 *  confidence; `lowConfFrac` is the fraction of tokens the model emitted below p=0.3 (where it was
 *  effectively guessing). Both fall as an answer gets shakier. */
export interface LlmConfidence {
  readonly meanProb: number
  readonly lowConfFrac: number
  readonly tokens: number
}

// ── VAD worker (Silero gate) ─────────────────────────────────────────────────
export type VadIn =
  | { readonly kind: 'init'; readonly assetBase: string }
  | { readonly kind: 'frame'; readonly samples: Float32Array } // 512 samples @ 16 kHz
  | { readonly kind: 'reset' }
export type VadOut =
  | Lifecycle
  | { readonly kind: 'speech-start' }
  | { readonly kind: 'speech-end'; readonly durationMs: number; readonly audio: Float32Array }
  | { readonly kind: 'misfire' }

// ── ASR worker (Nemotron streaming) ──────────────────────────────────────────
// Live streaming: `reset` at turn start, `chunk` per mic frame as the user speaks
// (the worker buffers to one 560 ms encoder chunk and emits partials), `flush` at
// turn end for the final. `id` is the turn id so stale results (after a barge-in)
// are dropped. All messages are serialized on the worker's promise chain.
export type AsrIn =
  | { readonly kind: 'init'; readonly modelBase: string; readonly device: Device; readonly debug?: boolean }
  // Download all weights to the OPFS cache without creating sessions (parallel-download phase).
  | { readonly kind: 'prefetch'; readonly modelBase: string }
  | { readonly kind: 'chunk'; readonly id: number; readonly samples: Float32Array }
  | { readonly kind: 'flush'; readonly id: number }
  | { readonly kind: 'reset' }
export type AsrOut =
  | Lifecycle
  | { readonly kind: 'prefetched' } // all weights are cached; ready to init from disk
  | { readonly kind: 'partial'; readonly id?: number; readonly text: string } // running transcript fragment
  | { readonly kind: 'final'; readonly id?: number; readonly text: string }

// ── LLM worker (Bonsai via bitgpu) ──────────────────────────────
export interface ChatMessage {
  readonly role: 'system' | 'user' | 'assistant'
  readonly content: string
}
export type LlmIn =
  | {
      readonly kind: 'init'
      readonly manifestUrl: string //     engine manifest.json (manifest-format model)
      readonly dataUrl: string //         the ~237MB weights data file (the Bonsai GGUF)
      readonly auxUrl: string //          the small aux file (LUTs)
      readonly tokenizerModelId: string // HF repo for tokenizer.json + tokenizer_config.json
      readonly eosTokenId?: number //     stop token (default 151645)
      readonly maxSeqLen?: number //      KV-cache length cap (default 2048)
      readonly kvCache?: 'f32' | 'f16' | 'q8' // KV storage precision ('f16' halves KV memory and needs shader-f16; 'q8' quarters it on every adapter; f32 default)
      readonly overflow?: 'error' | 'sinks' // window-overflow policy: 'error' throws (default), 'sinks' rolls the window (unbounded chat in fixed memory)
      // Namespace for cross-reload KV-cache snapshots (chat.save/restore). Set = persistence on
      // (snapshots stored in IndexedDB, keyed by this + model + kvCache mode); unset = no snapshot
      // persistence. Mirrors the orchestrator's localStorage persistKey so both gate together.
      readonly persistSession?: string
      readonly debug?: boolean //         enable info-level worker logs (see core/log.ts)
    }
  | {
      readonly kind: 'generate'
      readonly id: number
      readonly messages: readonly ChatMessage[]
      /** Let the model reason internally (skip the /no_think soft-switch). The <think>
       *  block is still stripped from the visible output. Used for RAG-grounded turns. */
      readonly think?: boolean
      /** Fixed sampler seed for DETERMINISTIC replies (the behavioral eval). Unset in
       *  production: the worker seeds from entropy. */
      readonly seed?: number
      /** Enable prompt-lookup speculative decoding for this turn (bitgpu experimental). */
      readonly promptLookup?: boolean
      /** Force the worker to discard its cross-turn KV cache and re-prefill the whole
       *  transcript - set when the history prefix changed non-append (new session,
       *  cleared chat, system-prompt change, or a sliding-window trim). */
      readonly resetCache?: boolean
      /** Opt-in per turn: ask the engine for per-token logprobs and return a `confidence` summary
       *  on `done`. Costs a few % (disables promptLookup, routes through the sampler path), so it is
       *  OFF by default and never set on hot paths unless a caller wants the signal. */
      readonly wantConfidence?: boolean
    }
  | { readonly kind: 'abort'; readonly id: number }
  // Start a fresh conversation: forget the chat's committed transcript + KV cache (chat.reset) and
  // delete the persisted snapshot so a reload after "new chat" does NOT restore the old conversation.
  | { readonly kind: 'reset-session' }
  // Prewarm the KV cache with the (static) system prompt at load, so the user's FIRST turn is a
  // cache-append instead of a cold full prefill. Best-effort; the worker runs it when idle.
  | {
      readonly kind: 'prewarm'
      readonly system: ChatMessage
      /** Full transcript (model view, system first) to warm instead of just the system prompt -
       *  set when persisted history was restored, so a returning visitor's first turn is a
       *  cache-append rather than a full prefill of the whole restored conversation. */
      readonly messages?: readonly ChatMessage[]
    }
export type LlmOut =
  | Lifecycle
  | { readonly kind: 'token'; readonly id: number; readonly text: string }
  | {
      readonly kind: 'done'
      readonly id: number
      readonly text: string
      readonly tps?: number
      /** True when this turn extended the KV cache (clean append) instead of a full prefill - the
       *  signal that cross-turn reuse (or a restored session snapshot) is working. */
      readonly reusedCache?: boolean
      /** Present when prompt-lookup decoding ran: verify steps, drafted and accepted counts. */
      readonly speculation?: { steps: number; drafted: number; accepted: number }
      /** Present only when the turn set `wantConfidence`: a summary of the model's per-token
       *  certainty for the emitted answer (derived from bitgpu's true logprobs). */
      readonly confidence?: LlmConfidence
    }

// ── TTS worker (Supertonic) ──────────────────────────────────────────────────
export type TtsIn =
  | { readonly kind: 'init'; readonly modelBase: string; readonly device: Device; readonly debug?: boolean }
  // Download all weights to the OPFS cache without creating sessions (parallel-download phase).
  | { readonly kind: 'prefetch'; readonly modelBase: string }
  | { readonly kind: 'speak'; readonly id: number; readonly text: string }
  | { readonly kind: 'abort'; readonly id: number }
export type TtsOut =
  | Lifecycle
  | { readonly kind: 'prefetched' } // all weights are cached; ready to init from disk
  | { readonly kind: 'audio'; readonly id: number; readonly pcm: Float32Array; readonly sampleRate: number }
  | { readonly kind: 'done'; readonly id: number }

// ── Smart Turn v3 worker ─────────────────────────────────────────────────────
export type TurnIn =
  | { readonly kind: 'init'; readonly modelBase: string; readonly debug?: boolean }
  | { readonly kind: 'analyze'; readonly id: number; readonly samples: Float32Array }
export type TurnOut =
  | Lifecycle
  | { readonly kind: 'verdict'; readonly id: number; readonly complete: boolean; readonly prob: number }
