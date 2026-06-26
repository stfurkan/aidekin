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
}
type Lifecycle = LoadProgress | ReadyEvent | WorkerErrorEvent

export type Device = 'webgpu' | 'wasm'

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
  | { readonly kind: 'init'; readonly modelBase: string; readonly device: Device }
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

// ── LLM worker (Bonsai via transformers.js) ──────────────────────────────────
export interface ChatMessage {
  readonly role: 'system' | 'user' | 'assistant'
  readonly content: string
}
export type LlmIn =
  | {
      readonly kind: 'init'
      readonly model: string //       transformers.js HF repo id
      readonly dtype?: string //      transformers.js dtype (e.g. 'q1')
      readonly device?: Device //     transformers.js device
      readonly eosTokenId?: number // transformers.js stop token
    }
  | {
      readonly kind: 'generate'
      readonly id: number
      readonly messages: readonly ChatMessage[]
      /** Let the model reason internally (skip the /no_think soft-switch). The <think>
       *  block is still stripped from the visible output. Used for RAG-grounded turns. */
      readonly think?: boolean
      /** Force the worker to discard its cross-turn KV cache and re-prefill the whole
       *  transcript - set when the history prefix changed non-append (new session,
       *  cleared chat, system-prompt change, or a sliding-window trim). */
      readonly resetCache?: boolean
    }
  | { readonly kind: 'abort'; readonly id: number }
export type LlmOut =
  | Lifecycle
  | { readonly kind: 'token'; readonly id: number; readonly text: string }
  | { readonly kind: 'done'; readonly id: number; readonly text: string; readonly tps?: number }

// ── TTS worker (Supertonic) ──────────────────────────────────────────────────
export type TtsIn =
  | { readonly kind: 'init'; readonly modelBase: string; readonly device: Device }
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
  | { readonly kind: 'init'; readonly modelBase: string }
  | { readonly kind: 'analyze'; readonly id: number; readonly samples: Float32Array }
export type TurnOut =
  | Lifecycle
  | { readonly kind: 'verdict'; readonly id: number; readonly complete: boolean; readonly prob: number }
