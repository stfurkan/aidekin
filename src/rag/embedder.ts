// Embedding model (bge-small-en-v1.5, see registry EMBED) via transformers.js. Shared by the browser
// (query + builder) and the Node CLI — using the SAME model + dtype is what keeps the
// query vectors compatible with the precomputed index. WASM backend in the browser so
// it never competes with the LLM for the GPU. Dynamically imported, so transformers
// only loads when RAG is actually used.

import { pipeline, env, type FeatureExtractionPipeline } from '@huggingface/transformers'
import { EMBED } from '../models/registry'
import { withRetry } from '../core/retry'

env.allowRemoteModels = true
// ORT wasm: use transformers.js's default CDN (jsDelivr, pinned to the exact bundled
// onnxruntime-web build). The JS glue and wasm must match, so we can't point this at our root
// onnxruntime-web; the default is the supported path and is already proven under COEP in prod.

export type EmbedProgress = (p: {
  status?: string
  file?: string
  loaded?: number
  total?: number
  progress?: number
}) => void

let pipePromise: Promise<FeatureExtractionPipeline> | null = null

export function loadEmbedder(onProgress?: EmbedProgress): Promise<FeatureExtractionPipeline> {
  if (!pipePromise) {
    const isBrowser = typeof window !== 'undefined'
    pipePromise = withRetry(
      () =>
        pipeline('feature-extraction', EMBED.hfModelId, {
          dtype: EMBED.dtype as never,
          // Browser: pin WASM (CPU) so the GPU stays free for the LLM. Node: let
          // transformers pick onnxruntime-node.
          ...(isBrowser ? { device: 'wasm' as never } : {}),
          progress_callback: onProgress as never,
        }) as Promise<FeatureExtractionPipeline>,
      {
        onRetry: (n, _e, ms) =>
          console.warn(`[aidekin] embedder load failed (transient); retry ${n} in ${ms}ms`),
      },
    )
    // If it still fails after the retries (network / wasm / OOM), clear the cached promise so the
    // next call retries instead of re-throwing the stale rejection forever (which would silently
    // kill RAG for the whole session). The current caller still sees this rejection.
    pipePromise.catch(() => {
      pipePromise = null
    })
  }
  return pipePromise
}

const opts = { pooling: EMBED.pooling as 'mean', normalize: EMBED.normalize }

/** Embed a single string → a unit-normalized 384-dim vector. */
export async function embedOne(text: string, onProgress?: EmbedProgress): Promise<Float32Array> {
  const ext = await loadEmbedder(onProgress)
  const out = await ext(text, opts)
  return Float32Array.from(out.data as Float32Array)
}

/** Embed a batch in one forward pass → one vector per input. */
export async function embedMany(texts: string[], onProgress?: EmbedProgress): Promise<Float32Array[]> {
  if (texts.length === 0) return []
  const ext = await loadEmbedder(onProgress)
  const out = await ext(texts, opts)
  const dim = EMBED.dim
  const data = out.data as Float32Array
  const result: Float32Array[] = []
  for (let i = 0; i < texts.length; i++) {
    result.push(Float32Array.from(data.subarray(i * dim, (i + 1) * dim)))
  }
  return result
}
