// Browser embedder (bge-small-en-v1.5) on onnxruntime-web (single-thread WASM, so it never competes
// with the LLM for the GPU and works in non-cross-origin-isolated iframes). Replaces the transformers.js
// feature-extraction pipeline; byte-exact with it (scripts/verify-embedder.ts). Shared tokenize + pool
// logic lives in embedderCore.ts so the browser query, the browser builder, and the Node CLI all emit
// identical vectors. Dynamically imported, so onnxruntime-web only loads when RAG is actually used.
import * as ort from 'onnxruntime-web/wasm'
import { Tokenizer } from '@huggingface/tokenizers'
import { EMBED, ORT_WASM_CDN, embedModelUrls } from '../models/registry'
import { getModelAsset } from '../core/modelStore'
import { withRetry } from '../core/retry'
import { QUERY_INSTRUCTION, poolAndNormalize, tokenizeBatch } from './embedderCore'

export type EmbedProgress = (p: { status?: string; file?: string; loaded?: number; total?: number; progress?: number }) => void

ort.env.wasm.wasmPaths = ORT_WASM_CDN
ort.env.wasm.numThreads = 1 // single-thread: keep the GPU free for the LLM, no SAB/COI requirement

interface Embedder {
  tok: Tokenizer
  session: ort.InferenceSession
}
let embedderPromise: Promise<Embedder> | null = null

async function load(onProgress?: EmbedProgress): Promise<Embedder> {
  if (!embedderPromise) {
    embedderPromise = withRetry(
      async () => {
        const urls = embedModelUrls()
        const [tokJson, tokCfg] = await Promise.all([
          fetch(urls.tokenizerJsonUrl).then((r) => r.json()),
          fetch(urls.tokenizerConfigUrl).then((r) => r.json()),
        ])
        const tok = new Tokenizer(tokJson, tokCfg)
        const onnx = await getModelAsset('embed-bge-small-q8', urls.onnxUrl, (p) =>
          onProgress?.({ file: 'embedder', loaded: p.loaded, total: p.total, progress: p.total ? (100 * p.loaded) / p.total : 0 }),
        )
        const session = await ort.InferenceSession.create(onnx)
        return { tok, session }
      },
      { onRetry: (n, _e, ms) => console.warn(`[aidekin] embedder load failed (transient); retry ${n} in ${ms}ms`) },
    )
    embedderPromise.catch(() => {
      embedderPromise = null // let the next call retry instead of caching the rejection forever
    })
  }
  return embedderPromise
}

export function loadEmbedder(onProgress?: EmbedProgress): Promise<unknown> {
  return load(onProgress)
}

const big = (rows: number[][]): BigInt64Array => {
  const seq = rows[0]?.length ?? 0
  const out = new BigInt64Array(rows.length * seq)
  for (let b = 0; b < rows.length; b++) for (let t = 0; t < seq; t++) out[b * seq + t] = BigInt(rows[b][t])
  return out
}

/** Embed a batch in one forward pass -> one unit-normalized 384-dim vector per input. */
export async function embedMany(texts: string[], onProgress?: EmbedProgress): Promise<Float32Array[]> {
  if (texts.length === 0) return []
  const { tok, session } = await load(onProgress)
  const { ids, masks, batch, seq } = tokenizeBatch(tok, texts)
  const dims: [number, number] = [batch, seq]
  const feeds: Record<string, ort.Tensor> = {
    input_ids: new ort.Tensor('int64', big(ids), dims),
    attention_mask: new ort.Tensor('int64', big(masks), dims),
  }
  if (session.inputNames.includes('token_type_ids')) {
    feeds.token_type_ids = new ort.Tensor('int64', new BigInt64Array(batch * seq), dims)
  }
  const res = await session.run(feeds)
  const outName = session.outputNames.find((n) => /hidden|last|output/i.test(n)) ?? session.outputNames[0]
  const hidden = res[outName].data as Float32Array
  return poolAndNormalize(hidden, masks, batch, seq, EMBED.dim)
}

/** Embed a single string to a unit-normalized 384-dim vector. */
export async function embedOne(text: string, onProgress?: EmbedProgress): Promise<Float32Array> {
  return (await embedMany([text], onProgress))[0]
}

/** Embed a search QUERY (with the bge retrieval instruction). Use this, not embedOne, for runtime
 *  retrieval against an index built from raw-passage vectors (embedMany). */
export async function embedQuery(text: string, onProgress?: EmbedProgress): Promise<Float32Array> {
  return embedOne(QUERY_INSTRUCTION + text, onProgress)
}
