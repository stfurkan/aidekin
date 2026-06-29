// Environment-agnostic core of the embedder (no ONNX runtime, no DOM). The browser path
// (embedder.ts, onnxruntime-web) and the Node CLI path (embedderNode.ts, onnxruntime-node) share this:
// tokenize via @huggingface/tokenizers (the canonical WordPiece, byte-exact with transformers.js,
// verified in scripts/verify-embedder.ts), then a masked mean-pool + L2 normalize over the model's
// last_hidden_state. Keeping ONE tokenize+pool implementation is what guarantees the browser query
// vectors, the browser builder, and the CLI builder all produce identical vectors.
import { Tokenizer } from '@huggingface/tokenizers'
import { EMBED } from '../models/registry'

// bge-small-en-v1.5 is an ASYMMETRIC retrieval model: the QUERY is embedded with this instruction
// while passages are not. The index stores raw-passage vectors, so this is a query-side-only prefix.
export const QUERY_INSTRUCTION = 'Represent this sentence for searching relevant passages: '

export interface TokenizedBatch {
  ids: number[][]
  masks: number[][]
  batch: number
  seq: number // padded length (max over the batch)
}

/** Tokenize a batch and pad to the longest sequence (pad id 0, mask 0), matching transformers.js. */
export function tokenizeBatch(tok: Tokenizer, texts: string[]): TokenizedBatch {
  const encoded = texts.map((t) => {
    const e = tok.encode(t, { add_special_tokens: true })
    const ids = Array.from(e.ids, Number)
    const mask = e.attention_mask ? Array.from(e.attention_mask, Number) : ids.map(() => 1)
    return { ids, mask }
  })
  let seq = 0
  for (const e of encoded) seq = Math.max(seq, e.ids.length)
  seq = Math.min(seq, EMBED.maxSeqTokens)
  const ids: number[][] = []
  const masks: number[][] = []
  for (const e of encoded) {
    const id = e.ids.slice(0, seq)
    const mk = e.mask.slice(0, seq)
    while (id.length < seq) {
      id.push(0)
      mk.push(0)
    }
    ids.push(id)
    masks.push(mk)
  }
  return { ids, masks, batch: texts.length, seq }
}

/** Masked mean-pool the last_hidden_state [batch, seq, dim] then L2-normalize each row. */
export function poolAndNormalize(hidden: Float32Array, masks: number[][], batch: number, seq: number, dim: number): Float32Array[] {
  const out: Float32Array[] = []
  for (let b = 0; b < batch; b++) {
    const v = new Float32Array(dim)
    let n = 0
    for (let t = 0; t < seq; t++) {
      if (!masks[b][t]) continue
      n++
      const base = (b * seq + t) * dim
      for (let d = 0; d < dim; d++) v[d] += hidden[base + d]
    }
    const inv = n ? 1 / n : 1
    let norm = 0
    for (let d = 0; d < dim; d++) {
      v[d] *= inv
      norm += v[d] * v[d]
    }
    norm = Math.sqrt(norm) || 1
    for (let d = 0; d < dim; d++) v[d] /= norm
    out.push(v)
  }
  return out
}
