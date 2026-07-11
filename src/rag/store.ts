// Static knowledge-index format + in-browser brute-force cosine search.
//
// File layout (little-endian):
//   "SONR"(u32) | version(u32) | headerLen(u32) | header JSON
//   | int8[dim] × count            (per-row symmetric-quantized unit vectors)
//   | float32 scale × count        (per-row dequant scale)
//   | metaLen(u32) | meta JSON     ([{ source }])
//   | textLen(u32) | text JSON     (string[])
//
// Vectors are unit-normalized BEFORE quantization, so cosine ≈ dot product. Only the
// stored vectors are int8; the query stays fp32. Header is self-describing so a widget
// rejects an index built with a different embedding model before parsing the body.

import { EMBED } from '../models/registry'

const MAGIC = 0x534f4e52 // "SONR"
const VERSION = 1

export interface IndexChunk {
  text: string
  vector: Float32Array
  source?: string
}

export interface SearchHit {
  text: string
  score: number
  source?: string
  /** True when the chunk contains every DISTINCTIVE (corpus-rare, non-stopword) term of the query.
   *  A scale-free lexical-relevance signal the engine's gate accepts even when cosine is moderate, so
   *  an exact menu-term query ("do you have matcha") grounds even if the embedding scores it low. A
   *  greeting has no distinctive terms, so this is never true for it - the abstention gate stays intact. */
  lexMatch?: boolean
}

// ── lexical side of hybrid retrieval ──────────────────────────────────────────
// A lexical signal over the chunk texts (already in the index, so no format change). It does NOT
// reorder the semantic ranking - that hurt precision - it only lets the engine's gate rescue a chunk
// that scored just under the cosine threshold but is an exact term match. Pure set arithmetic over a
// small chunk set, so it adds no meaningful latency.
//
// Terms are matched EXACTLY, by design - no stemming. The embedding already handles morphology (park
// vs parking embed close), and the distinctive terms this gate targets are invariant nouns and names
// (matcha, gluten, a product SKU) that appear verbatim in question and content. Stemming would add
// over-match risk (university/universe -> univers) for negligible recall, so exact-match is the scope.
const DISTINCTIVE_DF_RATIO = 0.5 // a query term is "distinctive" if it occurs in <= half the chunks
// Function words that never signal topic. df-thresholding already drops corpus-common words; this
// also drops short function words a tiny corpus might not make common on its own.
const STOP = new Set(
  ('a an and are as at be been by can could do does for from had has have how i if in into is it its ' +
    'me my no not of on or our so that the their them then there these they this to us was we were what ' +
    'when where which who why will with would you your').split(' '),
)
const lexTokens = (s: string): string[] => (s.toLowerCase().match(/[a-z0-9]+/g) ?? []).filter((t) => t.length >= 2 && !STOP.has(t))

export interface IndexHeader {
  modelId: string
  dim: number
  count: number
  quant: 'int8'
  dtype: string
  builtAt: string
}

/** Serialize embedded chunks into a compact, self-describing knowledge.bin buffer. */
export function serializeIndex(chunks: IndexChunk[], builtAt: string): ArrayBuffer {
  const dim = EMBED.dim
  const count = chunks.length
  const int8 = new Int8Array(count * dim)
  const scales = new Float32Array(count)

  for (let i = 0; i < count; i++) {
    const v = chunks[i].vector
    let max = 0
    for (let j = 0; j < dim; j++) {
      const a = Math.abs(v[j])
      if (a > max) max = a
    }
    const scale = max > 0 ? max / 127 : 1
    scales[i] = scale
    const base = i * dim
    for (let j = 0; j < dim; j++) {
      int8[base + j] = Math.max(-127, Math.min(127, Math.round(v[j] / scale)))
    }
  }

  const header: IndexHeader = { modelId: EMBED.hfModelId, dim, count, quant: 'int8', dtype: EMBED.dtype, builtAt }
  const enc = new TextEncoder()
  const headerBytes = enc.encode(JSON.stringify(header))
  const metaBytes = enc.encode(JSON.stringify(chunks.map((c) => ({ source: c.source ?? '' }))))
  const textBytes = enc.encode(JSON.stringify(chunks.map((c) => c.text)))

  const size =
    12 + headerBytes.length + int8.length + scales.length * 4 + 4 + metaBytes.length + 4 + textBytes.length
  const buf = new ArrayBuffer(size)
  const dv = new DataView(buf)
  const u8 = new Uint8Array(buf)
  let o = 0
  dv.setUint32(o, MAGIC, true); o += 4
  dv.setUint32(o, VERSION, true); o += 4
  dv.setUint32(o, headerBytes.length, true); o += 4
  u8.set(headerBytes, o); o += headerBytes.length
  u8.set(int8, o); o += int8.length
  for (let i = 0; i < count; i++) { dv.setFloat32(o, scales[i], true); o += 4 }
  dv.setUint32(o, metaBytes.length, true); o += 4
  u8.set(metaBytes, o); o += metaBytes.length
  dv.setUint32(o, textBytes.length, true); o += 4
  u8.set(textBytes, o)
  return buf
}

export class VectorStore {
  private constructor(
    private readonly int8: Int8Array,
    private readonly scales: Float32Array,
    private readonly texts: string[],
    private readonly sources: string[],
    readonly dim: number,
    readonly count: number,
    readonly header: IndexHeader,
  ) {}

  static fromBytes(buf: ArrayBuffer): VectorStore {
    const dv = new DataView(buf)
    const dec = new TextDecoder()
    let o = 0
    if (dv.getUint32(o, true) !== MAGIC) throw new Error('Not an aidekin knowledge file.')
    o += 4
    const version = dv.getUint32(o, true); o += 4
    if (version !== VERSION) throw new Error(`Unsupported knowledge file version ${version}.`)
    const headerLen = dv.getUint32(o, true); o += 4
    const header = JSON.parse(dec.decode(new Uint8Array(buf, o, headerLen))) as IndexHeader
    o += headerLen
    if (header.modelId !== EMBED.hfModelId || header.dim !== EMBED.dim) {
      throw new Error(
        `Knowledge file was built with a different embedding model (${header.modelId}, dim ${header.dim}). Rebuild it.`,
      )
    }
    const { count, dim } = header
    const int8 = new Int8Array(buf, o, count * dim); o += count * dim
    const scales = new Float32Array(count)
    for (let i = 0; i < count; i++) { scales[i] = dv.getFloat32(o, true); o += 4 }
    const metaLen = dv.getUint32(o, true); o += 4
    const meta = JSON.parse(dec.decode(new Uint8Array(buf, o, metaLen))) as { source: string }[]
    o += metaLen
    const textLen = dv.getUint32(o, true); o += 4
    const texts = JSON.parse(dec.decode(new Uint8Array(buf, o, textLen))) as string[]
    if (texts.length !== count || meta.length !== count) {
      throw new Error(
        `Corrupt knowledge file: header says ${count} chunks but found ${texts.length} texts and ${meta.length} sources.`,
      )
    }
    return new VectorStore(int8, scales, texts, meta.map((m) => m.source), dim, count, header)
  }

  // Lazily-built lexical index over the chunk texts: document frequency per term (for distinctiveness)
  // and a term set per chunk (for containment). Built on the first hybrid query, then cached. No BM25,
  // no scoring - the lexical side is a gate rescue only, never a re-ranker.
  private lex?: { df: Map<string, number>; docSets: Set<string>[] }

  private buildLex(): void {
    const docSets = this.texts.map((t) => new Set(lexTokens(t)))
    const df = new Map<string, number>()
    for (const s of docSets) for (const w of s) df.set(w, (df.get(w) ?? 0) + 1)
    this.lex = { df, docSets }
  }

  /** Full cosine (≈ dot product; query and stored rows are unit vectors) for every row. */
  private cosineAll(query: Float32Array): number[] {
    const { int8, scales, dim, count } = this
    const out = new Array<number>(count)
    for (let i = 0; i < count; i++) {
      const base = i * dim
      const scale = scales[i]
      let dot = 0
      for (let j = 0; j < dim; j++) dot += query[j] * int8[base + j] * scale
      out[i] = dot
    }
    return out
  }

  /** The chunk text at a given row (used by the parity-guard script). */
  textAt(i: number): string {
    return this.texts[i]
  }

  /** Top-k by cosine. Kept for the parity guard; runtime retrieval uses searchHybrid. */
  search(query: Float32Array, k: number): SearchHit[] {
    const cos = this.cosineAll(query)
    return Array.from({ length: this.count }, (_, i) => ({ i, score: cos[i] }))
      .sort((a, b) => b.score - a.score)
      .slice(0, Math.max(0, k))
      .map((s) => ({ text: this.texts[s.i], score: s.score, source: this.sources[s.i] || undefined }))
  }

  /** Hybrid retrieval: cosine RANKING (semantic precision preserved - the lexical side never reorders)
   *  plus a per-chunk `lexMatch` flag = the chunk contains every DISTINCTIVE (corpus-rare, non-stopword)
   *  query term. The engine's gate admits a lexMatch chunk that scored just under the cosine threshold,
   *  so an exact-term query ("do you have matcha", "what do you recommend") grounds even when the
   *  embedding underscores it. queryText is the raw user query for lexical tokenization. */
  searchHybrid(query: Float32Array, queryText: string, k: number): SearchHit[] {
    if (!this.lex) this.buildLex()
    const count = this.count
    const cos = this.cosineAll(query)
    const qterms = [...new Set(lexTokens(queryText))]
    const distinctive = qterms.filter((t) => t.length >= 3 && (this.lex!.df.get(t) ?? 0) <= Math.max(1, count * DISTINCTIVE_DF_RATIO))
    return Array.from({ length: count }, (_, i) => ({ i, score: cos[i] }))
      .sort((a, b) => b.score - a.score)
      .slice(0, Math.max(0, k))
      .map((s) => ({
        text: this.texts[s.i],
        score: s.score,
        source: this.sources[s.i] || undefined,
        lexMatch: distinctive.length > 0 && distinctive.every((t) => this.lex!.docSets[s.i].has(t)),
      }))
  }
}
