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
}

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
  u8.set(textBytes, o); o += textBytes.length
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

  /** The chunk text at a given row (used by the parity-guard script). */
  textAt(i: number): string {
    return this.texts[i]
  }

  /** Top-k by cosine (≈ dot product, since both query and stored rows are unit vectors). */
  search(query: Float32Array, k: number): SearchHit[] {
    const { int8, scales, dim, count } = this
    const scored: { i: number; score: number }[] = new Array(count)
    for (let i = 0; i < count; i++) {
      const base = i * dim
      const scale = scales[i]
      let dot = 0
      for (let j = 0; j < dim; j++) dot += query[j] * int8[base + j] * scale
      scored[i] = { i, score: dot }
    }
    scored.sort((a, b) => b.score - a.score)
    return scored.slice(0, Math.max(0, k)).map((s) => ({
      text: this.texts[s.i],
      score: s.score,
      source: this.sources[s.i] || undefined,
    }))
  }
}
