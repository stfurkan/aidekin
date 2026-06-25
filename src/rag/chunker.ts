// Recursive-ish text chunker shared by the browser builder and the CLI. Accumulates
// sentence units up to a token budget (≈4 chars/token) with a small overlap so context
// isn't lost at chunk boundaries. Defaults are tuned to MiniLM's 256-token limit.

import { EMBED } from '../models/registry'

export interface ChunkOptions {
  /** Target chunk size in approximate tokens (kept under the embedder's max). */
  targetTokens?: number
  /** Fraction of a chunk repeated at the start of the next (0-0.5). */
  overlapRatio?: number
}

/** Keep a word-aligned tail of `maxChars` from the end of `s` (for overlap). */
function tail(s: string, maxChars: number): string {
  if (maxChars <= 0 || s.length <= maxChars) return s.length <= maxChars ? s : ''
  const slice = s.slice(s.length - maxChars)
  const sp = slice.indexOf(' ')
  return sp > 0 ? slice.slice(sp + 1) : slice
}

export function chunkText(text: string, options: ChunkOptions = {}): string[] {
  const targetTokens = Math.min(options.targetTokens ?? 220, EMBED.maxSeqTokens)
  const targetChars = targetTokens * 4
  const overlapChars = Math.floor(targetChars * (options.overlapRatio ?? 0.15))

  const clean = text.replace(/\r\n/g, '\n').replace(/[ \t]+/g, ' ').trim()
  if (!clean) return []

  // Sentence-ish units (sentence terminators or a line of text).
  const units = clean.match(/\s*\S[^.!?…\n]*[.!?…]*/g)?.map((u) => u.trim()).filter(Boolean) ?? [clean]

  const chunks: string[] = []
  let buf = ''
  for (const unit of units) {
    if (buf && buf.length + 1 + unit.length > targetChars) {
      chunks.push(buf.trim())
      buf = tail(buf, overlapChars)
    }
    buf = buf ? `${buf} ${unit}` : unit
    // Hard-split a single oversized unit (e.g. a giant line with no punctuation).
    while (buf.length > targetChars * 1.6) {
      chunks.push(buf.slice(0, targetChars).trim())
      buf = buf.slice(targetChars - overlapChars)
    }
  }
  if (buf.trim()) chunks.push(buf.trim())
  return chunks.map((c) => c.trim()).filter(Boolean)
}
