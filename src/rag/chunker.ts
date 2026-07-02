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

/** Split markdown into sections at headings, keeping each heading's TEXT (no #'s: stored chunks
 *  are plain text the model may recite) as the section's first line. */
function splitSections(clean: string): string[] {
  const lines = clean.split('\n')
  const sections: string[] = []
  let cur: string[] = []
  let curHasHeading = false
  const push = (): void => {
    // Drop heading-only sections (e.g. a document's bare H1 before its first real section):
    // a stub chunk like "Frequently asked questions" attracts generic questions and outranks
    // the chunks that actually answer them.
    const body = (curHasHeading ? cur.slice(1) : cur).join('').trim()
    if (body) sections.push(cur.join('\n'))
  }
  for (const ln of lines) {
    if (/^#{1,6} /.test(ln.trim())) {
      push()
      cur = [ln.trim().replace(/^#+ /, '')]
      curHasHeading = true
    } else {
      cur.push(ln)
    }
  }
  push()
  return sections
}

function packUnits(text: string, targetChars: number, overlapChars: number): string[] {
  // Sentence-ish units. Split ONLY at a terminator followed by whitespace, or at a line break - so
  // "aidekin.com", "1.5" and "e.g." are never broken mid-token. The old split-on-every-terminator +
  // rejoin-with-a-space corrupted URLs, decimals and abbreviations right in the stored chunk
  // ("aidekin.com" -> "aidekin. com"), which the model then reproduced and mangled further.
  const units = text.split(/(?<=[.!?…])\s+|\n+/).map((u) => u.trim()).filter(Boolean)

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

export function chunkText(text: string, options: ChunkOptions = {}): string[] {
  const targetTokens = Math.min(options.targetTokens ?? 220, EMBED.maxSeqTokens)
  const targetChars = targetTokens * 4
  const overlapChars = Math.floor(targetChars * (options.overlapRatio ?? 0.15))

  const clean = text.replace(/\r\n/g, '\n').replace(/[ \t]+/g, ' ').trim()
  if (!clean) return []

  // Markdown-heading-aware: packing unrelated sections into one chunk gives it a mean-pooled
  // embedding that matches nothing strongly (an FAQ was the worst case: every question scored
  // ~0.45-0.52, under the relevance gate). Pack each heading's section independently so a chunk
  // states one topic, opening with its own heading text.
  const sections = splitSections(clean)
  if (sections.length > 1) return sections.flatMap((s) => packUnits(s, targetChars, overlapChars))
  return packUnits(clean, targetChars, overlapChars)
}
