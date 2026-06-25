// Splits streaming LLM tokens into clause/sentence chunks at boundaries so each
// completed clause can be sent to TTS immediately (the key to low first-audio
// latency via streaming overlap). A minimum length avoids flushing tiny fragments.

// Break at sentence enders (. ! ? … ; : newline) AND at a comma that's followed by
// whitespace - the comma break lets the FIRST clause reach TTS sooner (lower time-to-
// first-audio) without splitting numbers like "1,000" (the lookahead requires a space).
const BOUNDARY = /.*?(?:[.!?…;:\n]+|,(?=\s))["'”’)\]]*\s*/gs

export class SentenceChunker {
  private buffer = ''
  private readonly minChars: number

  constructor(minChars = 2) {
    this.minChars = minChars
  }

  /** Append streamed text; return any newly-completed clauses. */
  push(text: string): string[] {
    this.buffer += text
    const out: string[] = []
    let consumed = 0
    for (const match of this.buffer.matchAll(BOUNDARY)) {
      const seg = match[0].trim()
      consumed = (match.index ?? 0) + match[0].length
      if (seg.length >= this.minChars) out.push(seg)
    }
    if (consumed > 0) this.buffer = this.buffer.slice(consumed)
    return out
  }

  /** Return the trailing remainder (no terminal punctuation), if any, and clear. */
  flush(): string | null {
    const rest = this.buffer.trim()
    this.buffer = ''
    return rest.length >= this.minChars ? rest : null
  }

  reset(): void {
    this.buffer = ''
  }
}
