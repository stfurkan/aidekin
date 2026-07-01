// Splits streaming LLM tokens into clause/sentence chunks at boundaries so each
// completed clause can be sent to TTS immediately (the key to low first-audio
// latency via streaming overlap). A minimum length avoids flushing tiny fragments.

// Break at sentence enders. `! ? … ; newline` always break. `.` `:` and `,` break ONLY when
// followed (past any closing quote/bracket) by whitespace - so we never split inside a URL
// ("https://aidekin.com/x"), a decimal ("1.5"), a time ("9:30") or "1,000". Requiring the trailing
// whitespace also holds a boundary back mid-stream until the next token confirms it, so a URL that
// arrives across tokens stays in one clause (then `speakable()` rewrites it for TTS). flush() emits
// the final clause, which has no trailing whitespace.
const BOUNDARY = /.*?(?:[!?…;\n]+|[.:,](?=["'”’)\]]*\s))["'”’)\]]*\s*/gs

/** Prepare a clause for text-to-speech. A small model sometimes emits raw URLs (echoed from RAG
 *  context) despite the system prompt; without this, TTS reads "h-t-t-p-s colon slash slash aide-kin
 *  dot com" aloud. Rewrite URLs to a spoken phrase and drop stray markdown markers. The visible
 *  transcript keeps the original text - this only changes what is spoken. */
export function speakable(clause: string): string {
  return clause
    .replace(/<\/?[a-zA-Z][^>\n]*>/g, '') // drop stray tag leakage (e.g. the RAG "<info>" wrapper)
    .replace(/\b(?:https?:\/\/|www\.)[^\s)]+/gi, 'the link')
    .replace(/[*_`#]+/g, '')
    .replace(/\s{2,}/g, ' ')
    .trim()
}

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
