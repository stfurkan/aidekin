// Pure, structural lints over the knowledge a builder is about to embed: no model, no network, no
// server - just text checks, so they cost nothing and run as content is added. They flag what hurts
// on-device retrieval (unstructured walls, duplicate chunks) plus the one safety footgun (a secret in
// a file that ships PUBLIC). Scoped to Latin-script content: word/break counting is whitespace- and
// punctuation-based, which covers space-separated languages (en/es/fr/de/pt/...).
//
// Structure + secrets are judged per SOURCE (what the author sees and edits); duplicates are judged on
// the CHUNKS the file is actually built from. Findings aggregate to one line per source per issue.

export type LintCode = 'unstructured' | 'duplicate' | 'secret'

export interface Lint {
  source?: string
  severity: 'warn' | 'danger'
  code: LintCode
  message: string
}

// A source this long with breaks (sentence terminators or newlines) this sparse is a wall of text:
// the chunker can only cut it at arbitrary character boundaries, mid-thought, so the pieces embed
// vaguely. Well-punctuated prose and lists break often and never trip this.
const WALL_MIN_CHARS = 600
const WALL_AVG_RUN = 400 // average characters between one break and the next

// Secret shapes are formats, not natural language, so this stays language-agnostic. Kept conservative
// (well-known prefixes / key blocks / JWTs) to avoid crying wolf on ordinary IDs and long words.
const SECRET_PATTERNS: { re: RegExp; what: string }[] = [
  { re: /-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----/, what: 'a private key' },
  { re: /\bAKIA[0-9A-Z]{16}\b/, what: 'an AWS access key' },
  { re: /\b(?:ghp|gho|ghu|ghs|ghr|github_pat)_[A-Za-z0-9_]{20,}\b/, what: 'a GitHub token' },
  { re: /\bxox[baprs]-[A-Za-z0-9-]{10,}\b/, what: 'a Slack token' },
  { re: /\bsk-[A-Za-z0-9]{20,}\b/, what: 'an API secret key' },
  { re: /\bAIza[0-9A-Za-z_-]{35}\b/, what: 'a Google API key' },
  { re: /\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b/, what: 'a JWT' },
]

const normalize = (s: string): string => s.replace(/\s+/g, ' ').trim().toLowerCase()
const plural = (n: number, w: string): string => `${n} ${w}${n > 1 ? 's' : ''}`

function isWall(text: string): boolean {
  if (text.length < WALL_MIN_CHARS) return false
  const breaks = (text.match(/[.!?…]\s|\n/g) ?? []).length
  return text.length / (breaks + 1) > WALL_AVG_RUN
}

/** Structural lints for the builder. `sources` are the raw inputs; `chunks` are what the file is built
 *  from. Advisory only, secrets-first (safety), one finding per source per issue. */
export function lintKnowledge(
  sources: { name: string; text: string }[],
  chunks: { text: string; source?: string }[],
): Lint[] {
  const danger: Lint[] = []
  const warn: Lint[] = []

  for (const s of sources) {
    const found = new Set<string>()
    for (const { re, what } of SECRET_PATTERNS) if (re.test(s.text)) found.add(what)
    for (const what of found)
      danger.push({ source: s.name, severity: 'danger', code: 'secret', message: `This looks like ${what}. The knowledge file is delivered to every visitor, so remove it before you publish.` })

    if (isWall(s.text))
      warn.push({ source: s.name, severity: 'warn', code: 'unstructured', message: `This is a long block with few sentence or paragraph breaks, so it gets split mid-thought. Add headings and paragraph breaks.` })
  }

  // Duplicate chunks (same content indexed twice competes in search and wastes the retrieval budget).
  const seen = new Set<string>()
  const dup = new Map<string, number>()
  for (const c of chunks) {
    const n = normalize(c.text)
    if (!n) continue
    if (seen.has(n)) dup.set(c.source ?? '', (dup.get(c.source ?? '') ?? 0) + 1)
    else seen.add(n)
  }
  for (const [s, n] of dup)
    warn.push({ source: s || undefined, severity: 'warn', code: 'duplicate', message: `${plural(n, 'duplicate chunk')}. Remove the repeated content so chunks don't compete in search.` })

  return [...danger, ...warn]
}
