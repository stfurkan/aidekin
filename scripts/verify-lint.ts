// Behavior guard for the knowledge-builder lints (src/rag/lint.ts). Locks the intended-fire cases,
// the must-NOT-fire cases (false-positive surface), and robustness on degenerate input, so a future
// threshold or pattern change cannot silently regress the builder's authoring feedback.
//   npm run verify-lint
import { chunkText } from '../src/rag/chunker.ts'
import { lintKnowledge, type LintCode } from '../src/rag/lint.ts'

let fails = 0
const check = (ok: boolean, label: string, detail = ''): void => {
  if (!ok) fails++
  if (!ok || process.env.VERBOSE) console.log(`  ${ok ? '[PASS]' : '[FAIL]'} ${label}${detail ? '  ' + detail : ''}`)
}

/** Run the real pipeline (chunk like the builder, then lint) and return the set of codes emitted. */
function codes(sources: { name: string; text: string }[]): Set<LintCode> {
  const chunks = sources.flatMap((s) => chunkText(s.text).map((text) => ({ text, source: s.name })))
  return new Set(lintKnowledge(sources, chunks).map((l) => l.code))
}
const one = (name: string, text: string): { name: string; text: string }[] => [{ name, text }]

// ── secrets fire (safety; over-warning is the safe direction) ─────────────────
const SECRETS: [string, string][] = [
  ['AWS access key', 'internal AKIAIOSFODNN7EXAMPLE do not share'],
  ['OpenAI-style key', 'token sk-abcdefghijklmnopqrstuvwxyz012345'],
  ['GitHub token', 'ci uses ghp_ABCdef0123456789ABCdef0123456789ABCD'],
  ['Slack token', 'bot xoxb-123456789012-abcdefABCDEF'],
  ['Google API key', 'maps AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZ0123456 key'], // AIza + 35 chars = 39 total
  ['private key block', '-----BEGIN RSA PRIVATE KEY-----\nMIIabc\n-----END RSA PRIVATE KEY-----'],
]
for (const [label, text] of SECRETS) check(codes(one(label, text)).has('secret'), `secret fires: ${label}`)

// ── secrets do NOT fire on ordinary content ───────────────────────────────────
check(!codes(one('ids', 'Order ABC-12345, product SKU XYZ99, invoice 2024-0007 are all fine.')).has('secret'), 'no secret FP on ordinary ids')
check(!codes(one('prose', 'We open at 7:30 and close at 6. Espresso is 3.0 and matcha is 4.5.')).has('secret'), 'no secret FP on normal prose')

// ── unstructured wall fires; well-formed text does not ────────────────────────
const wall = 'we are a cafe that serves coffee tea and brunch every single day of the week come and visit us '.repeat(9)
check(codes(one('wall', wall)).has('unstructured'), 'unstructured fires on a long break-less wall')
const punctuated = 'We open at 7:30 on weekdays. Brunch runs until 2:30. We serve espresso, matcha, and chai. Parking is behind the building. '.repeat(6)
check(!codes(one('punctuated', punctuated)).has('unstructured'), 'no unstructured FP on well-punctuated long prose')
const list = Array.from({ length: 30 }, (_, i) => `Item ${i}: a short line describing one thing`).join('\n')
check(!codes(one('list', list)).has('unstructured'), 'no unstructured FP on a newline list')
const spanish = 'Bienvenidos a nuestro café. Abrimos a las ocho de la mañana. Servimos café con leche, té y pasteles. '.repeat(6)
check(!codes(one('spanish', spanish)).has('unstructured'), 'no unstructured FP on Latin (Spanish) prose')
check(!codes(one('short', 'A short note about our hours.')).has('unstructured'), 'no unstructured FP under the length floor')

// ── duplicate fires across identical content, not across distinct content ─────
const same = 'The loyalty card gives you the tenth coffee free, one stamp per visit.'
check(codes([{ name: 'a', text: same }, { name: 'b', text: same }]).has('duplicate'), 'duplicate fires on identical sources')
check(!codes([{ name: 'a', text: 'We open at 7:30 on weekdays.' }, { name: 'b', text: 'Parking is behind the building.' }]).has('duplicate'), 'no duplicate FP on distinct sources')

// ── robustness: degenerate input never throws and emits nothing ───────────────
for (const [label, srcs] of [
  ['empty list', []],
  ['empty text', one('e', '')],
  ['whitespace only', one('w', '   \n\n\t ')],
  ['one char', one('c', 'x')],
] as [string, { name: string; text: string }[]][]) {
  try {
    const n = codes(srcs).size
    check(n === 0, `robust + quiet: ${label}`, `${n} codes`)
  } catch (e) {
    check(false, `robust + quiet: ${label}`, `THREW ${(e as Error).message}`)
  }
}

console.log(`\n${fails === 0 ? 'LINT VERIFY: ALL PASS' : fails + ' CHECK(S) FAILED'}`)
process.exit(fails === 0 ? 0 : 1)
