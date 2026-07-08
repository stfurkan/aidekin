// Headless parity gate: bitgpu/chat's ChatTokenizer (which inlines @huggingface/tokenizers +
// @huggingface/jinja) must be BYTE-EXACT with @huggingface/transformers' AutoTokenizer for our model,
// across a broad + fuzz corpus (encode), decode round-trips, chat-template renders, and streaming
// decode. It is the guard that the on-device text boundary matches the reference: run it on every
// bitgpu version bump.  Run: npm run verify-tokenizer
import { AutoTokenizer } from '@huggingface/transformers'
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { ChatTokenizer } from 'bitgpu/chat'

const ID = 'onnx-community/Bonsai-1.7B-ONNX'
const CACHE = join(dirname(fileURLToPath(import.meta.url)), '..', 'node_modules', '.cache', 'tok-parity')

async function hubJson(name: string): Promise<unknown> {
  if (!existsSync(CACHE)) mkdirSync(CACHE, { recursive: true })
  const p = join(CACHE, name)
  if (existsSync(p)) return JSON.parse(readFileSync(p, 'utf8'))
  const j: unknown = await (await fetch(`https://huggingface.co/${ID}/resolve/main/${name}`)).json()
  writeFileSync(p, JSON.stringify(j))
  return j
}

let fails = 0
const check = (ok: boolean, label: string, detail = ''): void => {
  if (!ok) fails++
  if (!ok || process.env.VERBOSE) console.log(`  ${ok ? '[PASS]' : '[FAIL]'} ${label}${detail ? '  ' + detail : ''}`)
}

const tokJson = await hubJson('tokenizer.json')
const tokCfg = (await hubJson('tokenizer_config.json')) as Record<string, unknown>
const ref = await AutoTokenizer.from_pretrained(ID)
const mine = new ChatTokenizer(tokJson, tokCfg)

const refEnc = (t: string): number[] => Array.from(ref.encode(t, { add_special_tokens: false }), Number)

// curated tricky cases
const curated = [
  'The capital of Japan is', 'Hello, world! How are you today?', 'def add(a, b):\n  return a + b  # sum',
  'Numbers: 1234567890 and 3.14159 and -42', '   leading and   multiple   spaces\tand\ttabs\nand newlines',
  'Unicode: café, naïve, Zürich, 日本語, 北京, emoji 🚀🔥😀👨‍👩‍👧‍👦', 'Mixed CASE Punctuation?!... (a) [b] {c}',
  "Contractions: don't, it's, they're, I'll, we've", '<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n',
  '<think>\nreason\n</think>\n\nanswer', 'a'.repeat(300), 'tab\tand\rcarriage', '한국어 테스트 中文测试 العربية',
  'math: ∑ ∫ √ π ≈ ∞ ≤ ≥ ≠', '', ' ', '\n\n\n', 'ＦＵＬＬＷＩＤＴＨ', 'zero​width​space',
]

// seeded fuzz corpus (LCG, reproducible): random unicode incl. astral planes + control + special tokens
let seed = 1234567
const rnd = (): number => (seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff
const SPECIALS = ['<|im_start|>', '<|im_end|>', '<think>', '</think>', '<|endoftext|>', '<tool_call>']
const fuzz: string[] = []
for (let i = 0; i < 300; i++) {
  const len = 1 + Math.floor(rnd() * 60)
  let s = ''
  for (let j = 0; j < len; j++) {
    const r = rnd()
    if (r < 0.08) s += SPECIALS[Math.floor(rnd() * SPECIALS.length)]
    else if (r < 0.55) s += String.fromCharCode(32 + Math.floor(rnd() * 95)) // printable ASCII
    else if (r < 0.85) s += String.fromCharCode(Math.floor(rnd() * 0x2000)) // BMP incl. CJK/latin-ext
    else s += String.fromCodePoint(0x10000 + Math.floor(rnd() * 0x5000)) // astral plane
  }
  fuzz.push(s)
}

console.log(`encode parity (${curated.length} curated + ${fuzz.length} fuzz)...`)
let encMis = 0
for (const t of [...curated, ...fuzz]) {
  const a = refEnc(t)
  const b = mine.encode(t, false)
  if (a.length !== b.length || !a.every((x, i) => x === b[i])) {
    encMis++
    if (encMis <= 5) console.log('  MISMATCH', JSON.stringify(t.slice(0, 40)), '\n    ref ', a.slice(0, 16).join(','), '\n    mine', b.slice(0, 16).join(','))
  }
}
check(encMis === 0, `encode ids identical`, `${encMis} mismatches`)

console.log('decode round-trip parity...')
let decMis = 0
for (const t of [...curated, ...fuzz]) {
  const ids = refEnc(t)
  if (!ids.length) continue
  if (ref.decode(ids, { skip_special_tokens: true }) !== mine.decode(ids, true)) decMis++
}
check(decMis === 0, 'decode identical', `${decMis} mismatches`)

console.log('chat-template parity...')
const chats: { msgs: { role: string; content: string }[]; think: boolean }[] = [
  { msgs: [{ role: 'user', content: 'What is the capital of Japan?' }], think: false },
  { msgs: [{ role: 'system', content: 'You are helpful.' }, { role: 'user', content: 'Hi' }, { role: 'assistant', content: 'Hello!' }, { role: 'user', content: 'Bye' }], think: false },
  { msgs: [{ role: 'user', content: 'Solve 2+2' }], think: true },
  { msgs: [{ role: 'system', content: 'Be terse.' }, { role: 'user', content: 'café ∑ 日本語 🚀' }], think: false },
]
let tmplMis = 0
for (const { msgs, think } of chats) {
  const a = ref.apply_chat_template(msgs, { add_generation_prompt: true, tokenize: false, enable_thinking: think } as never) as unknown as string
  const b = mine.applyChatTemplate(msgs, { addGenerationPrompt: true, enableThinking: think })
  if (a !== b) {
    tmplMis++
    console.log('  TMPL MISMATCH think=' + think, '\n    ref ', JSON.stringify(a), '\n    mine', JSON.stringify(b))
  }
  // and the rendered string must encode identically
  if (refEnc(a).join(',') !== mine.encode(a, false).join(',')) tmplMis++
}
check(tmplMis === 0, 'chat-template render + encode identical', `${tmplMis} mismatches`)

console.log('streaming decode parity...')
let streamMis = 0
for (const t of curated.slice(0, 12)) {
  const ids = refEnc(t)
  if (!ids.length) continue
  const stream = mine.createDecoderStream(true)
  let acc = ''
  for (const id of ids) acc += stream.push(id)
  acc += stream.flush()
  if (acc !== mine.decode(ids, true)) streamMis++ // streamed output reassembles to the full decode
}
check(streamMis === 0, 'streaming decode reassembles to full decode', `${streamMis} mismatches`)

console.log(`\neos token id: ${mine.eosTokenId} (expect 151645)`)
check(mine.eosTokenId === 151645, 'eos token id correct')

console.log(`\n${fails === 0 ? 'TOKENIZER PARITY: ALL PASS' : fails + ' CHECK(S) FAILED'}`)
process.exit(fails === 0 ? 0 : 1)
