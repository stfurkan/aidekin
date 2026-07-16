// Multi-seed robustness gate for the SEED-FRAGILE behaviors. The golden set (main.ts) judges each
// scenario on ONE fixed seed - great for regression detection, but a single-seed pass overstates how
// reliably a borderline behavior holds under production sampling (random seed). The failures that
// matter here are seed-dependent: a false "yes" to an unowned feature, a leaked instruction, an
// over-refused greeting. This sweeps those scenarios x SEEDS on one loaded engine and reports a
// PASS RATE per scenario, so a KV-precision change (f16 -> q8) is compared on the SAME distribution
// rather than a lucky sample. Flip `kvCache` in the registry and re-run to A/B the two modes.
// Dev-only (not a vite build input); driven by scripts/eval-robustness.mjs.
import { ConversationEngine } from '@/engine/conversationEngine'
import { createRetriever } from '@/rag/retriever'
import { resolveSystemPrompt } from '@/widget/protocol'
import { GLOBAL_MUST_NOT, SCENARIOS, type Scenario } from './scenarios'

const out = document.getElementById('out')!
const log = (s: string): void => {
  out.textContent += s + '\n'
}

// The borderline behaviors whose correctness is seed-fragile and safety-relevant: false-affirmation
// of unowned features, instruction leakage, and the two over-triggering guards (world knowledge must
// still answer, small talk must not be refused). The grounded lookups and format rules are robust
// across seeds and are covered by the single-seed golden set, so they stay out to keep this fast.
const BORDERLINE = new Set([
  'unowned feature: no false claim (phone support)',
  'unowned feature: no false claim (mobile app)',
  'unowned feature: no false claim (enterprise plan)',
  'no context bleed: still no false claim after a grounded turn',
  'unknown integration: admits not knowing instead of inventing',
  'prompt injection does not leak instructions',
  'general-knowledge question may be answered (world knowledge is allowed)',
  'social question is answered warmly, not refused (how are you)',
])
const CASES: Scenario[] = SCENARIOS.filter((s) => BORDERLINE.has(s.name))

// Fixed seeds -> reproducible pass rates. Includes 42 (the golden-set seed) so the single-seed
// result is visible inside the distribution.
const SEEDS = [1, 42, 99, 777, 2024]

// One scenario turn-sequence against the engine; returns the list of assertion problems (empty = pass).
async function judge(engine: ConversationEngine, sc: Scenario, usedRef: { used: number }): Promise<string[]> {
  engine.clearHistory()
  let reply = ''
  for (const turn of sc.turns) {
    usedRef.used = 0
    reply = await engine.sendUserMessage(turn)
  }
  const problems: string[] = []
  if (!reply.trim()) problems.push('empty reply')
  for (const re of GLOBAL_MUST_NOT) if (re.test(reply)) problems.push(`format ${re}`)
  const e = sc.expect
  for (const re of e.mustMatch ?? []) if (!re.test(reply)) problems.push(`missing ${re}`)
  for (const re of e.mustNotMatch ?? []) if (re.test(reply)) problems.push(`forbidden ${re}`)
  if (e.maxChars && reply.length > e.maxChars) problems.push(`too long (${reply.length})`)
  if (e.grounded && usedRef.used === 0) problems.push('expected grounded')
  if (e.ungrounded && usedRef.used > 0) problems.push(`expected ungrounded (used=${usedRef.used})`)
  return problems
}

async function run(): Promise<void> {
  out.textContent = ''
  const usedRef = { used: 0 }
  const engine = new ConversationEngine({
    systemPrompt: resolveSystemPrompt({ title: 'aidekin' }),
    brandName: 'aidekin',
    ragTopK: 3,
    callbacks: { onRetrieval: (info) => (usedRef.used = info.used) },
  })
  const { retriever } = await createRetriever('/aidekin-knowledge.bin')
  engine.setRetriever(retriever)
  await engine.loadLlm() // without this, sendUserMessage rejects "LLM worker not ready"
  log(`robustness: ${CASES.length} borderline scenarios x ${SEEDS.length} seeds (kvCache = whatever registry ships)\n`)

  const summary: string[] = []
  let worst = 100
  for (const sc of CASES) {
    let pass = 0
    const misses: string[] = []
    for (const seed of SEEDS) {
      engine.setSamplerSeed(seed)
      const problems = await judge(engine, sc, usedRef)
      if (problems.length === 0) pass++
      else misses.push(`seed ${seed}: ${problems.join(', ')}`)
    }
    const pct = Math.round((100 * pass) / SEEDS.length)
    worst = Math.min(worst, pct)
    summary.push(`${pct}% ${sc.name}`)
    log(`[${pass}/${SEEDS.length} = ${pct}%] ${sc.name}`)
    for (const m of misses.slice(0, 4)) log(`    ${m}`)
  }
  log('')
  summary.sort((a, b) => parseInt(a) - parseInt(b))
  for (const s of summary) log(`SUMMARY ${s}`)
  log(`WORST ${worst}%`)
  log('ROBUSTNESS DONE')
}

run().catch((e) => log(`FATAL: ${(e as Error).message}`))
