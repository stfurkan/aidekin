// Recall-robustness gate: the behavioral golden set checks multi-turn recall on ONE fixed seed with
// an easy phrasing, which OVERSTATES how reliably the 1.7B recalls a personal fact under production
// conditions (random seed, terser phrasing, the always-on SITE_GROUNDING abstention pressure). A
// single-seed pass hides a seed-fragile behaviour. This sweeps SEEDS x phrasings on ONE loaded engine
// and reports a PASS RATE per case, so "does <the shipping kvCache> recall reliably?" is answered from
// a distribution, not a lucky sample - and flipping kvCache in the registry + re-running compares
// modes on equal footing. Dev-only (not a vite build input); driven by scripts/eval-recall.mjs.
import { ConversationEngine } from '@/engine/conversationEngine'
import { createRetriever } from '@/rag/retriever'
import { resolveSystemPrompt } from '@/widget/protocol'

const out = document.getElementById('out')!
const log = (s: string): void => {
  out.textContent += s + '\n'
}

// Two phrasings: the eval's easy version, and the demo's terser one (no "remember", abbreviated).
const CASES = [
  { name: 'easy (eval phrasing)', set: 'My favorite color is blue. Please remember that.', ask: 'What is my favorite color?' },
  { name: 'terse (demo phrasing)', set: 'my favorite color is blue', ask: 'what is my fav color' },
]
// Fixed seeds so the pass rate is REPRODUCIBLE (a moving rate would be a flaky gate). Keep the set
// small when the machine is warm - headless WebGPU throttles hard under sustained load, so each
// generation can crawl; run on a cooled machine (or trim SEEDS) for a fuller picture.
const SEEDS = [1, 42, 99, 777, 2024, 31337]

async function run(): Promise<void> {
  out.textContent = ''
  const engine = new ConversationEngine({
    systemPrompt: resolveSystemPrompt({ title: 'aidekin' }),
    brandName: 'aidekin',
    ragTopK: 3,
  })
  // RAG on (like the demo) so SITE_GROUNDING is in the prompt - that is the abstention pressure that
  // makes the model treat a personal question as a site question and refuse.
  const { retriever } = await createRetriever('/aidekin-knowledge.bin')
  engine.setRetriever(retriever)
  await engine.loadLlm() // creates + initializes the LLM worker (without this, generate rejects "not ready")
  log(`recall robustness: ${SEEDS.length} seeds x ${CASES.length} phrasings (kvCache = whatever registry ships)\n`)

  const rates: string[] = []
  for (const c of CASES) {
    let pass = 0
    const misses: string[] = []
    for (const seed of SEEDS) {
      engine.setSamplerSeed(seed) // sweep seeds on the one loaded engine; clearHistory = fresh chat
      engine.clearHistory()
      await engine.sendUserMessage(c.set)
      const reply = await engine.sendUserMessage(c.ask)
      if (/blue/i.test(reply)) pass++
      else misses.push(`seed ${seed}: ${reply.slice(0, 70)}`)
    }
    const pct = Math.round((100 * pass) / SEEDS.length)
    rates.push(`${c.name}: ${pass}/${SEEDS.length} (${pct}%)`)
    log(`[${c.name}] recalled "blue" in ${pass}/${SEEDS.length} seeds (${pct}%)`)
    for (const m of misses.slice(0, 4)) log(`    miss ${m}`)
    log('')
  }
  log(`SUMMARY  ${rates.join('  |  ')}`)
  log('RECALL DONE')
}

void run()
