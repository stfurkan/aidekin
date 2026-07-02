// Behavioral eval runner. Drives the REAL product stack - ConversationEngine (prompt
// composition, RAG gate, fabrication guard, trim) -> llm.worker -> bitgpu on WebGPU, with the
// real knowledge index and the DEFAULT widget prompt - through the golden set in scenarios.ts.
// Deterministic via a fixed sampler seed. Served by the vite DEV server only (eval.html is not
// a build input); driven headlessly by scripts/eval-chat.mjs.
import { ConversationEngine } from '@/engine/conversationEngine'
import { createRetriever } from '@/rag/retriever'
import { resolveSystemPrompt } from '@/widget/protocol'
import { GLOBAL_MUST_NOT, SCENARIOS } from './scenarios'

const out = document.getElementById('out')!
const log = (s: string): void => {
  out.textContent += s + '\n'
}

interface TurnStats {
  used: number
}

async function run(): Promise<void> {
  out.textContent = ''
  const stats: TurnStats = { used: 0 }
  let lastError = ''

  const engine = new ConversationEngine({
    systemPrompt: resolveSystemPrompt({ title: 'aidekin' }), // the DEFAULT prompt every embedder gets
    ragTopK: 3,
    samplerSeed: 42,
    callbacks: {
      onRetrieval: (info) => {
        stats.used = info.used
      },
      onError: (where, message) => {
        lastError = `${where}: ${message}`
        log(`  [engine error] ${lastError}`)
      },
    },
  })

  log('loading knowledge index + LLM...')
  const t0 = performance.now()
  const { retriever, count } = await createRetriever('/aidekin-knowledge.bin')
  engine.setRetriever(retriever)
  await engine.loadLlm()
  log(`ready in ${((performance.now() - t0) / 1000).toFixed(1)}s (index: ${count} chunks)\n`)

  let passed = 0
  const failures: string[] = []

  for (const sc of SCENARIOS) {
    engine.clearHistory() // fresh conversation per scenario (also re-prewarms the system prefix)
    lastError = ''
    let reply = ''
    const t = performance.now()
    if (sc.supersede) {
      // Fire turn 1 and interrupt it mid-generation with turn 2 (the supersede/commit path).
      const first = engine.sendUserMessage(sc.turns[0])
      await new Promise((r) => setTimeout(r, 900))
      reply = await engine.sendUserMessage(sc.turns[1])
      await first.catch(() => undefined)
    } else {
      for (const turn of sc.turns) {
        stats.used = 0
        reply = await engine.sendUserMessage(turn)
      }
    }
    const ms = performance.now() - t

    const problems: string[] = []
    if (lastError) problems.push(`engine error: ${lastError}`)
    if (!reply.trim()) problems.push('empty reply')
    for (const re of GLOBAL_MUST_NOT) if (re.test(reply)) problems.push(`format violation ${re}`)
    const e = sc.expect
    for (const re of e.mustMatch ?? []) if (!re.test(reply)) problems.push(`missing ${re}`)
    for (const re of e.mustNotMatch ?? []) if (re.test(reply)) problems.push(`forbidden ${re}`)
    if (e.maxChars && reply.length > e.maxChars) problems.push(`too long (${reply.length} > ${e.maxChars} chars)`)
    if (e.grounded && stats.used === 0) problems.push('expected grounded (RAG used=0)')
    if (e.ungrounded && stats.used > 0) problems.push(`expected ungrounded (RAG used=${stats.used})`)
    if (sc.supersede) {
      const h = engine.history
      const users = h.filter((m) => m.role === 'user').length
      if (users < 2) problems.push(`supersede lost a user turn (${users} recorded)`)
    }

    const ok = problems.length === 0
    if (ok) passed++
    else failures.push(sc.name)
    log(`[${ok ? 'PASS' : 'FAIL'}] ${sc.name}  (${(ms / 1000).toFixed(1)}s)`)
    log(`   reply: ${reply.replace(/\n/g, ' ').slice(0, 220)}${reply.length > 220 ? '...' : ''}`)
    for (const p of problems) log(`   !! ${p}`)
  }

  const ok = passed === SCENARIOS.length
  log(`\n${ok ? 'EVAL OK' : 'EVAL FAIL'} - ${passed}/${SCENARIOS.length} scenarios passed${ok ? '' : `\nfailed: ${failures.join(' | ')}`}`)
}

run().catch((e) => log(`FATAL: ${(e as Error).message}\n${(e as Error).stack ?? ''}`))
