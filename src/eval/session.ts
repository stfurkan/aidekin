// Session-persistence verification: proves the bitgpu chat.save/restore path end to end through the
// REAL product stack (ConversationEngine -> llm.worker -> bitgpu), which the behavioral eval does NOT
// cover (it never sets persistKey). Served by the vite DEV server only (not a build input); driven
// headlessly by scripts/eval-session.mjs, which additionally captures the worker's console to confirm
// the snapshot path (vs a plain re-prefill).
//
// Three phases, each with its OWN worker (a fresh ConversationEngine = a simulated page reload; the KV
// snapshot lives in IndexedDB and the plain transcript in localStorage, both surviving across engines
// in the same origin):
//   1. Establish a fact ("my favorite color is teal"), let the snapshot save.
//   2. Reload: a new engine restores the snapshot -> the follow-up must RECALL the fact AND run as a
//      cache-append (reusedCache), i.e. the conversation continued with no cold re-prefill.
//   3. New conversation (clearHistory deletes the snapshot), then reload again: the fact must be GONE.
import { ConversationEngine } from '@/engine/conversationEngine'
import { resolveSystemPrompt } from '@/widget/protocol'

const out = document.getElementById('out')!
const log = (s: string): void => {
  out.textContent += s + '\n'
}
const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms))

const PERSIST_KEY = 'aidekin:eval:session'
const SEED = 42

// No retriever: the snapshot mechanism is independent of RAG, and dropping it keeps the system prompt
// identical across the three engines (so the restored cache is a clean append) and the test fast.
function makeEngine(): ConversationEngine {
  return new ConversationEngine({
    systemPrompt: resolveSystemPrompt({ title: 'aidekin' }),
    brandName: 'aidekin',
    samplerSeed: SEED,
    persistKey: PERSIST_KEY,
  })
}

async function run(): Promise<void> {
  out.textContent = ''
  // Start from a clean slate so a previous run's snapshot/history can't mask a regression. Await the
  // IndexedDB drop so worker A can't race a stale snapshot from an earlier run.
  try {
    localStorage.removeItem(PERSIST_KEY)
    await new Promise<void>((resolve) => {
      const req = indexedDB.deleteDatabase('aidekin-sessions')
      req.onsuccess = req.onerror = req.onblocked = (): void => resolve()
    })
  } catch {
    /* best-effort */
  }

  let passed = true
  const check = (name: string, ok: boolean, detail: string): void => {
    log(`[${ok ? 'PASS' : 'FAIL'}] ${name}  ${detail}`)
    if (!ok) passed = false
  }

  try {
    // ── Phase 1: establish the fact, let the snapshot save ────────────────────
    log('phase 1: establish fact (worker A)...')
    const a = makeEngine()
    await a.loadLlm()
    await a.sendUserMessage('My favorite color is teal. Please remember that.')
    await a.sendUserMessage('Thanks!')
    await sleep(2500) // let the post-turn chat.save() finish its GPU readback + IndexedDB write
    a.dispose() // terminate worker A (frees the GPU device + chat); the snapshot stays in IndexedDB
    await sleep(500)

    // ── Phase 2: reload -> restore -> recall + cache-append ───────────────────
    log('phase 2: reload + restore (worker B)...')
    const b = makeEngine()
    await b.loadLlm() // worker B restores the snapshot during init
    const r2 = await b.sendUserMessage('What is my favorite color?')
    const reused2 = b.lastGenStats?.reusedCache === true
    check('restored conversation recalls the fact', /teal/i.test(r2), `reply: ${r2.slice(0, 90)}`)
    check('post-reload turn is a cache-append (no cold re-prefill)', reused2, `reusedCache=${reused2}`)
    b.clearHistory() // new conversation: deletes the snapshot + localStorage, resets the worker chat
    await sleep(800)
    b.dispose()
    await sleep(500)

    // ── Phase 3: reload after "new chat" -> the fact is gone ───────────────────
    log('phase 3: reload after new-chat (worker C)...')
    const c = makeEngine()
    await c.loadLlm() // nothing to restore: the snapshot was deleted in phase 2
    const r3 = await c.sendUserMessage('What is my favorite color?')
    check('new-conversation reload does NOT recall the cleared fact', !/teal/i.test(r3), `reply: ${r3.slice(0, 90)}`)
    c.dispose()
    await sleep(500)

    // ── Phase 4: a prompt that overruns the window -> onOverflow trims + retries ──
    // Seed a long transcript into localStorage so the first turn full-prefills a prompt WELL past the
    // 2048 window; the worker's onOverflow must trim the oldest turns and retry, so the turn still
    // completes instead of throwing. (The runner also confirms the worker logged the overflow trim.)
    log('phase 4: window overflow -> onOverflow trim (worker D)...')
    const overflowKey = 'aidekin:eval:overflow'
    try {
      localStorage.removeItem(overflowKey)
      await new Promise<void>((resolve) => {
        const req = indexedDB.deleteDatabase('aidekin-sessions')
        req.onsuccess = req.onerror = req.onblocked = (): void => resolve()
      })
    } catch {
      /* best-effort */
    }
    // ~10 exchanges of filler (~1.3k chars each) = well over 2048 tokens once rendered.
    const filler = 'This is a long stretch of earlier conversation used only to fill the context window past its limit so the overflow path is exercised. '.repeat(9)
    const longHistory: { role: string; content: string }[] = []
    for (let i = 0; i < 10; i++) {
      longHistory.push({ role: 'user', content: `Question number ${i}: ${filler}` })
      longHistory.push({ role: 'assistant', content: `Answer number ${i}: ${filler}` })
    }
    localStorage.setItem(overflowKey, JSON.stringify(longHistory))
    const d = new ConversationEngine({
      systemPrompt: resolveSystemPrompt({ title: 'aidekin' }),
      brandName: 'aidekin',
      samplerSeed: SEED,
      persistKey: overflowKey,
    })
    await d.loadLlm()
    const r4 = await d.sendUserMessage('Ignoring all of the above, what is two plus two? Answer in one word.')
    check('overflowing prompt still completes (onOverflow trimmed + retried)', r4.trim().length > 0, `reply: ${r4.slice(0, 90)}`)
    localStorage.removeItem(overflowKey)
    d.dispose()
  } catch (e) {
    passed = false
    log(`FATAL: ${(e as Error).message}`)
  }

  log(`\n${passed ? 'SESSION OK' : 'SESSION FAIL'}`)
}

void run()
