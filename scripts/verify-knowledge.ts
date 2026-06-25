// Parity guard for a knowledge.bin: re-embed each chunk's text and confirm it
// self-retrieves as the top hit with a high cosine score. This exercises the full
// embed → quantize → serialize → parse → search round-trip with the SAME code the
// browser uses, catching format/quantization regressions.
//
//   npx tsx scripts/verify-knowledge.ts ./public/knowledge.bin

import { readFile } from 'node:fs/promises'
import { embedOne } from '../src/rag/embedder.ts'
import { VectorStore } from '../src/rag/store.ts'

const MIN_SCORE = 0.9

async function main(): Promise<void> {
  const path = process.argv[2] ?? 'knowledge.bin'
  const file = await readFile(path)
  const store = VectorStore.fromBytes(file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength))
  console.log(`Loaded ${path}: ${store.count} chunks, model ${store.header.modelId}, dim ${store.header.dim}.`)
  if (store.count === 0) {
    console.error('Index is empty (0 chunks).')
    process.exit(1)
  }

  const sampleN = Math.min(12, store.count)
  const step = Math.max(1, Math.floor(store.count / sampleN))
  let pass = 0
  let checked = 0
  for (let i = 0; i < store.count && checked < sampleN; i += step) {
    const text = store.textAt(i)
    const q = await embedOne(text)
    const top = store.search(q, 1)[0]
    const ok = top?.text === text && top.score >= MIN_SCORE
    if (ok) pass++
    else console.warn(`  ✗ chunk ${i} self-retrieve failed (score=${top?.score?.toFixed(3) ?? 'n/a'})`)
    checked++
  }
  console.log(`Parity: ${pass}/${checked} sampled chunks self-retrieve as top-1 (score ≥ ${MIN_SCORE}).`)
  if (pass < checked) {
    console.error('FAILED - embedder/format mismatch.')
    process.exit(1)
  }
  console.log('OK.')
}

main().catch((e) => {
  console.error(e)
  process.exit(1)
})
