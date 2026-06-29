// Headless parity gate: the standalone embedder (onnxruntime + @huggingface/tokenizers, embedderNode.ts
// == embedder.ts logic) must be BYTE-EXACT with @huggingface/transformers for bge-small, across single
// + batched (padding) inputs + the query instruction. This guards dropping transformers.js from RAG:
// existing knowledge.bin indexes were built with transformers.js, so vectors must match. Run: npm run verify-embedder
import { pipeline, env } from '@huggingface/transformers'
import { embedMany, embedOne, embedQuery } from '../src/rag/embedderNode.ts'
import { QUERY_INSTRUCTION } from '../src/rag/embedderCore.ts'
import { EMBED } from '../src/models/registry.ts'

env.allowRemoteModels = true
const ref = await pipeline('feature-extraction', EMBED.hfModelId, { dtype: EMBED.dtype as never })

const cos = (a: Float32Array, b: Float32Array): number => {
  let d = 0, na = 0, nb = 0
  for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
  return d / (Math.sqrt(na) * Math.sqrt(nb) + 1e-12)
}
async function refMany(texts: string[]): Promise<Float32Array[]> {
  const out = await ref(texts, { pooling: 'mean', normalize: true })
  const data = out.data as Float32Array
  const dim = EMBED.dim
  return texts.map((_, i) => Float32Array.from(data.subarray(i * dim, (i + 1) * dim)))
}

let worst = 1
const note = (label: string, c: number): void => {
  worst = Math.min(worst, c)
  console.log(`  cos=${c.toFixed(7)}  ${label}`)
}

// 1) singles
console.log('single embeds...')
const singles = ['The capital of Japan is Tokyo.', 'def add(a,b): return a+b', 'café ∑ 日本語 🚀', 'pricing and plans']
for (const t of singles) note(JSON.stringify(t.slice(0, 36)), cos(await embedOne(t), (await refMany([t]))[0]))

// 2) batch (exercises padding to the longest row)
console.log('batched embeds (padding)...')
const batch = ['short', 'a somewhat longer sentence about retrieval augmented generation systems', 'mid length text here', 'x']
const mine = await embedMany(batch)
const refs = await refMany(batch)
for (let i = 0; i < batch.length; i++) note(`batch[${i}] ${JSON.stringify(batch[i].slice(0, 24))}`, cos(mine[i], refs[i]))

// 3) query instruction path (embedQuery must equal ref of the instruction-prefixed text)
console.log('query-instruction path...')
const q = 'how do I inject my own documents'
note('embedQuery', cos(await embedQuery(q), (await refMany([QUERY_INSTRUCTION + q]))[0]))

console.log(`\nworst cosine: ${worst.toFixed(7)} ${worst > 0.9999 ? '-> EMBEDDER PARITY OK' : '-> MISMATCH'}`)
process.exit(worst > 0.9999 ? 0 : 1)
