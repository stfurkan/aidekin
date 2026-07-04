// Node (CLI) embedder: same bge-small model + tokenize/pool as the browser embedder.ts, but on
// onnxruntime-node (native) and a disk cache. Used by the knowledge builder + verifier so the CLI
// index is identical to what the browser produces. Shares embedderCore.ts -> identical vectors.
// Verified byte-exact vs transformers.js in scripts/verify-embedder.ts.
import * as ort from 'onnxruntime-node'
import { Tokenizer } from '@huggingface/tokenizers'
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { EMBED, embedModelUrls } from '../models/registry'
import { QUERY_INSTRUCTION, poolAndNormalize, tokenizeBatch } from './embedderCore'

const CACHE = join(process.cwd(), 'node_modules', '.cache', 'aidekin-embed')
async function cached(url: string, name: string): Promise<Buffer> {
  if (!existsSync(CACHE)) mkdirSync(CACHE, { recursive: true })
  const p = join(CACHE, name)
  if (existsSync(p)) return readFileSync(p)
  const buf = Buffer.from(await (await fetch(url)).arrayBuffer())
  writeFileSync(p, buf)
  return buf
}

interface Embedder {
  tok: Tokenizer
  session: ort.InferenceSession
}
let promise: Promise<Embedder> | null = null
async function load(): Promise<Embedder> {
  if (!promise) {
    promise = (async () => {
      const urls = embedModelUrls()
      const [tokJson, tokCfg, onnx] = await Promise.all([
        cached(urls.tokenizerJsonUrl, 'tokenizer.json').then((b) => JSON.parse(b.toString('utf8')) as Record<string, unknown>),
        cached(urls.tokenizerConfigUrl, 'tokenizer_config.json').then((b) => JSON.parse(b.toString('utf8')) as Record<string, unknown>),
        cached(urls.onnxUrl, 'model_quantized.onnx'),
      ])
      const tok = new Tokenizer(tokJson, tokCfg)
      const session = await ort.InferenceSession.create(onnx)
      return { tok, session }
    })()
  }
  return promise
}

const big = (rows: number[][]): BigInt64Array => {
  const seq = rows[0]?.length ?? 0
  const out = new BigInt64Array(rows.length * seq)
  for (let b = 0; b < rows.length; b++) for (let t = 0; t < seq; t++) out[b * seq + t] = BigInt(rows[b][t])
  return out
}

export async function embedMany(texts: string[]): Promise<Float32Array[]> {
  if (texts.length === 0) return []
  const { tok, session } = await load()
  const { ids, masks, batch, seq } = tokenizeBatch(tok, texts)
  const dims: [number, number] = [batch, seq]
  const feeds: Record<string, ort.Tensor> = {
    input_ids: new ort.Tensor('int64', big(ids), dims),
    attention_mask: new ort.Tensor('int64', big(masks), dims),
  }
  if (session.inputNames.includes('token_type_ids')) {
    feeds.token_type_ids = new ort.Tensor('int64', new BigInt64Array(batch * seq), dims)
  }
  const res = await session.run(feeds)
  const outName = session.outputNames.find((n) => /hidden|last|output/i.test(n)) ?? session.outputNames[0]
  const hidden = res[outName].data as Float32Array
  return poolAndNormalize(hidden, masks, batch, seq, EMBED.dim)
}

export async function embedOne(text: string): Promise<Float32Array> {
  return (await embedMany([text]))[0]
}

export async function embedQuery(text: string): Promise<Float32Array> {
  return embedOne(QUERY_INSTRUCTION + text)
}
