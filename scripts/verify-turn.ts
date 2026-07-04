// Headless verdict-parity gate for the standalone Whisper feature extractor (turnFeatures.ts) vs
// transformers.js AutoProcessor. Smart Turn is a binary classifier, so the bar is: feed BOTH feature
// paths through the REAL Smart Turn ONNX and assert the end-of-turn probabilities agree across varied
// audio. If they do, we can drop transformers.js from the turn worker. Run: npm run verify-turn
import { AutoProcessor, env } from '@huggingface/transformers'
import * as ort from 'onnxruntime-node'
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { whisperFeatures } from '../src/workers/turnFeatures.ts'
import { TURN } from '../src/models/registry.ts'

env.allowRemoteModels = true
const proc = await AutoProcessor.from_pretrained(TURN.hfModelId)
const CACHE = '/private/tmp/claude-501/-Users-sft-Desktop-sonari/a5d44c61-e66c-4947-9db8-b198d15f0221/scratchpad/turn-v.onnx'
if (!existsSync(CACHE)) {
  mkdirSync('/private/tmp/claude-501/-Users-sft-Desktop-sonari/a5d44c61-e66c-4947-9db8-b198d15f0221/scratchpad', { recursive: true })
  writeFileSync(CACHE, Buffer.from(await (await fetch(`https://huggingface.co/${TURN.hfModelId}/resolve/main/onnx/model_quantized.onnx`)).arrayBuffer()))
}
const session = await ort.InferenceSession.create(readFileSync(CACHE))
const sig = (x: number): number => 1 / (1 + Math.exp(-x))

async function refFeatures(audio: Float32Array): Promise<Float32Array> {
  const out = (await (proc as unknown as (a: Float32Array) => Promise<{ input_features: { data: Float32Array; dims: number[] } }>)(audio)).input_features
  const frames = out.dims[out.dims.length - 1]
  if (frames === 800) return out.data
  const fixed = new Float32Array(80 * 800) // pinFeatures: right-pad/truncate to 800
  const copy = Math.min(frames, 800)
  for (let m = 0; m < 80; m++) for (let f = 0; f < copy; f++) fixed[m * 800 + f] = out.data[m * frames + f]
  return fixed
}
async function runProb(feat: Float32Array): Promise<number> {
  const o = await session.run({ input_features: new ort.Tensor('float32', Float32Array.from(feat), [1, 80, 800]) })
  return sig(Number((o.logits ?? o[session.outputNames[0]]).data[0]))
}
const cos = (a: Float32Array, b: Float32Array): number => {
  let d = 0, na = 0, nb = 0
  for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
  return d / (Math.sqrt(na) * Math.sqrt(nb) + 1e-12)
}

// varied clips (deterministic): different lengths + content
let seed = 7
const rnd = (): number => (seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff - 0.5
function clip(len: number, kind: string): Float32Array {
  const a = new Float32Array(len)
  for (let i = 0; i < len; i++) {
    if (kind === 'sine') a[i] = 0.4 * Math.sin(i * 0.06)
    else if (kind === 'noise') a[i] = 0.5 * rnd()
    else if (kind === 'chirp') a[i] = 0.4 * Math.sin(i * i * 1e-7)
    else if (kind === 'speech') a[i] = 0.3 * Math.sin(i * 0.05) + 0.15 * rnd() + 0.1 * Math.sin(i * 0.013)
    else a[i] = 0 // silence
  }
  return a
}
const cases: { len: number; kind: string }[] = []
for (const kind of ['sine', 'noise', 'chirp', 'speech', 'silence']) for (const len of [128000, 96000, 48000, 16000, 4000]) cases.push({ len, kind })

let worstCos = 1, worstDp = 0, fails = 0
for (const { len, kind } of cases) {
  const audio = clip(len, kind)
  const rf = await refFeatures(audio)
  const mf = whisperFeatures(audio)
  const c = cos(rf, mf)
  const pr = await runProb(rf), pm = await runProb(mf)
  const dp = Math.abs(pr - pm)
  worstCos = Math.min(worstCos, c)
  worstDp = Math.max(worstDp, dp)
  const ok = dp < 0.01
  if (!ok) fails++
  console.log(`  ${ok ? '[PASS]' : '[FAIL]'} ${kind.padEnd(8)} len=${String(len).padStart(6)}  featCos=${c.toFixed(5)}  p_ref=${pr.toFixed(4)} p_mine=${pm.toFixed(4)} dp=${dp.toFixed(4)}`)
}
console.log(`\nworst featCosine=${worstCos.toFixed(5)}  worst |dprob|=${worstDp.toFixed(5)}  ${fails === 0 ? '-> TURN PARITY OK' : fails + ' FAIL(S)'}`)
process.exit(fails === 0 ? 0 : 1)
