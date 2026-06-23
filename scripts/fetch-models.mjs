// Download model weights from the Hugging Face Hub into public/models/<role>/.
// Usage:  node scripts/fetch-models.mjs [asr|tts|turn|all]
//
// This is OPTIONAL and DEV-ONLY. The app streams every weight straight from the HF
// CDN and caches it in the browser (OPFS) — production and dev behave identically and
// ship NO weights. Use this only to (a) serve models same-origin for offline dev
// (VITE_MODEL_CDN=/models), or (b) feed the headless ASR test (FP16_DIR=public/models/asr
// npm run asr-test). public/models is gitignored. Files skip if already present.

import { createWriteStream } from 'node:fs'
import { mkdir, stat } from 'node:fs/promises'
import { dirname, join } from 'node:path'
import { Readable } from 'node:stream'
import { pipeline } from 'node:stream/promises'

const HF = 'https://huggingface.co'
const OUT_ROOT = 'public/models'

/** @type {Record<string, { repo: string, dir: string, files: string[] }>} */
const SETS = {
  asr: {
    repo: 'soniqo/Nemotron-3.5-ASR-Streaming-Multilingual-0.6B-ONNX-FP16',
    dir: 'asr',
    files: [
      'encoder.onnx',
      'encoder.onnx.data',
      'decoder.onnx',
      'decoder.onnx.data',
      'joint.onnx',
      'joint.onnx.data',
      'config.json',
      'vocab.json',
      'languages.json',
    ],
  },
  turn: {
    repo: 'onnx-community/smart-turn-v3-ONNX',
    dir: 'turn',
    files: ['onnx/model_quantized.onnx', 'config.json', 'preprocessor_config.json'],
  },
  tts: {
    repo: 'Supertone/supertonic-3',
    dir: 'tts',
    files: [
      'onnx/duration_predictor.onnx',
      'onnx/text_encoder.onnx',
      'onnx/vector_estimator.onnx',
      'onnx/vocoder.onnx',
      'onnx/tts.json',
      'onnx/unicode_indexer.json',
      'config.json',
      'voice_styles/F1.json',
      'voice_styles/M1.json',
    ],
  },
}

async function fileExists(path) {
  try {
    const s = await stat(path)
    return s.size > 0 ? s.size : 0
  } catch {
    return 0
  }
}

async function download(repo, file, destDir) {
  const dest = join(destDir, file)
  await mkdir(dirname(dest), { recursive: true })
  const existing = await fileExists(dest)
  if (existing) {
    console.log(`  skip  ${file}  (${(existing / 1e6).toFixed(1)} MB)`)
    return
  }
  const url = `${HF}/${repo}/resolve/main/${file}?download=true`
  process.stdout.write(`  get   ${file}  …`)
  const res = await fetch(url, { redirect: 'follow' })
  if (!res.ok || !res.body) throw new Error(`${file}: HTTP ${res.status}`)
  await pipeline(Readable.fromWeb(res.body), createWriteStream(dest))
  const size = await fileExists(dest)
  process.stdout.write(`\r  done  ${file}  (${(size / 1e6).toFixed(1)} MB)\n`)
}

const arg = (process.argv[2] || 'all').toLowerCase()
const wanted = arg === 'all' ? Object.keys(SETS) : [arg]

for (const key of wanted) {
  const set = SETS[key]
  if (!set) {
    console.error(`unknown set "${key}" (have: ${Object.keys(SETS).join(', ')}, all)`)
    process.exit(1)
  }
  console.log(`\n[${key}] ${set.repo} → ${OUT_ROOT}/${set.dir}`)
  for (const f of set.files) await download(set.repo, f, join(OUT_ROOT, set.dir))
}
console.log('\nfetch-models complete')
