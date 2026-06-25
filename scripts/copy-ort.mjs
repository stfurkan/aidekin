// Copy the onnxruntime-web wasm that @huggingface/transformers bundles into public/ort/, so we
// self-host it (same-origin) instead of transformers' default dev-tag jsDelivr CDN — which could
// be GC'd and silently break the LLM + RAG. Runs before dev/build (predev/prebuild hooks).
//
// Only the asyncify + plain variants transformers' default actually references are copied; both
// are < Cloudflare's 25 MiB per-file asset limit (the 25 MiB jsep build is NOT used by that
// default, so we skip it).
//
// transformers pins a DEV build of onnxruntime-web that differs from our root 1.27 dependency,
// so npm nests it under the transformers package. We read that nested dist by path (the package
// blocks `exports` resolution of ./package.json) and deliberately do NOT fall back to the root
// onnxruntime-web — a different version's wasm would mismatch the bundled JS glue's ABI.

import { copyFileSync, existsSync, mkdirSync, readdirSync } from 'node:fs'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = fileURLToPath(new URL('..', import.meta.url))
const ortDist = join(root, 'node_modules/@huggingface/transformers/node_modules/onnxruntime-web/dist')
if (!existsSync(ortDist)) {
  console.error(`[copy-ort] bundled onnxruntime-web not found at ${ortDist}; the ORT self-host would break the LLM`)
  process.exit(1)
}

const dest = join(root, 'public/ort')
const WANT = /^ort-wasm-simd-threaded(\.asyncify)?\.(mjs|wasm)$/

mkdirSync(dest, { recursive: true })
let copied = 0
for (const file of readdirSync(ortDist)) {
  if (WANT.test(file)) {
    copyFileSync(join(ortDist, file), join(dest, file))
    copied++
  }
}
if (copied === 0) {
  console.error(`[copy-ort] no matching wasm files in ${ortDist} — the ORT self-host would break the LLM`)
  process.exit(1)
}
console.log(`[copy-ort] copied ${copied} onnxruntime-web wasm file(s) from ${ortDist} -> public/ort/`)
