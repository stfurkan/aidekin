// Same-origin path for the onnxruntime-web wasm that transformers.js (the LLM + RAG embedder)
// loads. By default transformers points wasmPaths at a DEV-TAG jsDelivr URL
// (onnxruntime-web@1.26.0-dev…), which could be garbage-collected and silently break the brain
// + RAG in every visitor's browser. We self-host the exact bundled wasm under /ort/ (copied at
// build by scripts/copy-ort.mjs) and point wasmPaths there instead.
//
// We mirror transformers' own default object EXACTLY (asyncify variant elsewhere, plain on
// Safari) so the same wasm loads as in working production — just from our origin. transformers'
// real Safari check reads navigator.vendor, which is undefined in workers, so we approximate it
// from the user-agent (Safari but no Chromium / iOS-wrapper marker).

export function selfHostedOrtWasmPaths(): { mjs: string; wasm: string } {
  const base = '/ort/'
  const ua = typeof navigator !== 'undefined' ? navigator.userAgent : ''
  const isSafari = /safari/i.test(ua) && !/chrome|chromium|crios|fxios|edg|opr|android/i.test(ua)
  return isSafari
    ? { mjs: `${base}ort-wasm-simd-threaded.mjs`, wasm: `${base}ort-wasm-simd-threaded.wasm` }
    : { mjs: `${base}ort-wasm-simd-threaded.asyncify.mjs`, wasm: `${base}ort-wasm-simd-threaded.asyncify.wasm` }
}
