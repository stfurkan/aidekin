// Small runtime/UA helpers (safe to import in workers).

/** True on Safari / WebKit (Safari in the UA, but not a Chromium-based browser). */
export function isWebKit(): boolean {
  const ua = navigator.userAgent
  return /Safari/.test(ua) && !/Chrome|Chromium|Android|Edg|OPR/.test(ua)
}

/**
 * onnxruntime-web thread count.
 *
 * WebKit's JSEP (the WebGPU-over-WASM path) balloons CPU/memory with many threads
 * and contributes to memory-pressure crashes, so JSEP-adjacent workers are capped
 * hard on Safari. `heavyCompute` workers (the ASR encoder, pure WASM with no JSEP)
 * don't have that problem, so they get more threads for acceptable latency.
 */
export function wasmThreads(opts: { heavyCompute?: boolean } = {}): number {
  const cores = navigator.hardwareConcurrency || 4
  if (isWebKit()) return Math.min(opts.heavyCompute ? 6 : 2, cores)
  return Math.max(1, Math.min(cores, 8))
}
