/** Thrown when WebGPU is unavailable: no `navigator.gpu` (unsupported browser, or a
 *  non-secure context) or no adapter could be acquired. Catch this to render a
 *  "your browser doesn't support WebGPU yet" fallback instead of crashing. */
export class WebGPUUnavailableError extends Error {
  override readonly name = 'WebGPUUnavailableError'
  constructor(message: string) {
    super(message)
  }
}
