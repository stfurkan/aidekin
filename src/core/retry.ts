// Retry an async operation with exponential backoff + full jitter. Used for model-weight
// fetches, which can hit transient CDN resets / rate limits (Hugging Face Xet, jsDelivr) that
// a single retry usually clears. Full jitter (delay = random in [0, cap], cap doubling each
// attempt) is the AWS-recommended backoff: it spreads concurrent retries so they don't thunder.
// An error carrying `permanent: true` (set where the failure is KNOWN definitive, e.g. an HTTP
// 4xx) is rethrown immediately - backoff would only delay the user-visible error.

export interface RetryOptions {
  /** Max retries AFTER the first attempt (default 3 → up to 4 total tries). */
  retries?: number
  /** First backoff ceiling in ms; doubles each attempt (default 800). */
  baseMs?: number
  /** Per-delay cap in ms (default 8000). */
  maxMs?: number
  /** Return false to stop retrying a given error (e.g. quota, a 404). Default: always retry. */
  shouldRetry?: (err: unknown) => boolean
  /** Notified before each backoff wait, for logging. */
  onRetry?: (attempt: number, err: unknown, delayMs: number) => void
}

const sleep = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms))

export async function withRetry<T>(fn: () => Promise<T>, opts: RetryOptions = {}): Promise<T> {
  const retries = opts.retries ?? 3
  const baseMs = opts.baseMs ?? 800
  const maxMs = opts.maxMs ?? 8000
  const shouldRetry = opts.shouldRetry ?? (() => true)
  let lastErr: unknown
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      return await fn()
    } catch (err) {
      lastErr = err
      const permanent = (err as { permanent?: unknown } | null)?.permanent === true
      if (attempt === retries || permanent || !shouldRetry(err)) break
      const ceiling = Math.min(maxMs, baseMs * 2 ** attempt)
      const delay = Math.floor(Math.random() * ceiling)
      opts.onRetry?.(attempt + 1, err, delay)
      await sleep(delay)
    }
  }
  throw lastErr
}
