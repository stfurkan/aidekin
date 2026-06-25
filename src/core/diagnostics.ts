// Crash diagnostics that SURVIVE a page reload. A mysterious "the page refreshed
// on click" is almost always an uncaught error or a renderer/GPU crash - neither of
// which is visible after the reload wipes React state. We persist the last error to
// localStorage and mark "loading in progress" in sessionStorage, so on the next load
// we can tell the user the page died mid-load and show what killed it.

export interface CapturedError {
  message: string
  where?: string
  stack?: string
  time: number
}

const ERR_KEY = 'aidekin:last-error'

export function installErrorCapture(): void {
  window.addEventListener('error', (e: ErrorEvent) => {
    persist({
      message: e.message || 'Uncaught error',
      where: e.filename ? `${e.filename}:${e.lineno}:${e.colno}` : undefined,
      stack: e.error instanceof Error ? e.error.stack : undefined,
      time: Date.now(),
    })
  })
  window.addEventListener('unhandledrejection', (e: PromiseRejectionEvent) => {
    const r: unknown = e.reason
    persist({
      message: r instanceof Error ? `Unhandled rejection: ${r.message}` : `Unhandled rejection: ${String(r)}`,
      stack: r instanceof Error ? r.stack : undefined,
      time: Date.now(),
    })
  })
}

function persist(err: CapturedError): void {
  try {
    localStorage.setItem(ERR_KEY, JSON.stringify(err))
  } catch {
    /* storage blocked */
  }
  console.error('[Aidekin] captured error:', err)
}
