// Info-level logging, gated OFF in production: a customer's site console stays clean. But with
// zero telemetry these logs are the ONLY diagnostics, so they stay one switch away for a bug
// report: run localStorage.setItem('aidekin:debug', '1') in the console (on the embedding page)
// and reload. Dev builds always log. Warnings and errors are NOT gated - use console.warn/error
// directly for anything that indicates a real problem.
//
// Workers cannot read localStorage: they receive the flag in their init message and call
// setDebug(), so one switch covers the whole pipeline.
let enabled = ((): boolean => {
  if (import.meta.env?.DEV) return true
  try {
    return globalThis.localStorage?.getItem('aidekin:debug') === '1'
  } catch {
    return false
  }
})()

export const debugEnabled = (): boolean => enabled
export const setDebug = (on: boolean): void => {
  enabled = enabled || on // never turn dev logging off, only enable
}
export const dlog = (...args: unknown[]): void => {
  if (enabled) console.info(...args)
}
