// Browser storage helpers for the cached model weights. Everything Aidekin caches
// lives in ORIGIN-SCOPED storage:
//   • OPFS dir "aidekin-models"    → all model weights (LLM, ASR/TTS/VAD, Smart-Turn, RAG embedder)
//                                    via our modelStore.getModelAsset
//   • OPFS dir "aidekin-llm-cache" → legacy transformers.js LLM cache (no longer written; swept on clear)
//   • Cache Storage                → no aidekin model files anymore (transformers.js is gone); the
//                                    clear sweep only scrubs any legacy/HTTP-injected stores
//   • IndexedDB                    → our "aidekin" fallback
// In a normal window this persists on disk across restarts; in a private window
// it is ephemeral and wiped when the session closes.
// (LLM "is it cached?" detection: modelStore.hasModelAsset('llm-bonsai-1.7b-q1').)

const OPFS_DIR = 'aidekin-models'

export interface StorageInfo {
  usageBytes: number
  quotaBytes: number
}

/** Outcome of a clear: what was removed, and any failures (surfaced to the user). */
export interface ClearResult {
  ok: boolean
  cleared: string[]
  errors: string[]
}

export async function estimateStorage(): Promise<StorageInfo | null> {
  if (!navigator.storage?.estimate) return null
  const est = await navigator.storage.estimate()
  return { usageBytes: est.usage ?? 0, quotaBytes: est.quota ?? 0 }
}

/** Ask the browser not to evict our storage under disk pressure. Requested
 *  automatically before the first download (see useOrchestrator.load()). */
export async function requestPersist(): Promise<boolean> {
  if (!navigator.storage?.persist) return false
  return navigator.storage.persist()
}

const delay = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms))

type DirWithEntries = FileSystemDirectoryHandle & {
  entries?: () => AsyncIterableIterator<[string, FileSystemHandle]>
}

/** Empty a directory post-order (recurse into subdirs first), then it's removable. */
async function emptyDir(dir: FileSystemDirectoryHandle): Promise<void> {
  const d = dir as DirWithEntries
  if (!d.entries) return
  const children: [string, FileSystemHandle][] = []
  for await (const entry of d.entries()) children.push(entry)
  for (const [name, handle] of children) {
    if (handle.kind === 'directory') await emptyDir(handle as FileSystemDirectoryHandle)
    await dir.removeEntry(name, { recursive: true }).catch(() => undefined)
  }
}

/**
 * Remove an OPFS subtree robustly. Safari only frees a sync-access-handle lock on
 * close() (NOT on worker.terminate()), and the lock can linger a beat - so we retry
 * with backoff, then fall back to manual post-order removal. Throws only if the tree
 * still exists afterwards.
 */
async function removeOpfsDir(parent: FileSystemDirectoryHandle, name: string): Promise<void> {
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      await parent.removeEntry(name, { recursive: true })
      return
    } catch (e) {
      const err = e as DOMException
      if (err.name === 'NotFoundError') return // already gone → success
      if (attempt < 2) {
        await delay(200 * (attempt + 1)) // absorb a briefly lingering WebKit lock
        continue
      }
      // Last resort: empty it by hand, then drop the (now-empty) directory.
      const dir = await parent.getDirectoryHandle(name).catch(() => null)
      if (!dir) return
      await emptyDir(dir)
      await parent.removeEntry(name, { recursive: true }).catch(() => undefined)
    }
  }
  // Structural verification - estimate() is fuzzed/lazy on Safari, so the directory
  // listing is the source of truth.
  const stillThere = await parent.getDirectoryHandle(name).then(
    () => true,
    () => false,
  )
  if (stillThere) throw new Error('files are still locked (a model worker may be open)')
}

function deleteDb(name: string): Promise<void> {
  return new Promise((resolve) => {
    const req = indexedDB.deleteDatabase(name)
    req.onsuccess = () => resolve()
    req.onerror = () => resolve()
    req.onblocked = () => resolve()
  })
}

/**
 * Delete every cached model weight from this browser (re-downloads next run).
 * Returns a structured result - failures are reported, never swallowed. Callers
 * MUST tear down the model workers first (release OPFS locks) before calling this.
 */
export async function clearModelCaches(): Promise<ClearResult> {
  const cleared: string[] = []
  const errors: string[] = []

  // 1) OPFS - sweep EVERY top-level entry (our 'aidekin-models' plus any legacy/orphaned
  //    dir, e.g. a previous build's web-llm 'tvmjs-opfs-store'). Our origin uses OPFS only
  //    for model weights, so clearing all of it is safe and far more thorough than
  //    removing one known dir. There is no class of hidden 'orphaned bytes' to worry about:
  //    every file we write is named in its directory (so it is enumerated here), and an
  //    interrupted write just leaves a named, marker-less partial that this sweep removes
  //    too. A file with a live sync-access handle cannot be unlinked, but callers tear the
  //    workers down (releasing the handles) before calling this, so the sweep completes; if
  //    one is somehow still held it is reported as 'still locked' and reclaimed on the next
  //    load. Nothing is left behind that only a browser site-data delete could reach.
  try {
    const root = await navigator.storage.getDirectory()
    const d = root as DirWithEntries
    const names: string[] = []
    if (d.entries) {
      for await (const [name] of d.entries()) names.push(name)
    } else {
      names.push(OPFS_DIR)
    }
    let removed = 0
    const failed: string[] = []
    for (const name of names) {
      try {
        await removeOpfsDir(root, name)
        removed++
      } catch (e) {
        failed.push(`${name} (${(e as Error).message})`)
      }
    }
    if (removed > 0) cleared.push(`on-device weights (OPFS · ${removed} item${removed === 1 ? '' : 's'})`)
    if (failed.length) errors.push(`OPFS still locked: ${failed.join(', ')}`)
  } catch (e) {
    errors.push(`OPFS: ${(e as Error).message}`)
  }

  // 2) Cache Storage - no aidekin model files live here anymore (LLM + speech + embedder all cache in
  //    OPFS, swept by (1) above). Clear it best-effort to scrub any legacy / HTTP-injected stores.
  try {
    const keys = await caches.keys()
    const results = await Promise.allSettled(keys.map((k) => caches.delete(k)))
    const failed = results.filter((r) => r.status === 'rejected').length
    cleared.push(`model cache (${keys.length} stores)`)
    if (failed > 0) errors.push(`Cache Storage: ${failed}/${keys.length} stores failed`)
  } catch (e) {
    errors.push(`Cache Storage: ${(e as Error).message}`)
  }

  // 3) IndexedDB - our fallback store (+ any leftover bookkeeping DBs).
  try {
    const idb = indexedDB as IDBFactory & { databases?: () => Promise<{ name?: string }[]> }
    const found = idb.databases ? (await idb.databases()).map((d) => d.name).filter(Boolean) : []
    const names = new Set<string>([...(found as string[]), 'aidekin'])
    await Promise.all([...names].map((n) => deleteDb(n)))
    cleared.push('bookkeeping (IndexedDB)')
  } catch (e) {
    errors.push(`IndexedDB: ${(e as Error).message}`)
  }

  // 4) Named storage buckets (Chrome StorageBuckets API) - a library or earlier build
  //    could have parked OPFS data in a non-default bucket, which the default-root sweep
  //    in (1) can't see. Best-effort; the API is Chromium-only.
  try {
    const sb = (navigator as Navigator & {
      storageBuckets?: { keys(): Promise<string[]>; delete(name: string): Promise<void> }
    }).storageBuckets
    if (sb?.keys) {
      const names = await sb.keys()
      const results = await Promise.allSettled(names.map((n) => sb.delete(n)))
      const ok = results.filter((r) => r.status === 'fulfilled').length
      if (ok > 0) cleared.push(`storage buckets (${ok})`)
      const failed = results.length - ok
      if (failed > 0) errors.push(`Storage buckets: ${failed}/${results.length} failed`)
    }
  } catch (e) {
    errors.push(`Storage buckets: ${(e as Error).message}`)
  }

  return { ok: errors.length === 0, cleared, errors }
}
