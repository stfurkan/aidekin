// OPFS-backed cache for transformers.js model files.
//
// WHY: the default browser Cache Storage chokes on a single very large entry - the ~290 MB
// LLM weights fail with "Failed to execute 'put' on 'Cache': Unexpected internal error", so
// they never persist and re-download on every visit (breaking the "download once, works
// offline" promise). OPFS handles large files well - it is what we already use for the ASR/TTS
// weights (see modelStore.ts) - and works inside workers on Safari via sync access handles. We
// wire this in as transformers.js `env.customCache` (the CacheInterface: match/put) so the LLM
// weights land in OPFS instead of Cache Storage.
//
// SAFETY: every operation is best-effort and never throws.
//   • match() returns undefined on any error → transformers.js downloads from the network.
//   • put() swallows errors → the model still loads from the in-memory buffer it already holds.
// So the worst case equals today's behaviour (re-download); it can only improve. A `.done`
// marker records the verified byte size, so a write interrupted by a crash is treated as a
// miss rather than served truncated.

const DIR = 'aidekin-llm-cache'
const MARKER = '.done'
const sanitize = (key: string): string => key.replace(/[^a-zA-Z0-9._-]/g, '_')

// transformers.js calls match() on the same file more than once per load (an upfront
// metadata pre-scan, then the real session build), so the cache-hit line would log 2x for
// any file in both paths. Log once per file per worker session; a reload re-spawns the
// worker and resets this, so a fresh load still logs.
const loggedHits = new Set<string>()

type SyncAccessHandle = {
  read(buf: BufferSource, opts?: { at?: number }): number
  write(buf: BufferSource, opts?: { at?: number }): number
  truncate(n: number): void
  getSize(): number
  flush(): void
  close(): void
}
type SyncCapableFileHandle = FileSystemFileHandle & {
  createSyncAccessHandle?: () => Promise<SyncAccessHandle>
}

async function cacheDir(): Promise<FileSystemDirectoryHandle | null> {
  try {
    const storage = navigator.storage as StorageManager & {
      getDirectory?: () => Promise<FileSystemDirectoryHandle>
    }
    if (!storage?.getDirectory) return null
    const root = await storage.getDirectory()
    return await root.getDirectoryHandle(DIR, { create: true })
  } catch {
    return null
  }
}

/** The verified byte size recorded for `name`, or null if it never completed. */
async function markerSize(dir: FileSystemDirectoryHandle, name: string): Promise<number | null> {
  try {
    const h = await dir.getFileHandle(name + MARKER)
    const n = Number((await (await h.getFile()).text()).trim())
    return Number.isFinite(n) && n > 0 ? n : null
  } catch {
    return null
  }
}

async function match(request: string): Promise<Response | undefined> {
  try {
    const dir = await cacheDir()
    if (!dir) return undefined
    const name = sanitize(request)
    const expected = await markerSize(dir, name)
    if (expected === null) return undefined // never completed, so a miss and re-download
    // Set Content-Length on the cache hit. Without it, transformers.js logs "Unable to
    // determine content-length" even though this is served from disk (not the network), and
    // it can't show load progress. With it: clean log + a real progress bar on cached loads.
    const headers = { 'Content-Length': String(expected) }
    const hit = (bytes: number): void => {
      if (bytes > 10_000_000 && !loggedHits.has(name)) {
        loggedHits.add(name)
        console.info(`[aidekin] LLM weight served from OPFS cache: ${Math.round(bytes / 1048576)} MB (no download)`)
      }
    }
    const handle = (await dir.getFileHandle(name)) as SyncCapableFileHandle
    if (!handle.createSyncAccessHandle) {
      const file = await handle.getFile()
      if (file.size !== expected) return undefined
      hit(expected)
      return new Response(file, { headers })
    }
    const access = await handle.createSyncAccessHandle()
    try {
      const size = access.getSize()
      if (size !== expected) return undefined // truncated or mismatched, so a miss
      const buf = new ArrayBuffer(size)
      access.read(new Uint8Array(buf), { at: 0 })
      hit(expected)
      return new Response(buf, { headers })
    } finally {
      access.close()
    }
  } catch {
    return undefined
  }
}

async function put(request: string, response: Response): Promise<void> {
  // The Response here is a throwaway wrapper transformers.js builds around the already-loaded
  // buffer, so consuming its body is safe. Stream it to OPFS in chunks (no extra full copy).
  try {
    const dir = await cacheDir()
    if (!dir || !response.body) return
    const name = sanitize(request)
    const handle = (await dir.getFileHandle(name, { create: true })) as SyncCapableFileHandle
    if (!handle.createSyncAccessHandle) return // createWritable isn't supported on Safari

    // Drop any stale marker first, so a crash mid-write can't be mistaken for complete.
    await dir.removeEntry(name + MARKER).catch(() => {})

    const access = await handle.createSyncAccessHandle()
    let size = 0
    try {
      access.truncate(0)
      const reader = response.body.getReader()
      for (;;) {
        const { done, value } = await reader.read()
        if (done) break
        if (value && value.byteLength) {
          access.write(value, { at: size })
          size += value.byteLength
        }
      }
      access.flush()
    } finally {
      access.close()
    }

    // Mark complete only after a clean write, recording the size for verification on read.
    const mh = (await dir.getFileHandle(name + MARKER, { create: true })) as SyncCapableFileHandle
    if (mh.createSyncAccessHandle) {
      const a = await mh.createSyncAccessHandle()
      try {
        a.truncate(0)
        a.write(new TextEncoder().encode(String(size)), { at: 0 })
        a.flush()
      } finally {
        a.close()
      }
    }
  } catch {
    /* best-effort: transformers.js keeps the in-memory model; we just don't persist it */
  }
}

interface CustomCacheEnv {
  useCustomCache: boolean
  customCache: unknown
}

/**
 * Route transformers.js model-file caching through OPFS instead of Cache Storage. Call once,
 * inside a worker (OPFS sync access handles require a worker context), before loading a model.
 * Pass the transformers.js `env` singleton.
 */
export function installOpfsModelCache(env: CustomCacheEnv): void {
  try {
    env.useCustomCache = true
    env.customCache = { match, put }
  } catch {
    /* leave the default cache in place */
  }
}

/**
 * True if at least one LLM weight file is fully cached in OPFS (a `.done` marker present),
 * so a repeat visit can show "Loading" instead of "Downloading". Read-only - does NOT create
 * the directory, and works on the main thread (no sync access handle needed for this check).
 */
export async function hasLlmCache(): Promise<boolean> {
  try {
    const storage = navigator.storage as StorageManager & {
      getDirectory?: () => Promise<FileSystemDirectoryHandle>
    }
    if (!storage?.getDirectory) return false
    const root = await storage.getDirectory()
    const dir = (await root.getDirectoryHandle(DIR).catch(() => null)) as
      | (FileSystemDirectoryHandle & { entries?: () => AsyncIterableIterator<[string, FileSystemHandle]> })
      | null
    if (!dir?.entries) return false
    for await (const [name] of dir.entries()) if (name.endsWith(MARKER)) return true
    return false
  } catch {
    return false
  }
}
