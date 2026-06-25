// OPFS-first model-weight cache. Large weights (the ASR encoder data alone is
// ~690 MB) are STREAMED chunk-by-chunk straight to disk - never accumulated in the
// JS heap - which avoids the ~2×-per-file memory spike that crashes Safari's
// WebContent process. Writes use OPFS createSyncAccessHandle (worker-only, and the
// path Safari actually supports), falling back to an in-memory fetch where OPFS is
// unavailable.
//
// COMPLETENESS: a download is only considered "cached" once it has fully arrived
// AND its byte length matches Content-Length. We record that by writing a tiny
// `<key>.done` marker (holding the verified size) ONLY after a successful stream.
// A file with no marker - or whose size no longer matches - is treated as absent
// and re-downloaded. Without this, a partially-written .onnx.data from an
// interrupted run is silently handed to ORT and fails with "Out of bounds".

import { withRetry } from './retry'

export interface FetchProgress {
  readonly loaded: number
  readonly total: number
}
export type ProgressFn = (p: FetchProgress) => void

const OPFS_DIR = 'aidekin-models'
const MARKER_SUFFIX = '.done'
const sanitize = (key: string): string => key.replace(/[^a-zA-Z0-9._-]/g, '_')

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

async function opfsDir(): Promise<FileSystemDirectoryHandle | null> {
  try {
    const storage = navigator.storage as StorageManager & {
      getDirectory?: () => Promise<FileSystemDirectoryHandle>
    }
    if (!storage?.getDirectory) return null
    const root = await storage.getDirectory()
    return await root.getDirectoryHandle(OPFS_DIR, { create: true })
  } catch {
    return null
  }
}

/** Read the completion marker for `key` → the verified byte size, or null if absent. */
async function readMarker(dir: FileSystemDirectoryHandle, key: string): Promise<number | null> {
  try {
    const h = await dir.getFileHandle(sanitize(key) + MARKER_SUFFIX)
    const n = Number((await (await h.getFile()).text()).trim())
    return Number.isFinite(n) && n > 0 ? n : null
  } catch {
    return null
  }
}

/** Record that `key` is fully downloaded (size bytes). Written ONLY after success. */
async function writeMarker(dir: FileSystemDirectoryHandle, key: string, size: number): Promise<void> {
  const h = (await dir.getFileHandle(sanitize(key) + MARKER_SUFFIX, { create: true })) as SyncCapableFileHandle
  if (!h.createSyncAccessHandle) return
  const a = await h.createSyncAccessHandle()
  try {
    a.truncate(0)
    a.write(new TextEncoder().encode(String(size)), { at: 0 })
    a.flush()
  } finally {
    a.close()
  }
}

async function removeMarker(dir: FileSystemDirectoryHandle, key: string): Promise<void> {
  try {
    await dir.removeEntry(sanitize(key) + MARKER_SUFFIX)
  } catch {
    /* not present */
  }
}

/**
 * Read a cached asset via a sync access handle (no extra heap copies). Returns null
 * unless the file is present, marked complete, and its size matches the marker.
 */
async function opfsRead(key: string): Promise<ArrayBuffer | null> {
  const dir = await opfsDir()
  if (!dir) return null
  const expected = await readMarker(dir, key)
  if (expected === null) return null // never completed → treat as absent
  try {
    const handle = (await dir.getFileHandle(sanitize(key))) as SyncCapableFileHandle
    if (handle.createSyncAccessHandle) {
      const access = await handle.createSyncAccessHandle()
      try {
        const size = access.getSize()
        if (size !== expected) return null // truncated / mismatched → re-download
        const buf = new ArrayBuffer(size)
        access.read(new Uint8Array(buf), { at: 0 })
        return buf
      } finally {
        access.close()
      }
    }
    const file = await handle.getFile()
    return file.size === expected ? await file.arrayBuffer() : null
  } catch {
    return null
  }
}

/** Stream a URL straight to an OPFS file, verify completeness, then read it back. */
async function streamToOpfs(
  dir: FileSystemDirectoryHandle,
  key: string,
  url: string,
  onProgress?: ProgressFn,
): Promise<ArrayBuffer | null> {
  const handle = (await dir.getFileHandle(sanitize(key), { create: true })) as SyncCapableFileHandle
  if (!handle.createSyncAccessHandle) return null

  // Drop any stale marker first, so a crash mid-write can't be mistaken for complete.
  await removeMarker(dir, key)

  const access = await handle.createSyncAccessHandle()
  let buf: ArrayBuffer
  let size: number
  try {
    access.truncate(0)
    const res = await fetch(url)
    if (!res.ok || !res.body) throw new Error(`fetch ${url} → HTTP ${res.status}`)
    // When the response is compressed (jsDelivr gzips .onnx), Content-Length is the
    // COMPRESSED size while the stream yields decompressed bytes - so it can't be
    // used to verify the final size. HF's .onnx.data is uncompressed (identity), so
    // the truncation guard still applies there, which is where it matters most.
    const compressed = !!res.headers.get('content-encoding')
    const total = Number(res.headers.get('content-length')) || 0
    const reportedTotal = compressed ? 0 : total
    const reader = res.body.getReader()
    let offset = 0
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      access.write(value, { at: offset })
      offset += value.byteLength
      onProgress?.({ loaded: offset, total: reportedTotal })
    }
    access.flush()
    size = access.getSize()
    // Guard against a server/connection that ended the stream early (identity only).
    if (!compressed && total > 0 && size !== total) {
      throw new Error(`incomplete download for ${key}: ${size}/${total} bytes`)
    }
    buf = new ArrayBuffer(size)
    access.read(new Uint8Array(buf), { at: 0 })
  } catch (err) {
    // Interrupted or failed mid-write: close the handle and DELETE the partial data
    // file so it can't linger as orphaned OPFS bytes. (The marker was already removed
    // above, so it would be re-fetched anyway - but we drop the bytes eagerly.) A worker
    // TERMINATED mid-write can't run this; pruneIncompleteAssets() sweeps those on next load.
    try {
      access.close()
    } catch {
      /* handle already closing */
    }
    await dir.removeEntry(sanitize(key)).catch(() => undefined)
    throw err
  }
  access.close()

  // Mark complete only after the data file is fully written and size-verified.
  await writeMarker(dir, key, size)
  return buf
}

// In-memory fallback for browsers without OPFS sync access (kept minimal).
async function fetchToBuffer(url: string, onProgress?: ProgressFn): Promise<ArrayBuffer> {
  const res = await fetch(url)
  if (!res.ok || !res.body) throw new Error(`fetch ${url} → HTTP ${res.status}`)
  const compressed = !!res.headers.get('content-encoding')
  const total = Number(res.headers.get('content-length')) || 0
  const reportedTotal = compressed ? 0 : total
  const reader = res.body.getReader()
  const chunks: Uint8Array[] = []
  let loaded = 0
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    chunks.push(value)
    loaded += value.byteLength
    onProgress?.({ loaded, total: reportedTotal })
  }
  if (!compressed && total > 0 && loaded !== total) {
    throw new Error(`incomplete download: ${loaded}/${total} bytes`)
  }
  const out = new Uint8Array(loaded)
  let off = 0
  for (const c of chunks) {
    out.set(c, off)
    off += c.byteLength
  }
  return out.buffer
}

/**
 * Persist an ALREADY-in-memory buffer to OPFS, then mark it complete. Unlike streamToOpfs
 * this never holds the sync access handle across network I/O (the bytes are already in hand),
 * so it is the safe way to cache the in-memory fallback below - mirrors opfsModelCache.put().
 * Best-effort; callers ignore failures (the model still works from the buffer).
 */
async function writeBufferToOpfs(dir: FileSystemDirectoryHandle, key: string, buf: ArrayBuffer): Promise<void> {
  const handle = (await dir.getFileHandle(sanitize(key), { create: true })) as SyncCapableFileHandle
  if (!handle.createSyncAccessHandle) return
  await removeMarker(dir, key)
  const access = await handle.createSyncAccessHandle()
  try {
    access.truncate(0)
    const bytes = new Uint8Array(buf)
    const CHUNK = 8 * 1024 * 1024 // write in 8 MB slices, not one giant call
    for (let off = 0; off < bytes.length; off += CHUNK) {
      access.write(bytes.subarray(off, Math.min(off + CHUNK, bytes.length)), { at: off })
    }
    access.flush()
  } finally {
    access.close()
  }
  await writeMarker(dir, key, buf.byteLength)
}

export async function hasModelAsset(key: string): Promise<boolean> {
  return (await opfsRead(key)) !== null
}

/**
 * Delete any cached data file that has no completion marker - i.e. a partial left by a
 * download that was interrupted (tab closed or worker terminated mid-stream, so the
 * `.catch` cleanup in streamToOpfs never ran). Best-effort, idempotent; call once before
 * a download session so interrupted bytes can't accumulate as orphaned OPFS storage.
 *
 * Concurrency-safe: overlapping calls (e.g. an abandon-dispose racing a remount, or a
 * dispose racing the next load) share ONE in-flight sweep instead of doubling the work.
 */
let pruneInFlight: Promise<number> | null = null
export function pruneIncompleteAssets(): Promise<number> {
  if (pruneInFlight) return pruneInFlight
  pruneInFlight = pruneIncompleteAssetsImpl().finally(() => {
    pruneInFlight = null
  })
  return pruneInFlight
}

async function pruneIncompleteAssetsImpl(): Promise<number> {
  const dir = await opfsDir()
  if (!dir) return 0
  const d = dir as FileSystemDirectoryHandle & {
    entries?: () => AsyncIterableIterator<[string, FileSystemHandle]>
  }
  if (!d.entries) return 0
  const names: string[] = []
  try {
    for await (const [name] of d.entries()) names.push(name)
  } catch {
    return 0
  }
  const present = new Set(names)
  let pruned = 0
  for (const name of names) {
    if (name.endsWith(MARKER_SUFFIX)) continue // markers are tiny; keep them
    // A data file is "complete" only if its `<name>.done` marker exists. No marker → it
    // was never finished → drop it (a fresh download recreates it).
    if (!present.has(name + MARKER_SUFFIX)) {
      try {
        await dir.removeEntry(name)
        pruned++
      } catch {
        /* locked by a live worker, or already gone - skip */
      }
    }
  }
  return pruned
}

/**
 * Get a model asset as an ArrayBuffer: served from the OPFS cache if present AND
 * verified complete, otherwise STREAMED from `url` to OPFS (never buffering the
 * whole file in heap) and returned.
 */
export async function getModelAsset(
  key: string,
  url: string,
  onProgress?: ProgressFn,
): Promise<ArrayBuffer> {
  const cached = await opfsRead(key)
  if (cached) {
    onProgress?.({ loaded: cached.byteLength, total: cached.byteLength })
    return cached
  }
  const onRetry = (n: number, e: unknown, ms: number): void => {
    const err = e as { name?: string; message?: string }
    // Include the real reason (handle DOMException vs network reset vs size mismatch) so a
    // recurring failure on the big external-data files is diagnosable from the console.
    console.warn(
      `[aidekin] model fetch failed for ${key} (${err?.name || 'Error'}: ${err?.message || String(e)}); retry ${n} in ${ms}ms`,
    )
  }
  const dir = await opfsDir()
  if (dir) {
    try {
      // Retry transient CDN resets / rate limits; a quota error is permanent, so don't retry it.
      const streamed = await withRetry(() => streamToOpfs(dir, key, url, onProgress), {
        shouldRetry: (e) => !(e instanceof DOMException && e.name === 'QuotaExceededError'),
        onRetry,
      })
      if (streamed) return streamed
    } catch (err) {
      // Out-of-space → surface clearly instead of silently re-downloading forever.
      if (err instanceof DOMException && err.name === 'QuotaExceededError') {
        throw new Error('Out of storage space while caching model weights (QuotaExceededError).')
      }
      // Otherwise fall through to the in-memory fetch below.
    }
  }
  // Fallback: stream into memory, then PERSIST the result to OPFS so a future load is served
  // from cache. The streaming path above can fail on the largest files (it holds an OPFS
  // handle open across the whole multi-hundred-MB transfer, which intermittently throws);
  // without writing the buffer back, those big voice weights re-download every session. The
  // bytes are already in memory here, so writeBufferToOpfs holds the handle only briefly.
  // Best-effort: a cache-write failure must never fail the load.
  const buf = await withRetry(() => fetchToBuffer(url, onProgress), { onRetry })
  if (dir) await writeBufferToOpfs(dir, key, buf).catch(() => undefined)
  return buf
}
