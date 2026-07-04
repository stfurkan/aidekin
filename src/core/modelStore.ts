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
// A file with no marker - or whose size no longer matches - is treated as absent.
// Without this, a partially-written .onnx.data from an interrupted run is silently
// handed to ORT and fails with "Out of bounds".
//
// RESUMABLE: an interrupted download keeps its partial bytes plus a `<key>.part`
// sidecar holding the remote ETag. The next attempt sends `Range: bytes=<have>-`
// with `If-Range: <etag>`, so the server resumes (206) when the file is unchanged
// and otherwise restarts (200). A dropped ~1.9 GB first-load picks up where it left
// off instead of re-downloading from zero.

import { withRetry } from './retry'
import { dlog } from './log'

export interface FetchProgress {
  readonly loaded: number
  readonly total: number
}
export type ProgressFn = (p: FetchProgress) => void

const OPFS_DIR = 'aidekin-models'
const MARKER_SUFFIX = '.done'
const PART_SUFFIX = '.part' // resume sidecar for an in-progress download: holds the remote ETag
const FLUSH_EVERY = 32 * 1024 * 1024 // flush to disk every 32 MB so a crash loses at most that much
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

/** Write `bytes` to OPFS file `name`. Uses the worker-only sync access handle when available (fast,
 *  no extra copies); otherwise the async createWritable() stream, which ALSO works on the MAIN
 *  THREAD - where the RAG embedder runs. Without the main-thread path the embedder could never
 *  persist (createSyncAccessHandle is worker-only), so it re-downloaded every session and left a
 *  0-byte marker-less file the pruner deleted each load. Returns false if no OPFS write API exists. */
async function opfsWrite(dir: FileSystemDirectoryHandle, name: string, bytes: Uint8Array<ArrayBuffer>): Promise<boolean> {
  const h = (await dir.getFileHandle(name, { create: true })) as SyncCapableFileHandle
  if (h.createSyncAccessHandle) {
    const a = await h.createSyncAccessHandle()
    try {
      a.truncate(0)
      const CHUNK = 8 * 1024 * 1024 // write in 8 MB slices, not one giant call
      for (let off = 0; off < bytes.length; off += CHUNK) a.write(bytes.subarray(off, Math.min(off + CHUNK, bytes.length)), { at: off })
      a.flush()
    } finally {
      a.close()
    }
    return true
  }
  if (typeof h.createWritable === 'function') {
    const w = await h.createWritable()
    try {
      await w.write(bytes)
    } finally {
      await w.close()
    }
    return true
  }
  return false
}

/** Record that `key` is fully downloaded (size bytes). Written ONLY after success. */
async function writeMarker(dir: FileSystemDirectoryHandle, key: string, size: number): Promise<void> {
  const text = String(size) // ASCII digits → one byte each; keep the buffer ArrayBuffer-backed
  const bytes = new Uint8Array(text.length)
  for (let i = 0; i < text.length; i++) bytes[i] = text.charCodeAt(i)
  await opfsWrite(dir, sanitize(key) + MARKER_SUFFIX, bytes)
}

async function removeMarker(dir: FileSystemDirectoryHandle, key: string): Promise<void> {
  try {
    await dir.removeEntry(sanitize(key) + MARKER_SUFFIX)
  } catch {
    /* not present */
  }
}

/** Read the resume sidecar → the remote ETag of the in-progress download for `key`, or null. */
async function readPart(dir: FileSystemDirectoryHandle, key: string): Promise<string | null> {
  try {
    const h = await dir.getFileHandle(sanitize(key) + PART_SUFFIX)
    const s = (await (await h.getFile()).text()).trim()
    return s || null
  } catch {
    return null
  }
}

/** Record the remote ETag so an interrupted download can be resumed against the SAME bytes. */
async function writePart(dir: FileSystemDirectoryHandle, key: string, etag: string): Promise<void> {
  const h = (await dir.getFileHandle(sanitize(key) + PART_SUFFIX, { create: true })) as SyncCapableFileHandle
  if (!h.createSyncAccessHandle) return
  const a = await h.createSyncAccessHandle()
  try {
    a.truncate(0)
    a.write(new TextEncoder().encode(etag), { at: 0 })
    a.flush()
  } finally {
    a.close()
  }
}

async function removePart(dir: FileSystemDirectoryHandle, key: string): Promise<void> {
  try {
    await dir.removeEntry(sanitize(key) + PART_SUFFIX)
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

/** Error for a definitive HTTP answer. A 4xx (other than 408/429) cannot be fixed by retrying,
 *  so it is marked `permanent` and withRetry rethrows it immediately instead of backing off. */
function httpError(url: string, status: number, res?: Response): Error {
  const err = new Error(`fetch ${url} → HTTP ${status}`) as Error & { permanent?: boolean; retryAfterMs?: number }
  if (status >= 400 && status < 500 && status !== 408 && status !== 429) err.permanent = true
  const ra = Number(res?.headers.get('retry-after'))
  if (Number.isFinite(ra) && ra > 0) err.retryAfterMs = Math.min(ra * 1000, 30_000) // honor the server's wait (HF rate limits)
  return err
}

/** Stream a URL straight to an OPFS file, verify completeness, then read it back. */
async function streamToOpfs(
  dir: FileSystemDirectoryHandle,
  key: string,
  url: string,
  onProgress?: ProgressFn,
  readBack = true, // false: persist to disk only and skip the whole-file ArrayBuffer (streaming consumers)
): Promise<ArrayBuffer | null> {
  const handle = (await dir.getFileHandle(sanitize(key), { create: true })) as SyncCapableFileHandle
  if (!handle.createSyncAccessHandle) {
    // Main thread (no sync access handle - e.g. the RAG embedder): drop the just-created 0-byte file
    // so the pruner doesn't treat it as an incomplete download. getModelAsset's in-memory fallback
    // (writeBufferToOpfs -> createWritable) persists it instead.
    await dir.removeEntry(sanitize(key)).catch(() => undefined)
    return null
  }

  // Drop any stale marker first, so a crash mid-write can't be mistaken for complete.
  await removeMarker(dir, key)

  const access = await handle.createSyncAccessHandle()
  let buf: ArrayBuffer
  let size: number
  try {
    // RESUME instead of restart: if partial bytes already exist AND we stored the remote ETag, ask
    // for just the remaining range with If-Range, so the server resumes only when the file is
    // unchanged (else it returns the full 200 and we start over). HF serves Accept-Ranges + stable
    // ETags, so a dropped 1.9 GB first-load picks up where it left off instead of re-downloading.
    const have = access.getSize()
    const priorEtag = have > 0 ? await readPart(dir, key) : null
    const headers: Record<string, string> = {}
    if (have > 0 && priorEtag) {
      headers.Range = `bytes=${have}-`
      headers['If-Range'] = priorEtag
    }
    let res = await fetch(url, Object.keys(headers).length ? { headers } : undefined)
    if (res.status === 416 && headers.Range) {
      // 416 = the resume Range starts at/past EOF: a byte-complete partial whose final flush
      // landed but whose .done marker never did. It can't be resumed or trusted, so drop it
      // and restart a clean download.
      access.truncate(0)
      await removePart(dir, key)
      res = await fetch(url)
    }
    if (!res.ok || !res.body) throw httpError(url, res.status, res)
    const resumed = res.status === 206 // server honored the range → keep what we have, append the rest
    let offset = resumed ? have : 0
    if (!resumed) access.truncate(0) // fresh, changed, or no-range server (200): start over
    // Persist the ETag so a LATER interruption can resume against these same bytes.
    const etag = res.headers.get('etag')
    if (etag) await writePart(dir, key, etag)
    // When the response is compressed (jsDelivr gzips .onnx), Content-Length is the COMPRESSED size
    // and can't verify the final bytes. HF's .onnx.data is identity, so the guard applies where it
    // matters most. Total = the WHOLE file: Content-Range's "/total" on a 206, else Content-Length.
    const compressed = !!res.headers.get('content-encoding')
    const cr = res.headers.get('content-range')
    const total = compressed ? 0 : cr ? Number(cr.split('/')[1]) || 0 : Number(res.headers.get('content-length')) || 0
    const reader = res.body.getReader()
    let sinceFlush = 0
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      access.write(value, { at: offset })
      offset += value.byteLength
      sinceFlush += value.byteLength
      if (sinceFlush >= FLUSH_EVERY) {
        access.flush() // bound how much a crash can lose; getSize() stays a safe resume point
        sinceFlush = 0
      }
      onProgress?.({ loaded: offset, total })
    }
    access.flush()
    size = access.getSize()
    // Guard against a server/connection that ended the stream early (identity only).
    if (total > 0 && size !== total) {
      throw new Error(`incomplete download for ${key}: ${size}/${total} bytes`)
    }
    // Streaming consumers re-read from disk via opfsReadStream; skipping the read-back keeps the
    // first download's peak at one chunk (the buffered peak is what iOS Safari kills the tab for).
    buf = readBack ? new ArrayBuffer(size) : new ArrayBuffer(0)
    if (readBack) access.read(new Uint8Array(buf), { at: 0 })
  } catch (err) {
    // Interrupted or failed mid-write: KEEP the partial bytes + the .part ETag sidecar so the NEXT
    // call resumes from here instead of re-downloading the whole (multi-hundred-MB) file. A worker
    // TERMINATED mid-write can't run this, but the bytes + sidecar persist and resume on next load.
    try {
      access.close()
    } catch {
      /* handle already closing */
    }
    throw err
  }
  access.close()

  // Mark complete only after the data file is fully written and size-verified; the resume sidecar
  // is no longer needed.
  await writeMarker(dir, key, size)
  await removePart(dir, key)
  return buf
}

// In-memory fallback for browsers without OPFS sync access (kept minimal).
async function fetchToBuffer(url: string, onProgress?: ProgressFn): Promise<ArrayBuffer> {
  const res = await fetch(url)
  if (!res.ok || !res.body) throw httpError(url, res.status, res)
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
 * so it is the safe way to cache the in-memory fallback below.
 * Best-effort; callers ignore failures (the model still works from the buffer).
 */
async function writeBufferToOpfs(dir: FileSystemDirectoryHandle, key: string, buf: ArrayBuffer): Promise<void> {
  await removeMarker(dir, key) // a crash mid-write must not leave a complete marker behind
  if (await opfsWrite(dir, sanitize(key), new Uint8Array(buf))) await writeMarker(dir, key, buf.byteLength)
}

export async function hasModelAsset(key: string): Promise<boolean> {
  // Metadata-only check (marker + file size). Never read the content here: answering a boolean by
  // materializing a ~290 MB weight file in memory was a real main-thread memory spike per mount.
  const dir = await opfsDir()
  if (!dir) return false
  const expected = await readMarker(dir, key)
  if (expected === null) return false
  try {
    const handle = await dir.getFileHandle(sanitize(key))
    const file = await handle.getFile()
    return file.size === expected
  } catch {
    return false
  }
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
  const dropped: string[] = []
  for (const name of names) {
    if (name.endsWith(MARKER_SUFFIX) || name.endsWith(PART_SUFFIX)) continue // sidecars are tiny; keep them
    // Keep a data file that is either COMPLETE (`.done` marker) or RESUMABLE (`.part` ETag sidecar -
    // a later getModelAsset resumes it). Drop only the truly orphaned: no marker AND no sidecar (e.g.
    // a partial from before resumable downloads existed, which can't be resumed without an ETag).
    if (present.has(name + MARKER_SUFFIX) || present.has(name + PART_SUFFIX)) continue
    try {
      await dir.removeEntry(name)
      dropped.push(name)
    } catch {
      /* locked by a live worker, or already gone - skip */
    }
  }
  // Name the casualties: a file pruned on EVERY load means a download keeps finishing without its
  // `.done` marker (it re-downloads each session), which this surfaces for diagnosis.
  if (dropped.length) dlog(`[aidekin] pruned incomplete cache file(s): ${dropped.join(', ')}`)
  return dropped.length
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
  // Cross-tab: two same-origin tabs loading at once must not download the same asset twice
  // (worse, the loser's cache write transiently removes the winner's fresh .done marker).
  // Serialize per key on a Web Lock; the waiter re-checks the cache the winner just filled.
  const locks = typeof navigator !== 'undefined' ? (navigator as Navigator & { locks?: LockManager }).locks : undefined
  if (locks?.request) {
    return locks.request('aidekin:model:' + key, async (): Promise<ArrayBuffer> => {
      const won = await opfsRead(key)
      if (won) {
        onProgress?.({ loaded: won.byteLength, total: won.byteLength })
        return won
      }
      return downloadAsset(key, url, onProgress)
    }) as Promise<ArrayBuffer>
  }
  return downloadAsset(key, url, onProgress)
}

/**
 * Stream a model asset WITHOUT ever holding the whole file in memory: served straight from the
 * OPFS file when cached; otherwise downloaded to OPFS (locked, resumable, no read-back) and then
 * streamed from disk. Cache-less sessions (private browsing, quota exhaustion) pipe the network
 * body straight through - they re-download next visit, which those sessions imply anyway. The
 * buffered fallback was the last remaining ~290MB peak, and iOS Safari kills the tab for it.
 */
export async function getModelAssetStream(key: string, url: string, onProgress?: ProgressFn): Promise<ReadableStream<Uint8Array>> {
  const cached = await opfsReadStream(key)
  if (cached) {
    onProgress?.({ loaded: cached.size, total: cached.size })
    return cached.stream
  }
  const run = async (): Promise<ReadableStream<Uint8Array>> => {
    const won = await opfsReadStream(key) // the lock winner may have just filled the cache
    if (won) {
      onProgress?.({ loaded: won.size, total: won.size })
      return won.stream
    }
    const dir = await opfsDir()
    if (dir) {
      try {
        const onRetry = (n: number, e: unknown, ms: number): void => {
          const err = e as { name?: string; message?: string }
          console.warn(`[aidekin] model fetch failed for ${key} (${err?.name || 'Error'}: ${err?.message || String(e)}); retry ${n} in ${ms}ms`)
        }
        const ok = await withRetry(() => streamToOpfs(dir, key, url, onProgress, false), {
          shouldRetry: (e) => !(e instanceof DOMException && e.name === 'QuotaExceededError'),
          onRetry,
        })
        if (ok) {
          const now = await opfsReadStream(key)
          if (now) return now.stream
        }
      } catch (err) {
        if (err instanceof DOMException && err.name === 'QuotaExceededError') {
          await dir.removeEntry(sanitize(key)).catch(() => undefined)
          await removePart(dir, key)
        }
        // fall through to the direct network stream
      }
    }
    const res = await fetch(url)
    if (!res.ok || !res.body) throw httpError(url, res.status, res)
    const total = Number(res.headers.get('content-length')) || 0
    let loaded = 0
    return res.body.pipeThrough(
      new TransformStream<Uint8Array, Uint8Array>({
        transform(chunk, ctrl) {
          loaded += chunk.byteLength
          onProgress?.({ loaded, total })
          ctrl.enqueue(chunk)
        },
      }),
    )
  }
  const locks = typeof navigator !== 'undefined' ? (navigator as Navigator & { locks?: LockManager }).locks : undefined
  if (locks?.request) return locks.request('aidekin:model:' + key, run) as Promise<ReadableStream<Uint8Array>>
  return run()
}

/** Stream a COMPLETE cached asset from OPFS (same completeness checks as opfsRead), or null. */
async function opfsReadStream(key: string): Promise<{ stream: ReadableStream<Uint8Array>; size: number } | null> {
  const dir = await opfsDir()
  if (!dir) return null
  const expected = await readMarker(dir, key)
  if (expected === null) return null
  try {
    const handle = await dir.getFileHandle(sanitize(key))
    const file = await handle.getFile()
    if (file.size !== expected) return null
    return { stream: file.stream() as ReadableStream<Uint8Array>, size: file.size }
  } catch {
    return null
  }
}

async function downloadAsset(key: string, url: string, onProgress?: ProgressFn): Promise<ArrayBuffer> {
  const onRetry = (n: number, e: unknown, ms: number): void => {
    const err = e as { name?: string; message?: string }
    // Include the real reason (handle DOMException vs network reset vs size mismatch) so a
    // recurring failure on the big external-data files is diagnosable from the console.
    console.warn(
      `[aidekin] model fetch failed for ${key} (${err?.name || 'Error'}: ${err?.message || String(e)}); retry ${n} in ${ms}ms`,
    )
  }
  const dir = await opfsDir()
  let persist = !!dir
  if (dir) {
    try {
      // Retry transient CDN resets / rate limits; a quota error is permanent, so don't retry it.
      const streamed = await withRetry(() => streamToOpfs(dir, key, url, onProgress), {
        shouldRetry: (e) => !(e instanceof DOMException && e.name === 'QuotaExceededError'),
        onRetry,
      })
      if (streamed) return streamed
    } catch (err) {
      // Out of quota mid-stream: the partial can never complete, and its bytes + .part sidecar
      // would sit in quota forever. Drop them, skip the cache write below (it would just hit
      // quota again), and serve this session via the in-memory path.
      if (err instanceof DOMException && err.name === 'QuotaExceededError') {
        await dir.removeEntry(sanitize(key)).catch(() => undefined)
        await removePart(dir, key)
        persist = false
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
  if (dir && persist) await writeBufferToOpfs(dir, key, buf).catch(() => undefined)
  return buf
}
