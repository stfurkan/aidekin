// Tiny IndexedDB key->value store for bitgpu chat snapshots (chat.save() returns a
// structured-cloneable object: the engine's KV cache buffer plus the committed-transcript
// bookkeeping). IndexedDB - not localStorage/JSON - because the snapshot carries a binary KV
// buffer that JSON.stringify would drop, and it can be tens of MB (larger than localStorage's
// ~5MB quota). Worker-side, so the KV buffer never crosses the postMessage boundary.
//
// Every op is BEST-EFFORT: on any failure (private mode, storage partitioning, quota, an
// unavailable IndexedDB) it resolves to a safe fallback (null / no-op) instead of throwing, so
// snapshot persistence can never break a conversation.
const DB_NAME = 'aidekin-sessions'
const STORE = 'snapshots'
const DB_VERSION = 1

let dbPromise: Promise<IDBDatabase | null> | null = null

function openDb(): Promise<IDBDatabase | null> {
  if (dbPromise) return dbPromise
  dbPromise = new Promise((resolve) => {
    try {
      if (typeof indexedDB === 'undefined') return resolve(null)
      const req = indexedDB.open(DB_NAME, DB_VERSION)
      req.onupgradeneeded = (): void => {
        const db = req.result
        if (!db.objectStoreNames.contains(STORE)) db.createObjectStore(STORE)
      }
      req.onsuccess = (): void => resolve(req.result)
      req.onerror = (): void => resolve(null)
      req.onblocked = (): void => resolve(null)
    } catch {
      resolve(null)
    }
  })
  return dbPromise
}

/** Run one request inside a transaction, resolving to its result (or null on any failure). */
function run<T>(mode: IDBTransactionMode, make: (store: IDBObjectStore) => IDBRequest<T>): Promise<T | null> {
  return openDb().then((db) => {
    if (!db) return null
    return new Promise<T | null>((resolve) => {
      try {
        const tx = db.transaction(STORE, mode)
        const req = make(tx.objectStore(STORE))
        req.onsuccess = (): void => resolve(req.result ?? null)
        req.onerror = (): void => resolve(null)
        tx.onabort = (): void => resolve(null)
      } catch {
        resolve(null)
      }
    })
  })
}

/** Read a stored snapshot (structured-clone), or null when absent/unavailable. */
export const sessionGet = <T>(key: string): Promise<T | null> => run<T>('readonly', (s) => s.get(key) as IDBRequest<T>)

/** Persist a snapshot under `key`. Best-effort: a quota/write failure resolves without throwing. */
export const sessionPut = (key: string, value: unknown): Promise<unknown> => run('readwrite', (s) => s.put(value, key))

/** Delete a stored snapshot (new conversation). Best-effort. */
export const sessionDelete = (key: string): Promise<unknown> => run('readwrite', (s) => s.delete(key))
