import { useCallback, useRef, useState, type ReactNode } from 'react'
import {
  Upload,
  FileText,
  Link2,
  Trash2,
  Download,
  AlertTriangle,
  Loader2,
  CheckCircle2,
  Database,
} from 'lucide-react'
import { chunkText } from '@/rag/chunker'
import { serializeIndex, type IndexChunk } from '@/rag/store'

interface Source {
  id: number
  name: string
  text: string
}

interface BuildResult {
  chunks: number
  bytes: number
}

type Stage = { label: string; pct: number } | null

// Extract plain text from a dropped file. PDF (pdfjs) and Word .docx (mammoth) are
// lazy-loaded only when such a file is added, so they never ship in the rest of the
// bundle (this route is already code-split).
let pdfWorkerReady = false

async function extractFile(file: File): Promise<string> {
  const name = file.name.toLowerCase()

  if (name.endsWith('.pdf')) {
    const pdfjs = await import('pdfjs-dist')
    if (!pdfWorkerReady) {
      pdfjs.GlobalWorkerOptions.workerSrc = (await import('pdfjs-dist/build/pdf.worker.min.mjs?url')).default
      pdfWorkerReady = true
    }
    const task = pdfjs.getDocument({ data: new Uint8Array(await file.arrayBuffer()) })
    const doc = await task.promise
    let out = ''
    for (let p = 1; p <= doc.numPages; p++) {
      const page = await doc.getPage(p)
      const content = await page.getTextContent()
      out += (content.items as Array<{ str?: string }>).map((it) => it.str ?? '').join(' ') + '\n'
    }
    await task.destroy()
    return out.trim()
  }

  if (/\.docx?$/.test(name)) {
    const mammoth = await import('mammoth')
    const { value } = await mammoth.extractRawText({ arrayBuffer: await file.arrayBuffer() })
    return value.trim()
  }

  const raw = await file.text()
  if (/\.(html?|xhtml)$/.test(name) || file.type.includes('html')) {
    const doc = new DOMParser().parseFromString(raw, 'text/html')
    doc.querySelectorAll('script,style,noscript').forEach((el) => el.remove())
    return doc.body?.textContent?.trim() ?? raw
  }
  return raw
}

export default function Builder() {
  const [sources, setSources] = useState<Source[]>([])
  const [paste, setPaste] = useState('')
  const [url, setUrl] = useState('')
  const [stage, setStage] = useState<Stage>(null)
  const [result, setResult] = useState<BuildResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const seq = useRef(0)
  const fileInput = useRef<HTMLInputElement>(null)

  const add = useCallback((name: string, text: string) => {
    const clean = text.trim()
    if (!clean) return
    setResult(null)
    setSources((prev) => [...prev, { id: ++seq.current, name, text: clean }])
  }, [])

  const addFiles = useCallback(
    async (files: FileList | null) => {
      if (!files) return
      for (const file of Array.from(files)) {
        try {
          add(file.name, await extractFile(file))
        } catch {
          setError(`Couldn't read ${file.name}. Make sure it's a text, PDF, or Word file.`)
        }
      }
    },
    [add],
  )

  const addUrl = useCallback(async () => {
    const u = url.trim()
    if (!u) return
    setError(null)
    try {
      const res = await fetch(u)
      if (!res.ok) throw new Error(String(res.status))
      const html = await res.text()
      const doc = new DOMParser().parseFromString(html, 'text/html')
      doc.querySelectorAll('script,style,noscript').forEach((el) => el.remove())
      add(new URL(u).hostname + new URL(u).pathname, doc.body?.textContent?.trim() ?? html)
      setUrl('')
    } catch {
      setError(`Couldn't fetch ${u}. The page must allow cross-origin requests (CORS).`)
    }
  }, [url, add])

  const totalChars = sources.reduce((n, s) => n + s.text.length, 0)

  const build = useCallback(async () => {
    if (!sources.length) return
    setError(null)
    setResult(null)
    try {
      // 1. Chunk every source.
      const texts: IndexChunk['text'][] = []
      const meta: { source: string }[] = []
      for (const s of sources) {
        for (const c of chunkText(s.text)) {
          texts.push(c)
          meta.push({ source: s.name })
        }
      }
      if (!texts.length) throw new Error('No text to index.')

      // 2. Embed in batches (loads the ~22 MB embedder once, then runs on WASM).
      setStage({ label: 'Loading the embedding model…', pct: 0 })
      const { embedMany } = await import('@/rag/embedder')
      const vectors: Float32Array[] = []
      const BATCH = 16
      for (let i = 0; i < texts.length; i += BATCH) {
        const batch = texts.slice(i, i + BATCH)
        const out = await embedMany(batch)
        vectors.push(...out)
        setStage({ label: `Embedding ${Math.min(i + BATCH, texts.length)} / ${texts.length} chunks…`, pct: (i + BATCH) / texts.length })
      }

      // 3. Serialize + download.
      setStage({ label: 'Packaging…', pct: 1 })
      const chunks: IndexChunk[] = texts.map((text, i) => ({ text, vector: vectors[i], source: meta[i].source }))
      const buf = serializeIndex(chunks, new Date().toISOString())
      const blob = new Blob([buf], { type: 'application/octet-stream' })
      const a = document.createElement('a')
      a.href = URL.createObjectURL(blob)
      a.download = 'knowledge.bin'
      a.click()
      URL.revokeObjectURL(a.href)

      setResult({ chunks: chunks.length, bytes: buf.byteLength })
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Build failed.')
    } finally {
      setStage(null)
    }
  }, [sources])

  const building = stage !== null

  return (
    <section className="mx-auto max-w-3xl px-5 py-14">
      <div className="mb-2 flex items-center gap-2.5">
        <Database className="size-6 text-muted-foreground" />
        <h1 className="font-display text-3xl font-semibold tracking-tight">Build your knowledge file</h1>
      </div>
      <p className="mb-6 text-muted-foreground">
        Add your content below. It’s chunked and embedded <strong>entirely in your browser</strong>. Nothing
        is uploaded. You’ll get a small <code className="rounded bg-secondary px-1.5 py-0.5 text-sm">knowledge.bin</code> to
        host and point the widget at.
      </p>

      <Callout>
        Your <code className="rounded bg-background/60 px-1 py-0.5">knowledge.bin</code> is downloaded by every
        visitor, so treat it as <strong>public</strong>. Never include secrets, credentials, or private
        personal data.
      </Callout>

      {/* Inputs */}
      <div
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault()
          void addFiles(e.dataTransfer.files)
        }}
        className="mt-6 grid place-items-center gap-2 rounded-lg border-2 border-dashed border-border bg-card/40 px-6 py-10 text-center"
      >
        <Upload className="size-6 text-muted-foreground" />
        <p className="text-sm">
          Drop files here or{' '}
          <button type="button" onClick={() => fileInput.current?.click()} className="font-semibold text-primary underline-offset-2 hover:underline">
            browse
          </button>
        </p>
        <p className="text-xs text-muted-foreground">PDF · Word · .txt · .md · .html · .csv · .json</p>
        <input
          ref={fileInput}
          type="file"
          multiple
          accept=".pdf,.docx,.doc,.txt,.md,.markdown,.html,.htm,.csv,.json,.text"
          className="hidden"
          onChange={(e) => void addFiles(e.target.files)}
        />
      </div>

      <div className="mt-4 grid gap-4 sm:grid-cols-2">
        <div className="rounded-lg border border-border bg-card/40 p-3">
          <label className="mb-1.5 flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
            <FileText className="size-3.5" /> Paste text
          </label>
          <textarea
            value={paste}
            onChange={(e) => setPaste(e.target.value)}
            rows={3}
            placeholder="Paste any text…"
            className="w-full resize-none rounded-md border border-input bg-background px-3 py-2 text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring"
          />
          <button
            type="button"
            onClick={() => {
              add('Pasted text', paste)
              setPaste('')
            }}
            disabled={!paste.trim()}
            className="mt-2 rounded-md bg-secondary px-3 py-1.5 text-xs font-semibold transition-colors hover:bg-secondary/70 disabled:opacity-40"
          >
            Add
          </button>
        </div>

        <div className="rounded-lg border border-border bg-card/40 p-3">
          <label className="mb-1.5 flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
            <Link2 className="size-3.5" /> From a URL
          </label>
          <input
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && void addUrl()}
            placeholder="https://example.com/page"
            className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring"
          />
          <button
            type="button"
            onClick={() => void addUrl()}
            disabled={!url.trim()}
            className="mt-2 rounded-md bg-secondary px-3 py-1.5 text-xs font-semibold transition-colors hover:bg-secondary/70 disabled:opacity-40"
          >
            Fetch &amp; add
          </button>
        </div>
      </div>

      {error && (
        <p className="mt-4 flex items-start gap-2 rounded-md border border-destructive/30 bg-destructive/10 px-4 py-2.5 text-sm text-destructive">
          <AlertTriangle className="mt-0.5 size-4 shrink-0" /> {error}
        </p>
      )}

      {/* Sources */}
      {sources.length > 0 && (
        <div className="mt-6">
          <div className="mb-2 flex items-center justify-between text-sm">
            <span className="font-semibold">{sources.length} source{sources.length > 1 ? 's' : ''}</span>
            <span className="text-muted-foreground">{(totalChars / 1000).toFixed(1)}k characters</span>
          </div>
          <ul className="divide-y divide-border overflow-hidden rounded-lg border border-border">
            {sources.map((s) => (
              <li key={s.id} className="flex items-center gap-3 bg-card/40 px-4 py-2.5 text-sm">
                <FileText className="size-4 shrink-0 text-muted-foreground" />
                <span className="flex-1 truncate">{s.name}</span>
                <span className="text-xs text-muted-foreground">{(s.text.length / 1000).toFixed(1)}k</span>
                <button
                  type="button"
                  onClick={() => setSources((prev) => prev.filter((x) => x.id !== s.id))}
                  className="text-muted-foreground hover:text-destructive"
                  aria-label={`Remove ${s.name}`}
                >
                  <Trash2 className="size-4" />
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Build */}
      <div className="mt-6">
        <button
          type="button"
          onClick={() => void build()}
          disabled={!sources.length || building}
          className="inline-flex items-center gap-2 rounded-md bg-primary px-6 py-3 text-sm font-semibold text-primary-foreground transition-transform hover:-translate-y-px disabled:translate-y-0 disabled:opacity-40"
        >
          {building ? <Loader2 className="size-4 animate-spin" /> : <Download className="size-4" />}
          {building ? 'Building…' : 'Build & download knowledge.bin'}
        </button>

        {stage && (
          <div className="mt-4">
            <div className="mb-1 flex justify-between text-xs text-muted-foreground">
              <span>{stage.label}</span>
              <span>{Math.round(stage.pct * 100)}%</span>
            </div>
            <div className="h-1.5 overflow-hidden rounded-full bg-secondary">
              <div className="h-full rounded-full bg-primary transition-[width]" style={{ width: `${Math.max(4, stage.pct * 100)}%` }} />
            </div>
          </div>
        )}
      </div>

      {result && <ResultPanel result={result} />}
    </section>
  )
}

function ResultPanel({ result }: { result: BuildResult }) {
  const kb = result.bytes / 1024
  const size = kb < 1024 ? `${kb.toFixed(0)} KB` : `${(kb / 1024).toFixed(2)} MB`
  return (
    <div className="mt-8 rounded-lg border border-primary/30 bg-primary/[0.06] p-6">
      <div className="mb-3 flex items-center gap-2 text-primary">
        <CheckCircle2 className="size-5" />
        <span className="font-semibold">knowledge.bin downloaded</span>
      </div>
      <p className="text-sm text-muted-foreground">
        {result.chunks} chunks · <strong>{size}</strong> · each visitor downloads this once, then it’s cached.
      </p>
      <div className="mt-4 space-y-2 text-sm">
        <p className="font-medium">Host it (any static host with cross-origin reads works):</p>
        <ul className="ml-4 list-disc space-y-1 text-muted-foreground">
          <li>A public GitHub repo, served free through a CDN like jsDelivr (<code className="rounded bg-secondary px-1">cdn.jsdelivr.net/gh/you/repo/knowledge.bin</code>)</li>
          <li>Cloudflare R2, GitHub Pages, Netlify, or Vercel (all have free tiers)</li>
          <li>Your own server or bucket (add <code className="rounded bg-secondary px-1">Access-Control-Allow-Origin: *</code>)</li>
        </ul>
        <p className="text-xs text-muted-foreground">
          These are independent services; aidekin is not affiliated with any of them.
        </p>
        <p className="pt-1 text-muted-foreground">
          Then set <code className="rounded bg-secondary px-1">data-knowledge-url</code> in your snippet (the
          Configurator does this for you).
        </p>
      </div>
    </div>
  )
}

function Callout({ children }: { children: ReactNode }) {
  return (
    <div className="flex items-start gap-2.5 rounded-md border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-sm">
      <AlertTriangle className="mt-0.5 size-4 shrink-0 text-amber-600 dark:text-amber-400" />
      <span>{children}</span>
    </div>
  )
}
