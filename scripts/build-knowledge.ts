// Build a knowledge.bin from local files and/or URLs, using the SAME chunker, embedder
// and binary format as the browser builder — that shared code is what guarantees the
// query vectors match these index vectors (parity). transformers.js uses
// onnxruntime-node here automatically.
//
//   npx tsx scripts/build-knowledge.ts --in ./docs --url https://a.com,https://b.com --out ./public/knowledge.bin
//   [--chunk-tokens 220] [--overlap 0.15]

import { mkdir, readdir, readFile, writeFile } from 'node:fs/promises'
import { dirname, extname, join } from 'node:path'
import { chunkText } from '../src/rag/chunker.ts'
import { embedMany } from '../src/rag/embedder.ts'
import { serializeIndex, type IndexChunk } from '../src/rag/store.ts'

const TEXT_EXT = new Set([
  '.txt', '.md', '.markdown', '.html', '.htm', '.csv', '.json', '.text', '.pdf', '.docx', '.doc',
])

function arg(name: string, fallback = ''): string {
  const i = process.argv.indexOf(name)
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback
}

function stripHtml(html: string): string {
  return html
    .replace(/<script[\s\S]*?<\/script>/gi, '')
    .replace(/<style[\s\S]*?<\/style>/gi, '')
    .replace(/<[^>]+>/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

// Extract text from a local file: PDF (pdfjs legacy, main-thread) and Word .docx
// (mammoth), plus the text formats. Mirrors the browser builder so both produce the
// same index from the same inputs.
async function extractFileNode(path: string): Promise<string> {
  const lower = path.toLowerCase()
  if (lower.endsWith('.pdf')) {
    const pdfjs = await import('pdfjs-dist/legacy/build/pdf.mjs')
    const doc = await pdfjs.getDocument({ data: new Uint8Array(await readFile(path)) }).promise
    let out = ''
    for (let p = 1; p <= doc.numPages; p++) {
      const page = await doc.getPage(p)
      const content = await page.getTextContent()
      out += (content.items as Array<{ str?: string }>).map((i) => i.str ?? '').join(' ') + '\n'
    }
    return out.trim()
  }
  if (/\.docx?$/.test(lower)) {
    const mammoth = await import('mammoth')
    const { value } = await mammoth.extractRawText({ buffer: await readFile(path) })
    return value.trim()
  }
  let text = await readFile(path, 'utf8')
  if (/\.html?$/i.test(lower)) text = stripHtml(text)
  return text
}

async function collectFiles(dir: string): Promise<string[]> {
  const found: string[] = []
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const p = join(dir, entry.name)
    if (entry.isDirectory()) found.push(...(await collectFiles(p)))
    else if (TEXT_EXT.has(extname(entry.name).toLowerCase())) found.push(p)
  }
  return found
}

async function main(): Promise<void> {
  const inDir = arg('--in')
  const urls = arg('--url').split(',').map((s) => s.trim()).filter(Boolean)
  const out = arg('--out', 'knowledge.bin')
  const chunkTokens = Number(arg('--chunk-tokens', '220'))
  const overlapRatio = Number(arg('--overlap', '0.15'))

  const sources: { name: string; text: string }[] = []
  if (inDir) {
    for (const file of await collectFiles(inDir)) {
      sources.push({ name: file, text: await extractFileNode(file) })
    }
  }
  for (const u of urls) {
    const res = await fetch(u)
    if (!res.ok) throw new Error(`Fetch ${u} failed (${res.status})`)
    sources.push({ name: u, text: stripHtml(await res.text()) })
  }
  if (!sources.length) {
    console.error('No input. Use --in <dir> and/or --url <a,b>.')
    process.exit(1)
  }

  const texts: string[] = []
  const srcOf: string[] = []
  for (const s of sources) {
    for (const c of chunkText(s.text, { targetTokens: chunkTokens, overlapRatio })) {
      texts.push(c)
      srcOf.push(s.name)
    }
  }
  if (!texts.length) {
    console.error('No text was extracted from the inputs (empty or unreadable). Nothing to build.')
    process.exit(1)
  }
  console.log(`Chunked ${sources.length} source(s) → ${texts.length} chunks. Embedding…`)

  const vectors: Float32Array[] = []
  const BATCH = 32
  for (let i = 0; i < texts.length; i += BATCH) {
    vectors.push(...(await embedMany(texts.slice(i, i + BATCH))))
    process.stdout.write(`\r  ${Math.min(i + BATCH, texts.length)}/${texts.length}`)
  }
  process.stdout.write('\n')

  const chunks: IndexChunk[] = texts.map((text, i) => ({ text, vector: vectors[i], source: srcOf[i] }))
  const buf = serializeIndex(chunks, new Date().toISOString())
  await mkdir(dirname(out), { recursive: true })
  await writeFile(out, Buffer.from(buf))
  console.log(`Wrote ${out} — ${(buf.byteLength / 1024).toFixed(0)} KB, ${chunks.length} chunks.`)
}

main().catch((e) => {
  console.error(e)
  process.exit(1)
})
