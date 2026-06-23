// Minimal, safe markdown → React elements for chat replies. Builds nodes directly (no
// dangerouslySetInnerHTML, no deps, no per-token re-parse of a heavy lib), covering what
// small LLMs actually emit: paragraphs, line breaks, **bold**, *italic*, `code`, fenced
// ```code```, bullet/numbered lists, headings, and [links](url) (http(s) only).

import { type ReactNode } from 'react'

const INLINE = /(`[^`]+`)|(\*\*[^*]+\*\*)|(\*[^*]+\*|_[^_]+_)|(\[[^\]]+\]\([^)]+\))/g

function renderInline(text: string, keyPrefix: string): ReactNode[] {
  const nodes: ReactNode[] = []
  let last = 0
  let i = 0
  let m: RegExpExecArray | null
  INLINE.lastIndex = 0
  while ((m = INLINE.exec(text)) !== null) {
    if (m.index > last) nodes.push(text.slice(last, m.index))
    const tok = m[0]
    const key = `${keyPrefix}-${i++}`
    if (tok.startsWith('`')) {
      nodes.push(
        <code key={key} className="rounded bg-foreground/10 px-1 py-0.5 font-mono text-[0.85em]">
          {tok.slice(1, -1)}
        </code>,
      )
    } else if (tok.startsWith('**')) {
      nodes.push(<strong key={key}>{tok.slice(2, -2)}</strong>)
    } else if (tok.startsWith('[')) {
      const link = /\[([^\]]+)\]\(([^)]+)\)/.exec(tok)
      const href = link ? link[2] : ''
      const safe = /^https?:\/\//i.test(href) ? href : '#'
      nodes.push(
        <a
          key={key}
          href={safe}
          target="_blank"
          rel="noopener noreferrer"
          className="text-primary underline underline-offset-2"
        >
          {link ? link[1] : tok}
        </a>,
      )
    } else {
      nodes.push(<em key={key}>{tok.slice(1, -1)}</em>)
    }
    last = m.index + tok.length
  }
  if (last < text.length) nodes.push(text.slice(last))
  return nodes
}

const isListItem = (l: string) => /^\s*([-*+]|\d+\.)\s+/.test(l)

export function Markdown({ text }: { text: string }) {
  const lines = text.replace(/\r\n/g, '\n').split('\n')
  const blocks: ReactNode[] = []
  let i = 0
  let key = 0

  while (i < lines.length) {
    const line = lines[i]

    if (line.trimStart().startsWith('```')) {
      const buf: string[] = []
      i++
      while (i < lines.length && !lines[i].trimStart().startsWith('```')) buf.push(lines[i++])
      i++ // closing fence
      blocks.push(
        <pre key={key++} className="overflow-x-auto rounded-lg bg-foreground/10 p-2.5 font-mono text-[0.85em]">
          <code>{buf.join('\n')}</code>
        </pre>,
      )
      continue
    }

    const heading = /^(#{1,6})\s+(.*)$/.exec(line)
    if (heading) {
      blocks.push(
        <p key={key++} className="font-semibold">
          {renderInline(heading[2], `h${key}`)}
        </p>,
      )
      i++
      continue
    }

    if (isListItem(line)) {
      const ordered = /^\s*\d+\.\s+/.test(line)
      const items: ReactNode[] = []
      while (i < lines.length && isListItem(lines[i])) {
        const content = lines[i].replace(/^\s*([-*+]|\d+\.)\s+/, '')
        items.push(<li key={items.length}>{renderInline(content, `li${key}-${items.length}`)}</li>)
        i++
      }
      const cls = 'space-y-0.5 pl-5 ' + (ordered ? 'list-decimal' : 'list-disc')
      blocks.push(
        ordered ? (
          <ol key={key++} className={cls}>
            {items}
          </ol>
        ) : (
          <ul key={key++} className={cls}>
            {items}
          </ul>
        ),
      )
      continue
    }

    if (line.trim() === '') {
      i++
      continue
    }

    // Paragraph: gather consecutive plain lines; single newlines become <br>.
    const para: string[] = []
    while (
      i < lines.length &&
      lines[i].trim() !== '' &&
      !lines[i].trimStart().startsWith('```') &&
      !isListItem(lines[i]) &&
      !/^#{1,6}\s+/.test(lines[i])
    ) {
      para.push(lines[i++])
    }
    const inner: ReactNode[] = []
    para.forEach((p, idx) => {
      if (idx > 0) inner.push(<br key={`br-${key}-${idx}`} />)
      inner.push(...renderInline(p, `p${key}-${idx}`))
    })
    blocks.push(<p key={key++}>{inner}</p>)
  }

  return <div className="flex flex-col gap-2">{blocks}</div>
}
