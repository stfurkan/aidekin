import { useMemo, useState, type ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { Check, Copy, Database } from 'lucide-react'
import { WidgetApp } from '@/widget/WidgetApp'
import { WidgetFrame } from '@/widget/WidgetFrame'
import { withDefaults, type WidgetConfig } from '@/widget/protocol'

const CDN = 'https://cdn.aidekin.com/loader.js'

interface Form {
  mode: 'text' | 'voice' | 'both'
  title: string
  greeting: string
  systemPrompt: string
  accent: string
  position: 'bottom-right' | 'bottom-left'
  knowledgeUrl: string
  persist: boolean
  reasoning: boolean
}

const INITIAL: Form = {
  mode: 'text',
  title: 'aidekin',
  greeting: 'Hi! Ask me anything. I run entirely in your browser.',
  systemPrompt: '',
  accent: '#29a383',
  position: 'bottom-right',
  knowledgeUrl: '',
  persist: true,
  reasoning: false,
}

const esc = (s: string) => s.replace(/"/g, '&quot;')

function buildSnippet(f: Form, src: string): string {
  const attrs = [`src="${src}"`]
  if (f.mode !== 'text') attrs.push(`data-mode="${f.mode}"`)
  if (f.title.trim()) attrs.push(`data-title="${esc(f.title.trim())}"`)
  if (f.greeting.trim()) attrs.push(`data-greeting="${esc(f.greeting.trim())}"`)
  if (f.systemPrompt.trim()) attrs.push(`data-system-prompt="${esc(f.systemPrompt.trim())}"`)
  if (f.accent) attrs.push(`data-accent="${f.accent}"`)
  if (f.position !== 'bottom-right') attrs.push(`data-position="${f.position}"`)
  if (f.knowledgeUrl.trim()) attrs.push(`data-knowledge-url="${f.knowledgeUrl.trim()}"`)
  if (f.reasoning) attrs.push(`data-reasoning="true"`)
  if (!f.persist) attrs.push(`data-persist="false"`)
  attrs.push('defer')
  return `<script\n  ${attrs.join('\n  ')}\n></script>`
}

export default function Configure() {
  const [f, setF] = useState<Form>(INITIAL)
  const set = <K extends keyof Form>(k: K, v: Form[K]) => setF((p) => ({ ...p, [k]: v }))

  const previewConfig: WidgetConfig = useMemo(
    () =>
      withDefaults({
        mode: f.mode,
        title: f.title,
        greeting: f.greeting,
        systemPrompt: f.systemPrompt || undefined,
        accent: f.accent,
        knowledgeUrl: f.knowledgeUrl || undefined,
        reasoning: f.reasoning,
      }),
    [f],
  )

  return (
    <section className="mx-auto max-w-6xl px-5 py-14">
      <h1 className="font-display text-3xl font-semibold tracking-tight">Configure your widget</h1>
      <p className="mt-2 text-muted-foreground">
        Tune it, watch the live preview, then copy the one-line snippet. Everything stays client-side.
      </p>

      <div className="mt-8 grid gap-8 lg:grid-cols-[1fr_360px]">
        {/* Form */}
        <div className="space-y-5">
          <Field label="Name (title)">
            <input className={inputCls} value={f.title} onChange={(e) => set('title', e.target.value)} />
          </Field>

          <Field label="Greeting">
            <input className={inputCls} value={f.greeting} onChange={(e) => set('greeting', e.target.value)} />
          </Field>

          <Field label="System prompt (optional, overrides the default persona)">
            <textarea
              className={`${inputCls} min-h-20 resize-y`}
              placeholder={`Default: "You are ${f.title || 'aidekin'}, a friendly assistant…"`}
              value={f.systemPrompt}
              onChange={(e) => set('systemPrompt', e.target.value)}
            />
          </Field>

          <div className="grid grid-cols-2 gap-5">
            <Field label="Accent colour">
              <div className="flex items-center gap-2">
                <input type="color" value={f.accent} onChange={(e) => set('accent', e.target.value)} className="size-10 cursor-pointer rounded-md border border-input bg-background" />
                <input className={inputCls} value={f.accent} onChange={(e) => set('accent', e.target.value)} />
              </div>
            </Field>
            <Field label="Launcher position">
              <select className={inputCls} value={f.position} onChange={(e) => set('position', e.target.value as Form['position'])}>
                <option value="bottom-right">Bottom right</option>
                <option value="bottom-left">Bottom left</option>
              </select>
            </Field>
          </div>

          <Field label="Mode">
            <select className={inputCls} value={f.mode} onChange={(e) => set('mode', e.target.value as Form['mode'])}>
              <option value="text">Text only (recommended)</option>
              <option value="both">Text + voice (beta)</option>
            </select>
            <p className="mt-1.5 text-xs text-muted-foreground">
              Text is light and works on any WebGPU browser. Voice is a beta opt-in: it adds a
              one-time ~1.6&nbsp;GB speech download on first mic tap and works best on desktop.
            </p>
          </Field>

          <Field label="Knowledge file URL (optional, for RAG)">
            <input
              className={inputCls}
              placeholder="https://your-host.com/knowledge.bin"
              value={f.knowledgeUrl}
              onChange={(e) => set('knowledgeUrl', e.target.value)}
            />
            <Link to="/builder" className="mt-1.5 inline-flex items-center gap-1 text-xs font-medium text-primary hover:underline">
              <Database className="size-3.5" /> Build a knowledge file
            </Link>
          </Field>

          <label className="flex cursor-pointer items-center gap-2 text-sm">
            <input type="checkbox" checked={f.persist} onChange={(e) => set('persist', e.target.checked)} className="size-4" />
            Remember the conversation across reloads
          </label>

          <label className="flex cursor-pointer items-start gap-2 text-sm">
            <input type="checkbox" checked={f.reasoning} onChange={(e) => set('reasoning', e.target.checked)} className="mt-0.5 size-4" />
            <span>
              Deeper reasoning on every reply
              <span className="block text-xs text-muted-foreground">More accurate, but slower to start each answer. (RAG answers always reason.)</span>
            </span>
          </label>
        </div>

        {/* Live preview */}
        <div className="lg:sticky lg:top-24 lg:self-start">
          <p className="mono-kicker mb-2">Live preview</p>
          <WidgetFrame className="shadow-xl">
            <WidgetApp config={previewConfig} loadOnMount={false} />
          </WidgetFrame>
          <p className="mt-2 text-xs text-muted-foreground">Type to load the model and try it for real.</p>
        </div>
      </div>

      <Snippet form={f} />
    </section>
  )
}

function Snippet({ form }: { form: Form }) {
  const [tab, setTab] = useState<'hosted' | 'selfhost'>('hosted')
  const [copied, setCopied] = useState(false)
  const src = tab === 'hosted' ? CDN : '/aidekin/loader.js'
  const snippet = useMemo(() => buildSnippet(form, src), [form, src])

  const copy = () => {
    void navigator.clipboard.writeText(snippet).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    })
  }

  return (
    <div className="mt-10">
      <div className="mb-3 flex items-center gap-2">
        <h2 className="font-display text-xl font-semibold tracking-tight">Your snippet</h2>
        <div className="ml-2 flex gap-1 rounded-md bg-secondary p-0.5 text-xs">
          {(['hosted', 'selfhost'] as const).map((t) => (
            <button
              key={t}
              type="button"
              onClick={() => setTab(t)}
              className={`rounded-md px-3 py-1 font-medium transition-colors ${tab === t ? 'bg-background shadow-sm' : 'text-muted-foreground'}`}
            >
              {t === 'hosted' ? 'Hosted' : 'Self-host'}
            </button>
          ))}
        </div>
      </div>
      <div className="relative">
        <pre className="overflow-x-auto rounded-lg border border-border bg-[#0f1a17] p-4 text-sm text-[#e6f0ec]">
          <code>
            {snippet}
            {'\n'}
            <span className="text-[#7fb8a6]">&lt;!-- add this once, before &lt;/body&gt; --&gt;</span>
          </code>
        </pre>
        <button
          type="button"
          onClick={copy}
          className="absolute right-3 top-3 inline-flex items-center gap-1.5 rounded-md bg-white/10 px-3 py-1.5 text-xs font-semibold text-white transition-colors hover:bg-white/20"
        >
          {copied ? <Check className="size-3.5" /> : <Copy className="size-3.5" />}
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
      {tab === 'selfhost' && (
        <p className="mt-2 text-xs text-muted-foreground">
          Self-host: deploy this repo’s build, then point <code className="rounded bg-secondary px-1">src</code> at your own
          <code className="rounded bg-secondary px-1">loader.js</code> (and the iframe origin via <code className="rounded bg-secondary px-1">data-widget-origin</code>).
        </p>
      )}
    </div>
  )
}

const inputCls =
  'w-full rounded-md border border-input bg-background px-3 py-2 text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring'

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="block">
      <span className="mb-1.5 block text-sm font-medium">{label}</span>
      {children}
    </label>
  )
}
