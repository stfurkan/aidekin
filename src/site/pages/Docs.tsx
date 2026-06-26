import { useState, type ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { Check, Copy } from 'lucide-react'

// Docs: a single page with a sticky table-of-contents and anchored sections. No MDX /
// extra deps - just Ledger-styled React. Content tracks the real implementation
// (src/widget/protocol.ts, src/embed/loader.ts, scripts/build-knowledge.ts).

const NAV: { id: string; label: string }[] = [
  { id: 'quickstart', label: 'Quickstart' },
  { id: 'how-it-works', label: 'How it works' },
  { id: 'config', label: 'Configuration' },
  { id: 'api', label: 'JavaScript API' },
  { id: 'knowledge', label: 'Knowledge & RAG' },
  { id: 'voice', label: 'Voice' },
  { id: 'privacy', label: 'Privacy & security' },
  { id: 'self-host', label: 'Self-hosting' },
  { id: 'support', label: 'Browser support' },
]

const SNIPPET = `<script
  src="https://cdn.aidekin.com/loader.js"
  data-title="Acme"
  data-greeting="Hi! How can I help?"
  defer
></script>`

const SNIPPET_RAG = `<script
  src="https://cdn.aidekin.com/loader.js"
  data-mode="both"
  data-knowledge-url="https://cdn.jsdelivr.net/gh/acme/site/knowledge.bin"
  defer
></script>`

const CLI = `npx tsx scripts/build-knowledge.ts \\
  --in ./docs \\
  --url https://acme.com/faq,https://acme.com/pricing \\
  --out ./public/knowledge.bin`

const API = `// Drive it from your own code:
window.Aidekin.open()
window.Aidekin.toggle()
window.Aidekin.on('message', (m) => {
  console.log(m.role, m.text) // 'user' | 'assistant'
})`

const CONFIG_OBJECT = `<script>
  window.AidekinConfig = {
    mode: 'both',
    title: 'Acme',
    accent: '#6d5efc',
    knowledgeUrl: '/knowledge.bin',
  }
</script>
<script src="https://cdn.aidekin.com/loader.js" defer></script>`

const SELF_HOST = `<script
  src="https://your-site.com/aidekin/loader.js"
  data-widget-origin="https://your-site.com/aidekin"
  defer
></script>`

export default function Docs() {
  return (
    <section className="mx-auto max-w-6xl px-5 py-14">
      <p className="mono-kicker">Docs</p>
      <h1 className="mt-3 font-display text-4xl font-semibold tracking-tight">Embed aidekin</h1>
      <p className="mt-3 max-w-2xl text-lg text-muted-foreground">
        One script tag gives your visitors a private voice and text assistant that runs entirely in
        their browser. No backend, no API keys, no per-message cost.
      </p>

      <div className="mt-10 grid gap-12 lg:grid-cols-[200px_1fr]">
        <nav className="hidden lg:block">
          <div className="sticky top-24 space-y-1">
            <p className="mono-kicker mb-2">On this page</p>
            {NAV.map((n) => (
              <a
                key={n.id}
                href={`#${n.id}`}
                className="block rounded-md px-2 py-1 text-sm text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
              >
                {n.label}
              </a>
            ))}
          </div>
        </nav>

        <div className="min-w-0 max-w-2xl">
          <Doc id="quickstart" title="Quickstart">
            <P>
              Paste this once, just before <Code>{'</body>'}</Code>. A launcher button appears in the
              corner; everything else loads only when a visitor opens it.
            </P>
            <CodeBlock label="index.html" code={SNIPPET} />
            <P>
              That is the whole install. Get a snippet tailored to your settings on the{' '}
              <Link to="/configure" className="text-primary hover:underline">
                configure
              </Link>{' '}
              page, or read the <A href="#config">configuration reference</A> below.
            </P>
            <Callout>
              The first visit downloads the model (about 290&nbsp;MB for text) and caches it in the
              browser. Repeat visits load from cache. Nothing is sent to a server.
            </Callout>
          </Doc>

          <Doc id="how-it-works" title="How it works">
            <P>
              The loader is about 2&nbsp;KB. On page load it draws only the launcher inside a Shadow
              DOM (so your CSS can never touch it). On the first open it creates a sandboxed iframe
              that hosts the widget app and starts downloading the model. The conversation, retrieval,
              and speech all run inside the visitor&rsquo;s browser; the host page and the iframe talk
              over an origin-checked <Code>postMessage</Code> channel.
            </P>
            <List
              items={[
                ['Page load', 'just the ~2 KB loader + a launcher button. Zero impact on your page.'],
                ['First open', 'the iframe + widget load; the model downloads once (with a progress bar and an estimate).'],
                ['First mic tap', 'the speech models load (voice is opt-in, never before).'],
              ]}
            />
          </Doc>

          <Doc id="config" title="Configuration">
            <P>
              Configure with <Code>data-*</Code> attributes on the script tag, or with a{' '}
              <Code>window.AidekinConfig</Code> object set before the loader runs. Attributes win if
              both are present.
            </P>
            <ConfigTable />
            <P className="mt-6">Object form (useful when values are dynamic):</P>
            <CodeBlock code={CONFIG_OBJECT} />
            <P className="mt-6">
              <strong>Theme:</strong> <Code>data-theme</Code> is <Code>light</Code>, <Code>dark</Code>,
              or <Code>auto</Code> (the default, which follows the visitor&rsquo;s operating-system
              color scheme). The widget runs in a sandboxed iframe, so it cannot read your page&rsquo;s
              CSS. To keep it in sync with your own light/dark toggle, call{' '}
              <Code>window.Aidekin.setTheme('dark')</Code> (or <Code>'light'</Code>) when your theme
              changes. Visitors can also switch light or dark themselves from the sun/moon button next
              to the widget&rsquo;s settings menu, and their choice is remembered.
            </P>
            <P className="mt-6">
              <strong>Custom system prompt and RAG:</strong> a custom system prompt changes the
              assistant&rsquo;s persona only. It does not disable retrieval. With{' '}
              <Code>data-knowledge-url</Code> set, answers are still grounded in your knowledge file
              regardless of the prompt.
            </P>
          </Doc>

          <Doc id="api" title="JavaScript API">
            <P>
              Once loaded, the widget exposes <Code>window.Aidekin</Code>:
            </P>
            <List
              items={[
                ['open() / close() / toggle()', 'show or hide the panel from your own buttons.'],
                ["setTheme('light' | 'dark')", "force the panel's theme, e.g. to match your site's own toggle."],
                ["on('open' | 'close' | 'ready' | 'message', cb)", 'subscribe to events. The message event fires for each user and assistant turn.'],
              ]}
            />
            <CodeBlock code={API} />
            <Callout>
              Only the embedding page can drive the widget by default. To allow other origins (for
              example a parent frame), list them in <Code>data-allowed-origins</Code>.
            </Callout>
          </Doc>

          <Doc id="knowledge" title="Knowledge & RAG">
            <P>
              To ground answers in your own content, build a <Code>knowledge.bin</Code> file and point
              the widget at it with <Code>data-knowledge-url</Code>. Retrieval runs in the browser; at
              query time only the visitor&rsquo;s question is embedded. Omit the URL and no RAG code or
              embedder ever loads.
            </P>
            <P>
              Build it in your browser on the{' '}
              <Link to="/knowledge" className="text-primary hover:underline">
                knowledge builder
              </Link>{' '}
              (drop in PDF, Word, Markdown, HTML, or text files; paste text; or add URLs), or from the
              command line:
            </P>
            <CodeBlock label="terminal" code={CLI} />
            <P>Host the small file anywhere that allows cross-origin reads, then reference it:</P>
            <CodeBlock label="index.html" code={SNIPPET_RAG} />
            <P>
              Good hosts (all independent services, aidekin is not affiliated with any): a GitHub repo
              served through a CDN like jsDelivr (<Code>cdn.jsdelivr.net/gh/...</Code>), Cloudflare
              R2, or your own server with <Code>Access-Control-Allow-Origin: *</Code>. Tune how many
              chunks are retrieved with <Code>data-rag-top-k</Code> (default 3).
            </P>
            <Callout tone="warn">
              The knowledge file is downloaded by every visitor, so treat it as <strong>public</strong>.
              Never put secrets, internal notes, or personal data in it.
            </Callout>
          </Doc>

          <Doc id="voice" title="Voice (beta)">
            <P>
              Voice is a <strong>beta</strong> opt-in; <strong>text is the recommended default</strong>{' '}
              for most sites (lighter, broader device support). Set <Code>data-mode="both"</Code> for text
              with a voice toggle, or <Code>data-mode="voice"</Code> for a voice-first experience. Voice
              still uses the same language model as text (one shared brain), plus speech recognition and
              synthesis. Those speech models (about 1.6&nbsp;GB) load only the first time a visitor taps the
              mic, never before - and if a visitor backs out mid-download, the partial files are cleaned up.
            </P>
            <P>
              Text works on any WebGPU browser with no special setup. Voice&rsquo;s fastest path uses
              threaded WebAssembly, which needs the <strong>embedding page</strong> to be cross-origin
              isolated. On a page that is not isolated, voice runs single-threaded if it can keep up,
              and otherwise stays text. For guaranteed full-speed voice, serve your page with:
            </P>
            <CodeBlock code={'Cross-Origin-Opener-Policy: same-origin\nCross-Origin-Embedder-Policy: require-corp'} />
          </Doc>

          <Doc id="privacy" title="Privacy & security">
            <P>
              Everything runs on the visitor&rsquo;s device. There is no backend, no API key, and no
              per-message cost, so there is nothing on your side for a visitor to steal or run up.
            </P>
            <List
              items={[
                ['The system prompt is config, not a secret', 'it is delivered to the browser. Set the persona, but never put credentials or private instructions you need to keep hidden.'],
                ['The knowledge file is public', 'every visitor downloads it. Keep secrets and PII out of it.'],
                ['The widget is sandboxed', 'it runs in a sandboxed iframe and only requests the mic when voice is used.'],
                ['Cross-origin messages are checked', 'only the embedding page (plus any data-allowed-origins) can drive the widget.'],
                ['Honest limit', 'because it all runs locally, a determined visitor can inspect or alter the prompt, config, and model on their own machine. Nothing to lose at your expense, but do not treat the prompt as a security boundary.'],
              ]}
            />
          </Doc>

          <Doc id="self-host" title="Self-hosting">
            <P>
              Prefer to serve everything yourself? Build this repo, deploy the output to a static host,
              then point the loader at your own copy and tell it where the widget app lives:
            </P>
            <CodeBlock label="index.html" code={SELF_HOST} />
            <P>
              Serve the widget path (<Code>/aidekin/widget/</Code>) with{' '}
              <Code>Cross-Origin-Opener-Policy: same-origin</Code> and{' '}
              <Code>Cross-Origin-Embedder-Policy: require-corp</Code> if you want voice. The model
              weights can still stream from the public CDN, or you can mirror them and set{' '}
              <Code>VITE_MODEL_CDN</Code> at build time.
            </P>
          </Doc>

          <Doc id="support" title="Browser support">
            <P>
              aidekin needs <strong>WebGPU</strong>, which is available on recent desktop Chrome and
              Edge, Safari 26+, and Firefox 145+. On an unsupported browser the widget shows a short,
              friendly notice instead of failing.
            </P>
            <List
              items={[
                ['One-time download', 'about 290 MB for text. Voice adds about 1.6 GB more the first time it is used. Cached after the first load.'],
                ['Per-site caching', 'browsers partition storage by site, so a visitor downloads once per site that embeds the widget, then loads from cache. This is a privacy protection, not a bug.'],
                ['Offline', 'after the first load, it keeps working with no network.'],
              ]}
            />
          </Doc>

          <div className="mt-12 border-t border-border pt-6">
            <p className="text-sm text-muted-foreground">
              Ready to ship?{' '}
              <Link to="/configure" className="text-primary hover:underline">
                Build your snippet
              </Link>
              .
            </p>
          </div>
        </div>
      </div>
    </section>
  )
}

// ── content primitives ────────────────────────────────────────────────────────
function Doc({ id, title, children }: { id: string; title: string; children: ReactNode }) {
  return (
    <section id={id} className="scroll-mt-24 border-t border-border py-8 first:border-t-0 first:pt-0">
      <h2 className="font-display text-2xl font-semibold tracking-tight">{title}</h2>
      <div className="mt-4 space-y-4">{children}</div>
    </section>
  )
}

function P({ children, className }: { children: ReactNode; className?: string }) {
  return <p className={`text-[15px] leading-relaxed text-muted-foreground ${className ?? ''}`}>{children}</p>
}

function A({ href, children }: { href: string; children: ReactNode }) {
  return (
    <a href={href} className="text-primary hover:underline">
      {children}
    </a>
  )
}

function Code({ children }: { children: ReactNode }) {
  return (
    <code className="rounded bg-secondary px-1.5 py-0.5 font-mono text-[0.85em] text-foreground">{children}</code>
  )
}

function List({ items }: { items: [string, string][] }) {
  return (
    <ul className="space-y-3">
      {items.map(([term, desc]) => (
        <li key={term} className="border-l-2 border-border pl-4">
          <span className="font-mono text-[13px] font-medium text-foreground">{term}</span>
          <span className="mt-0.5 block text-[15px] leading-relaxed text-muted-foreground">{desc}</span>
        </li>
      ))}
    </ul>
  )
}

function Callout({ children, tone = 'note' }: { children: ReactNode; tone?: 'note' | 'warn' }) {
  return (
    <div
      className={
        tone === 'warn'
          ? 'rounded-lg border border-destructive/30 bg-destructive/[0.06] p-4 text-[14px] leading-relaxed text-foreground'
          : 'rounded-lg border border-primary/25 bg-primary/[0.05] p-4 text-[14px] leading-relaxed text-foreground'
      }
    >
      {children}
    </div>
  )
}

function CodeBlock({ code, label }: { code: string; label?: string }) {
  const [copied, setCopied] = useState(false)
  const copy = () => {
    navigator.clipboard?.writeText(code).then(
      () => {
        setCopied(true)
        setTimeout(() => setCopied(false), 1500)
      },
      () => {},
    )
  }
  return (
    <div className="overflow-hidden rounded-lg border border-border bg-[#0c0e11]">
      <div className="flex items-center justify-between border-b border-white/10 px-4 py-2">
        <span className="font-mono text-xs text-white/40">{label ?? 'code'}</span>
        <button
          type="button"
          onClick={copy}
          className="inline-flex items-center gap-1.5 font-mono text-xs text-white/55 transition-colors hover:text-white"
        >
          {copied ? <Check className="size-3.5" /> : <Copy className="size-3.5" />}
          {copied ? 'copied' : 'copy'}
        </button>
      </div>
      <pre className="overflow-x-auto px-4 py-3.5 font-mono text-[13px] leading-[1.7] text-[#dfe3e6]">
        <code>{code}</code>
      </pre>
    </div>
  )
}

function ConfigTable() {
  const rows: [string, string, string][] = [
    ['data-mode', 'text · voice · both', 'text'],
    ['data-title', 'string', 'Assistant'],
    ['data-greeting', 'string', 'none'],
    ['data-system-prompt', 'string', 'auto from title'],
    ['data-accent', 'CSS color', 'brand jade'],
    ['data-position', 'bottom-right · bottom-left', 'bottom-right'],
    ['data-launcher-label', 'string', 'Chat with us'],
    ['data-knowledge-url', 'URL', 'none'],
    ['data-rag-top-k', 'number', '3'],
    ['data-reasoning', 'true · false', 'false'],
    ['data-persist', 'true · false', 'true'],
    ['data-theme', 'light · dark · auto', 'auto'],
    ['data-allowed-origins', 'comma-separated', 'embedding page'],
    ['data-widget-origin', 'URL', "loader's origin"],
  ]
  return (
    <div className="overflow-hidden rounded-lg border border-border">
      <div className="grid grid-cols-[1.4fr_1fr_0.9fr] border-b border-border bg-card font-mono text-[11px] uppercase tracking-wider text-muted-foreground">
        <div className="p-3">attribute</div>
        <div className="p-3">values</div>
        <div className="p-3">default</div>
      </div>
      {rows.map((r, i) => (
        <div
          key={r[0]}
          className={`grid grid-cols-[1.4fr_1fr_0.9fr] border-b border-border text-[13px] last:border-b-0 ${i % 2 === 1 ? 'bg-card/40' : ''}`}
        >
          <div className="p-3 font-mono text-foreground">{r[0]}</div>
          <div className="p-3 font-mono text-muted-foreground">{r[1]}</div>
          <div className="p-3 text-muted-foreground">{r[2]}</div>
        </div>
      ))}
    </div>
  )
}
