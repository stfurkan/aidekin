import { type ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { ArrowRight, Play, Check, X } from 'lucide-react'
import { cn } from '@/lib/utils'
import { HeroChat } from '../HeroChat'

const BTN_PRIMARY =
  'inline-flex items-center gap-2 rounded-md bg-primary px-5 py-2.5 text-sm font-semibold text-primary-foreground transition-transform hover:-translate-y-px'
const BTN_GHOST =
  'inline-flex items-center gap-2 rounded-md border border-border bg-card px-5 py-2.5 text-sm font-semibold transition-colors hover:bg-secondary'

export function Landing() {
  return (
    <>
      <Hero />
      <StatLedger />
      <HowItWorks />
      <Features />
      <Comparison />
      <Faq />
      <CtaBand />
    </>
  )
}

function Hero() {
  return (
    <section className="relative overflow-hidden border-b border-border">
      <div className="substrate pointer-events-none absolute inset-0 -z-10 opacity-50" />
      <div className="mx-auto grid max-w-6xl items-center gap-12 px-5 py-20 md:grid-cols-[1.05fr_0.95fr] md:py-28">
        <div>
          <p className="mono-kicker">00 / Entirely in the browser</p>
          <h1 className="mt-5 text-balance font-display text-4xl font-semibold leading-[1.05] tracking-tight sm:text-5xl md:text-[3.35rem]">
            An AI agent that runs inside your visitor’s <span className="ink-accent">browser</span>.
          </h1>
          <p className="mt-5 max-w-md text-pretty text-lg text-muted-foreground">
            One script tag. The model, retrieval, and voice all execute on-device. No backend, no
            API keys, nothing leaves the page.
          </p>
          <div className="mt-8 flex flex-wrap items-center gap-3">
            <Link to="/configure" className={BTN_PRIMARY}>
              Get the snippet <ArrowRight className="size-4" />
            </Link>
            <Link to="/demo" className={BTN_GHOST}>
              <Play className="size-4" /> Live demo
            </Link>
          </div>
          <p className="mono-kicker mt-6">Open source · MIT · works offline after first load</p>
        </div>
        <HeroChat />
      </div>
    </section>
  )
}

function StatLedger() {
  const stats: Array<[string, string]> = [
    ['0', 'servers'],
    ['~290 MB', 'text, downloaded once'],
    ['1', 'script tag'],
    ['$0', 'per message'],
  ]
  return (
    <div className="mx-auto max-w-6xl px-5">
      <div className="grid grid-cols-2 border-b border-border md:grid-cols-4">
        {stats.map(([n, l], i) => (
          <div
            key={l}
            className={cn(
              'px-5 py-8',
              i > 0 && 'border-l border-border',
              i === 2 && 'border-l-0 md:border-l',
            )}
          >
            <div className="font-mono text-3xl font-medium tracking-tight md:text-4xl">{n}</div>
            <div className="mt-1.5 text-sm text-muted-foreground">{l}</div>
          </div>
        ))}
      </div>
    </div>
  )
}

function HowItWorks() {
  const steps = [
    { n: '01', t: 'Paste one line', b: 'Add the script tag. A launcher appears, with no build step and no backend.' },
    { n: '02', t: 'Feed it your docs (optional)', b: 'Build a knowledge file in your browser and point the widget at it.' },
    { n: '03', t: 'That’s it', b: 'Visitors chat by text or voice. Everything runs on their device, and you pay nothing per message.' },
  ]
  return (
    <Section index="01 / How it works" title="Live in three steps" subtitle="From zero to your own assistant in a couple of minutes.">
      <div className="grid gap-px overflow-hidden rounded-lg border border-border bg-border md:grid-cols-3">
        {steps.map((s) => (
          <div key={s.n} className="bg-background p-6">
            <span className="font-mono text-sm text-primary">{s.n}</span>
            <h3 className="mt-3 font-display text-lg font-semibold">{s.t}</h3>
            <p className="mt-2 text-sm text-muted-foreground">{s.b}</p>
          </div>
        ))}
      </div>
    </Section>
  )
}

function Features() {
  const items = [
    { t: 'Text chat', d: 'A fast, lightweight chat that works on any modern WebGPU browser.' },
    { t: 'Voice, opt-in', d: 'Flip on the mic and the same agent talks back. Speech models load only when used.' },
    { t: 'Local RAG', d: 'Feed it your own content. Retrieval runs in the browser; your data never leaves it.' },
    { t: 'Zero per-message cost', d: 'The model runs on the visitor’s device. No tokens, no metering, no surprise bills.' },
    { t: 'Private by design', d: 'No backend, no API keys, nothing to leak. Works behind corporate firewalls.' },
    { t: 'Open source', d: 'MIT licensed. Self-host it, fork it, or use the free hosted snippet.' },
  ]
  return (
    <Section index="02 / Capabilities" title="Everything on-device" subtitle="A real assistant, without the cloud bill or the privacy trade-off.">
      <div className="grid gap-x-10 gap-y-8 sm:grid-cols-2 lg:grid-cols-3">
        {items.map((it) => (
          <div key={it.t} className="border-t border-border pt-4">
            <h3 className="font-display text-base font-semibold">{it.t}</h3>
            <p className="mt-2 text-sm text-muted-foreground">{it.d}</p>
          </div>
        ))}
      </div>
    </Section>
  )
}

function Comparison() {
  const rows = [
    ['Per-message cost', 'Per minute / per token', 'Free, runs on the visitor’s device'],
    ['Your visitors’ data', 'Sent to a vendor’s servers', 'Never leaves the browser'],
    ['API keys & secrets', 'Required', 'None'],
    ['Setup', 'Backend + billing account', 'One script tag'],
    ['Works offline', 'No', 'Yes, after first load'],
  ]
  return (
    <Section index="03 / Why local-first" title="The drop-in convenience, without the strings" subtitle="Everything a cloud chat widget gives you, minus the bill and the privacy trade-off.">
      <div className="overflow-hidden rounded-lg border border-border">
        <div className="grid grid-cols-3 border-b border-border bg-card font-mono text-xs uppercase tracking-wider">
          <div className="p-4 text-muted-foreground">&nbsp;</div>
          <div className="p-4 text-muted-foreground">Cloud widgets</div>
          <div className="p-4 text-primary">aidekin</div>
        </div>
        {rows.map((r, i) => (
          <div
            key={r[0]}
            className={cn('grid grid-cols-3 border-b border-border text-sm last:border-b-0', i % 2 === 1 && 'bg-card/40')}
          >
            <div className="p-4 font-medium">{r[0]}</div>
            <div className="flex items-start gap-2 p-4 text-muted-foreground">
              <X className="mt-0.5 size-4 shrink-0 text-destructive/70" /> {r[1]}
            </div>
            <div className="flex items-start gap-2 p-4">
              <Check className="mt-0.5 size-4 shrink-0 text-primary" /> {r[2]}
            </div>
          </div>
        ))}
      </div>
    </Section>
  )
}

function Faq() {
  const qa = [
    {
      q: 'Is it really free?',
      a: 'The widget is open source (MIT) and the hosted snippet runs on a free static host. The model streams once from a public CDN and caches in the visitor’s browser. There are no per-message charges.',
    },
    {
      q: 'What does the visitor download?',
      a: 'About 290 MB for text chat (the language model), once, then cached. Turning on voice adds its speech models (about 1.6 GB) the first time you use it, also cached. After that it works offline. It runs on WebGPU, so visitors on recent desktop browsers get the full experience.',
    },
    {
      q: 'Where does my data go?',
      a: 'Nowhere. Inference and retrieval happen entirely in the visitor’s browser. Your knowledge file is downloaded by visitors, so treat it as public. Never put secrets in it.',
    },
    {
      q: 'Can I use my own content?',
      a: 'Yes. Build a knowledge file in your browser (or via the CLI), host the small file anywhere, and point the widget at it. Answers get grounded in your content.',
    },
  ]
  return (
    <Section index="04 / Questions" title="The short version">
      <div className="max-w-3xl divide-y divide-border border-y border-border">
        {qa.map((item) => (
          <details key={item.q} className="group py-4">
            <summary className="flex cursor-pointer list-none items-center justify-between gap-4 font-medium">
              {item.q}
              <span className="font-mono text-muted-foreground transition-transform group-open:rotate-45">+</span>
            </summary>
            <p className="mt-3 max-w-2xl text-sm text-muted-foreground">{item.a}</p>
          </details>
        ))}
      </div>
    </Section>
  )
}

function CtaBand() {
  return (
    <section className="mx-auto max-w-6xl px-5 pb-24 pt-4">
      <div className="rounded-lg border border-border bg-card p-8 md:p-12">
        <div className="grid items-center gap-8 md:grid-cols-[1fr_auto]">
          <div>
            <h2 className="font-display text-2xl font-semibold tracking-tight sm:text-3xl">
              Give your site a voice, for free.
            </h2>
            <p className="mt-3 max-w-md text-muted-foreground">
              Configure it once, copy the snippet, paste it in. Your visitors get a private assistant
              in minutes.
            </p>
          </div>
          <div className="flex flex-wrap gap-3">
            <Link to="/configure" className={BTN_PRIMARY}>
              Build your widget <ArrowRight className="size-4" />
            </Link>
            <Link to="/demo" className={BTN_GHOST}>
              <Play className="size-4" /> See the demo
            </Link>
          </div>
        </div>
      </div>
    </section>
  )
}

function Section({
  index,
  title,
  subtitle,
  children,
}: {
  index: string
  title: string
  subtitle?: string
  children: ReactNode
}) {
  return (
    <section className="mx-auto max-w-6xl px-5 py-16 md:py-20">
      <div className="mb-10 border-t border-border pt-6">
        <p className="mono-kicker">{index}</p>
        <h2 className="mt-3 font-display text-3xl font-semibold tracking-tight sm:text-4xl">{title}</h2>
        {subtitle && <p className="mt-3 max-w-2xl text-lg text-muted-foreground">{subtitle}</p>}
      </div>
      {children}
    </section>
  )
}
