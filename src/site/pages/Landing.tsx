import { useEffect, useRef, useState, type ReactNode } from 'react'
import { Link } from 'react-router-dom'
import {
  ArrowRight,
  ArrowUpRight,
  Check,
  Coins,
  Database,
  GitFork,
  MessageSquareText,
  Mic,
  Play,
  ShieldCheck,
  X,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { DEMO_URL } from '../Layout'
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
      <DemoProof />
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
            An AI assistant that runs inside your visitor’s <span className="ink-accent">browser</span>.
          </h1>
          <p className="mt-5 max-w-md text-pretty text-lg text-muted-foreground">
            One script tag. The model, retrieval, and voice all execute on-device. No backend, no
            API keys, nothing leaves their device.
          </p>
          <div className="mt-8 flex flex-wrap items-center gap-3">
            <Link to="/configure" className={BTN_PRIMARY}>
              Get your snippet <ArrowRight className="size-4" />
            </Link>
            <a href={DEMO_URL} target="_blank" rel="noreferrer" className={BTN_GHOST}>
              <Play className="size-4" /> Live demo
            </a>
          </div>
          <p className="mono-kicker mt-6">Open source · MIT · works offline after first load</p>
        </div>
        <div>
          <HeroChat />
          <p className="mt-3 text-center">
            <button
              type="button"
              onClick={() => (window as Window & { Aidekin?: { open?: () => void } }).Aidekin?.open?.()}
              className="font-mono text-xs uppercase tracking-wider text-muted-foreground underline-offset-4 transition-colors hover:text-primary hover:underline"
            >
              preview reel · the real one lives in the corner →
            </button>
          </p>
        </div>
      </div>
    </section>
  )
}

function StatLedger() {
  const stats: Array<[string, string]> = [
    ['0', 'servers'],
    ['~290 MB', 'one-time download'],
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
    {
      n: '01',
      t: 'Paste one line',
      b: 'A launcher appears in the corner. No build step, no backend, no account.',
      code: '<script src="https://cdn.aidekin.com/loader.js" defer></script>',
    },
    { n: '02', t: 'Ground it in your content', b: 'Optional: build a knowledge file in your browser and point the widget at it. Answers then come from your pages, not thin air.' },
    { n: '03', t: 'That’s it', b: 'Visitors chat by text or voice. Everything runs on their device, and you pay nothing per message.' },
  ]
  return (
    <Section index="01 / How it works" title="Live in three steps" subtitle="From zero to your own assistant in a couple of minutes.">
      <div className="grid gap-px overflow-hidden rounded-lg border border-border bg-border md:grid-cols-3">
        {steps.map((s) => (
          <div key={s.n} className="flex flex-col bg-background p-6">
            <span className="font-mono text-sm text-primary">{s.n}</span>
            <h3 className="mt-3 font-display text-lg font-semibold">{s.t}</h3>
            <p className="mt-2 text-sm text-muted-foreground">{s.b}</p>
            {s.code && (
              <code className="mt-4 block overflow-x-auto whitespace-nowrap rounded-md border border-border bg-card px-3 py-2 font-mono text-xs text-muted-foreground">
                {s.code}
              </code>
            )}
          </div>
        ))}
      </div>
    </Section>
  )
}

function Features() {
  const items = [
    { icon: MessageSquareText, t: 'Text chat', d: 'The recommended default: a fast, lightweight chat that works on any modern WebGPU browser.' },
    { icon: Mic, t: 'Voice · beta', d: 'Flip on the mic and the same assistant talks back. The speech models (~1.6 GB) download only on first use, best on desktop.' },
    { icon: Database, t: 'Local RAG', d: 'Feed it your own content. Retrieval runs in the browser; your data never leaves it.' },
    { icon: Coins, t: 'Zero per-message cost', d: 'The model runs on the visitor’s device. No tokens, no metering, no surprise bills.' },
    { icon: ShieldCheck, t: 'Private by design', d: 'No backend, no API keys, nothing to leak. Works behind corporate firewalls.' },
    { icon: GitFork, t: 'Open source', d: 'MIT licensed. Self-host it, fork it, or use the free hosted snippet.' },
  ]
  return (
    <Section index="02 / Capabilities" title="Everything on-device" subtitle="A real assistant, without the cloud bill or the privacy trade-off.">
      <div className="grid gap-x-10 gap-y-8 sm:grid-cols-2 lg:grid-cols-3">
        {items.map((it) => (
          <div key={it.t} className="border-t border-border pt-4">
            <h3 className="flex items-center gap-2 font-display text-base font-semibold">
              <it.icon className="size-4 text-primary" aria-hidden="true" />
              {it.t}
            </h3>
            <p className="mt-2 text-sm text-muted-foreground">{it.d}</p>
          </div>
        ))}
      </div>
    </Section>
  )
}

function Comparison() {
  const rows = [
    ['Per-message cost', 'Per minute / per token', '$0, it runs on the visitor’s device'],
    ['Your visitors’ data', 'Sent to a vendor’s servers', 'Never leaves the browser'],
    ['API keys & secrets', 'Required', 'None'],
    ['Setup', 'Backend + billing account', 'One script tag'],
    ['Works offline', 'No', 'Yes, after first load'],
  ]
  return (
    <Section index="03 / Why local-first" title="The drop-in convenience, without the strings" subtitle="Everything a cloud chat widget gives you, minus the bill and the privacy trade-off.">
      <div className="overflow-hidden rounded-lg border border-border">
        {/* On phones the label gets its own line and the two value cells share the row. */}
        <div className="grid grid-cols-2 border-b border-border bg-card font-mono text-xs uppercase tracking-wider md:grid-cols-3">
          <div className="hidden p-4 text-muted-foreground md:block">&nbsp;</div>
          <div className="p-4 text-muted-foreground">Cloud widgets</div>
          <div className="p-4 text-primary">aidekin</div>
        </div>
        {rows.map((r, i) => (
          <div
            key={r[0]}
            className={cn('grid grid-cols-2 border-b border-border text-sm last:border-b-0 md:grid-cols-3', i % 2 === 1 && 'bg-card/40')}
          >
            <div className="col-span-2 px-4 pt-3 font-medium md:col-span-1 md:p-4">{r[0]}</div>
            <div className="flex items-start gap-2 p-4 pt-2 text-muted-foreground md:pt-4">
              <X className="mt-0.5 size-4 shrink-0 text-destructive/70" /> {r[1]}
            </div>
            <div className="flex items-start gap-2 p-4 pt-2 md:pt-4">
              <Check className="mt-0.5 size-4 shrink-0 text-primary" /> {r[2]}
            </div>
          </div>
        ))}
      </div>
      <CostLedger />
    </Section>
  )
}

/** The $0 argument, made concrete: drag the volume, watch the metered bill move while
 *  aidekin's line stays put. Deliberately conservative math, stated inline. */
function CostLedger() {
  const [monthly, setMonthly] = useState(5000)
  const RATE = 0.02 // $/message; metered chat AI commonly lands between $0.01 and $0.03
  const perMonth = monthly * RATE
  const money = (n: number) =>
    n.toLocaleString('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 })
  return (
    <div className="mt-6 rounded-lg border border-border bg-card p-6 md:p-8">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <label htmlFor="msgs" className="font-display text-base font-semibold">
          Run the numbers
        </label>
        <span className="font-mono text-sm text-muted-foreground">
          {monthly.toLocaleString('en-US')} messages / month
        </span>
      </div>
      <input
        id="msgs"
        type="range"
        min={500}
        max={100000}
        step={500}
        value={monthly}
        onChange={(e) => setMonthly(Number(e.target.value))}
        className="mt-4 w-full accent-primary"
      />
      <div className="mt-6 grid gap-px overflow-hidden rounded-md border border-border bg-border sm:grid-cols-2">
        <div className="bg-background p-5">
          <p className="font-mono text-xs uppercase tracking-wider text-muted-foreground">Metered widget</p>
          <p className="mt-2 font-mono text-3xl font-medium tracking-tight">
            {money(perMonth)}
            <span className="text-base text-muted-foreground"> /mo</span>
          </p>
          <p className="mt-1 text-sm text-muted-foreground">{money(perMonth * 12)} a year</p>
        </div>
        <div className="bg-background p-5">
          <p className="font-mono text-xs uppercase tracking-wider text-primary">aidekin</p>
          <p className="mt-2 font-mono text-3xl font-medium tracking-tight">
            $0<span className="text-base text-muted-foreground"> /mo</span>
          </p>
          <p className="mt-1 text-sm text-muted-foreground">$0 a year, at any volume</p>
        </div>
      </div>
      <p className="mt-4 text-xs text-muted-foreground">
        Metered chat AI commonly lands between $0.01 and $0.03 per message; the ledger charts $0.02.
        aidekin’s cost stays $0 because the visitor’s device does the work.
      </p>
    </div>
  )
}

function DemoProof() {
  const videoRef = useRef<HTMLVideoElement>(null)
  useEffect(() => {
    const v = videoRef.current
    if (!v) return
    v.muted = true // set imperatively so muted autoplay is allowed in every browser
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)')
    // It sits below the fold, so only play while visible, and never when the visitor
    // asked for reduced motion - the poster still stays up in that case.
    const io = new IntersectionObserver(
      ([entry]) => {
        if (reduce.matches) return
        if (entry.isIntersecting) void v.play().catch(() => {})
        else v.pause()
      },
      { threshold: 0.25 },
    )
    io.observe(v)
    return () => io.disconnect()
  }, [])
  return (
    <Section
      index="04 / On a real site"
      title="See it working, not a mockup"
      subtitle="Copperleaf Café is a fictional shop on its own domain with the widget pasted in - the same one script tag you would use."
    >
      <a
        href={DEMO_URL}
        target="_blank"
        rel="noreferrer"
        className="group block overflow-hidden rounded-lg border border-border bg-card transition-transform hover:-translate-y-0.5"
      >
        <div className="flex items-center gap-2 border-b border-border px-4 py-2.5">
          <span className="flex gap-1.5" aria-hidden="true">
            <span className="size-2.5 rounded-full border border-border bg-secondary" />
            <span className="size-2.5 rounded-full border border-border bg-secondary" />
            <span className="size-2.5 rounded-full border border-border bg-secondary" />
          </span>
          <span className="ml-2 flex-1 truncate rounded-md bg-background px-3 py-1 font-mono text-xs text-muted-foreground">
            stfurkan.github.io/aidekin-demo
          </span>
          <ArrowUpRight className="size-4 text-muted-foreground transition-colors group-hover:text-foreground" />
        </div>
        <video
          ref={videoRef}
          src="/copperleaf-demo.mp4"
          poster="/copperleaf-demo.jpg"
          muted
          loop
          playsInline
          preload="none"
          width={1280}
          height={800}
          aria-label="The aidekin widget answering questions from the café's own knowledge file on the Copperleaf Café demo site"
          className="block w-full"
        />
      </a>
      <p className="mt-4 text-sm text-muted-foreground">
        Ask it about opening hours, allergens, or dogs on the patio: it answers from the café’s own
        knowledge file, on your device.
      </p>
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
    {
      q: 'Which devices does it work on?',
      a: 'Recent desktop Chrome, Edge, Safari 26+, and Firefox 145+, plus current phones - text chat streams at reading speed on an iPhone 14 Pro. Devices that can’t run it get a friendly notice instead of a crash, and if a device turns out to be on the slow side, the widget says so rather than leaving visitors guessing.',
    },
    {
      q: 'Is a model this small actually useful?',
      a: 'It’s a 1.7-billion-parameter model: closer to a sharp product specialist than a general genius. Grounded in your knowledge file it answers questions about your product precisely, quotes your content, and admits what it doesn’t know. For open-ended reasoning on arbitrary topics a big cloud model still wins - that’s the honest trade for $0 and full privacy.',
    },
  ]
  return (
    <Section index="05 / Questions" title="Common questions, short answers">
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
              Your site’s own assistant. Free, forever.
            </h2>
            <p className="mt-3 max-w-md text-muted-foreground">
              Configure it once, copy the snippet, paste it in. Your visitors get a private
              assistant that costs you nothing per message.
            </p>
          </div>
          <div className="flex flex-wrap gap-3">
            <Link to="/configure" className={BTN_PRIMARY}>
              Get your snippet <ArrowRight className="size-4" />
            </Link>
            <a href={DEMO_URL} target="_blank" rel="noreferrer" className={BTN_GHOST}>
              <Play className="size-4" /> See the demo
            </a>
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
    <section className="mx-auto max-w-6xl px-5 py-12 md:py-16">
      <div className="mb-8 border-t border-border pt-6">
        <p className="mono-kicker">{index}</p>
        <h2 className="mt-3 font-display text-3xl font-semibold tracking-tight sm:text-4xl">{title}</h2>
        {subtitle && <p className="mt-3 max-w-2xl text-lg text-muted-foreground">{subtitle}</p>}
      </div>
      {children}
    </section>
  )
}
