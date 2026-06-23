// The widget panel UI. Lives inside the iframe (embedded) OR inline on the site (demo /
// configurator preview). Probes capabilities, then shows either a Type/Talk picker
// (text+voice widgets), the text chat, or the immersive voice view. Falls back to a
// friendly notice when WebGPU is missing.

import {
  useEffect,
  useRef,
  useState,
  type CSSProperties,
  type KeyboardEvent,
  type ReactNode,
} from 'react'
import { Send, Square, Trash2, X, Loader2, Settings, HardDrive, Mic, Keyboard, Sun, Moon } from 'lucide-react'
import {
  probeCapabilities,
  resolveWidgetCapabilities,
  type WidgetCapabilities,
} from '@/core/capabilities'
import { cn } from '@/lib/utils'
import { AidekinMark } from '@/site/icons'
import { useTextController, type WidgetTurn } from './useTextController'
import { SonarPing } from './SonarPing'
import { Markdown } from './Markdown'
import type { WidgetConfig } from './protocol'

interface Props {
  config: WidgetConfig
  /** Inside the iframe → show a close button that asks the host to hide us. */
  embedded?: boolean
  persistKey?: string
  /** Begin model download on mount. Default true; configurator preview passes false. */
  loadOnMount?: boolean
  onMessage?: (role: 'user' | 'assistant', text: string) => void
  onClose?: () => void
}

export function WidgetApp({ config, embedded, persistKey, loadOnMount, onMessage, onClose }: Props) {
  const [caps, setCaps] = useState<WidgetCapabilities | null>(null)
  const [tryAnyway, setTryAnyway] = useState(false)

  useEffect(() => {
    let alive = true
    void probeCapabilities().then((r) => {
      if (alive) setCaps(resolveWidgetCapabilities(r, config.mode))
    })
    return () => {
      alive = false
    }
  }, [config.mode])

  const accentStyle = config.accent ? ({ '--primary': config.accent } as CSSProperties) : undefined

  return (
    <div
      style={accentStyle}
      className="flex h-full min-h-0 w-full flex-col overflow-hidden bg-background text-left text-foreground"
    >
      <Header title={config.title ?? 'Assistant'} embedded={embedded} onClose={onClose} />
      {caps === null ? (
        <Centered>
          <Loader2 className="size-5 animate-spin text-muted-foreground" />
        </Centered>
      ) : caps.effectiveMode === 'unsupported' ? (
        <Unsupported reason={caps.reason} />
      ) : caps.constrained && !tryAnyway ? (
        <ConstrainedNotice reason={caps.constrainedReason} onProceed={() => setTryAnyway(true)} />
      ) : (
        <ChatPanel
          config={config}
          persistKey={persistKey}
          loadOnMount={loadOnMount}
          onMessage={onMessage}
          voiceAvailable={config.mode !== 'text' && caps.voiceAvailable}
        />
      )}
    </div>
  )
}

function ChatPanel({
  config,
  persistKey,
  loadOnMount,
  onMessage,
  voiceAvailable,
}: Pick<Props, 'config' | 'persistKey' | 'loadOnMount' | 'onMessage'> & { voiceAvailable?: boolean }) {
  const c = useTextController(config, { persistKey, loadOnMount, onMessage })
  const canText = config.mode !== 'voice'
  const canVoice = !!voiceAvailable
  // text+voice widgets open on a quick Type/Talk picker; single-mode widgets skip it
  // (text → straight to chat; voice → a one-tap start so the mic gesture + download are
  // intentional). `picked` flips once the user has chosen how to start.
  const [picked, setPicked] = useState(canText && !canVoice)
  const view = !picked ? 'picker' : c.voiceActive ? 'voice' : 'text'
  const greeting = config.greeting?.trim()

  return (
    <>
      <div className="flex items-center justify-end gap-1 px-3 pt-2">
        <ThemeToggle />
        <SettingsMenu
          onClearChat={c.clear}
          hasChat={c.turns.length > 0}
          canRemove={c.cached || c.status === 'ready' || c.status === 'thinking'}
          onForgetModel={c.forgetModel}
        />
      </div>

      {view === 'picker' && (
        <ModePicker
          greeting={greeting}
          canText={canText}
          canVoice={canVoice}
          onType={() => setPicked(true)}
          onTalk={() => {
            setPicked(true)
            c.toggleVoice()
          }}
        />
      )}

      {view === 'text' && (
        <TextView controller={c} greeting={greeting} canVoice={canVoice} onTalk={c.toggleVoice} />
      )}

      {view === 'voice' && <VoiceView controller={c} canText={canText} onType={c.toggleVoice} />}
    </>
  )
}

// ── Mode picker ───────────────────────────────────────────────────────────────
function ModePicker({
  greeting,
  canText,
  canVoice,
  onType,
  onTalk,
}: {
  greeting?: string
  canText: boolean
  canVoice: boolean
  onType: () => void
  onTalk: () => void
}) {
  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-5 px-6 text-center">
      <AidekinMark className="size-10 text-foreground" coreClassName="fill-primary" />
      <p className="max-w-[240px] text-sm text-muted-foreground">
        {greeting || 'How would you like to chat?'}
      </p>
      <div className="flex w-full max-w-[240px] flex-col gap-2.5">
        {canText && (
          <button
            type="button"
            onClick={onType}
            className="inline-flex flex-col items-center justify-center gap-0.5 rounded-xl border border-input px-4 py-3 font-semibold transition-colors hover:bg-secondary"
          >
            <span className="inline-flex items-center gap-2 text-sm">
              <Keyboard className="size-4" /> Type
            </span>
            <span className="text-[10px] font-normal text-muted-foreground">no microphone needed</span>
          </button>
        )}
        {canVoice && (
          <button
            type="button"
            onClick={onTalk}
            className="inline-flex flex-col items-center justify-center gap-0.5 rounded-xl bg-primary px-4 py-3 font-semibold text-primary-foreground transition-opacity hover:opacity-90"
          >
            <span className="inline-flex items-center gap-2 text-sm">
              <Mic className="size-4" /> Talk
            </span>
            <span className="text-[10px] font-normal opacity-80">uses your mic · one-time setup, then instant</span>
          </button>
        )}
      </div>
    </div>
  )
}

// ── Text view ─────────────────────────────────────────────────────────────────
function TextView({
  controller,
  greeting,
  canVoice,
  onTalk,
}: {
  controller: ReturnType<typeof useTextController>
  greeting?: string
  canVoice: boolean
  onTalk: () => void
}) {
  const { turns, status, loadPct, loadDetail, cached, error, send, stop, retry, trimmed } = controller
  const [draft, setDraft] = useState('')
  const scrollRef = useRef<HTMLDivElement>(null)
  const loading = status === 'loading'
  const thinking = status === 'thinking'

  useEffect(() => {
    const el = scrollRef.current
    if (!el) return
    if (el.scrollHeight - el.scrollTop - el.clientHeight < 120) el.scrollTop = el.scrollHeight
  }, [turns, status])

  const submit = () => {
    const text = draft.trim()
    if (!text || thinking || loading) return
    send(text)
    setDraft('')
  }
  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  return (
    <>
      <div ref={scrollRef} className="convo-scroll flex flex-1 flex-col gap-2.5 overflow-y-auto px-3.5 py-3">
        {turns.length === 0 && <EmptyState greeting={greeting} />}
        {trimmed && turns.length > 0 && (
          <p className="mono-kicker self-center py-1 text-center text-[10px] normal-case tracking-normal">
            earlier messages trimmed to keep replies fast
          </p>
        )}
        {turns.map((t) => (
          <Bubble key={t.id} role={t.role} text={t.text} />
        ))}
        {thinking && turns[turns.length - 1]?.role !== 'assistant' && <TypingDots />}
      </div>

      {loading && <LoadBar pct={loadPct} detail={loadDetail} cached={cached} />}
      {status === 'error' && error && <ErrorBar error={error} onRetry={retry} />}

      <form
        onSubmit={(e) => {
          e.preventDefault()
          submit()
        }}
        className="flex items-end gap-2 border-t border-border bg-card/60 p-2.5"
      >
        <textarea
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={onKey}
          rows={1}
          maxLength={4000}
          disabled={loading}
          placeholder={loading ? 'Setting up the assistant…' : 'Type a message…'}
          className="max-h-28 min-h-10 flex-1 resize-none rounded-xl border border-input bg-background px-3.5 py-2.5 text-sm outline-none placeholder:text-muted-foreground focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-60"
        />
        {canVoice && (
          <button
            type="button"
            onClick={onTalk}
            aria-label="Switch to voice"
            title="Switch to voice"
            className="grid size-10 shrink-0 place-items-center rounded-xl border border-input text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
          >
            <Mic className="size-4" />
          </button>
        )}
        {thinking ? (
          <button
            type="button"
            onClick={stop}
            aria-label="Stop generating"
            className="grid size-10 shrink-0 place-items-center rounded-xl bg-secondary text-foreground transition-colors hover:bg-secondary/70"
          >
            <Square className="size-3.5 fill-current" />
          </button>
        ) : (
          <button
            type="submit"
            disabled={!draft.trim() || loading}
            aria-label="Send message"
            className="grid size-10 shrink-0 place-items-center rounded-xl bg-primary text-primary-foreground transition-opacity hover:opacity-90 disabled:opacity-40"
          >
            <Send className="size-4" />
          </button>
        )}
      </form>
      <p className="px-3 pb-2 text-center text-[10px] leading-tight text-muted-foreground">
        AI can make mistakes. Double-check important info.
      </p>
    </>
  )
}

// ── Voice view (immersive: sonar ping + live transcript) ────────────────────────
function VoiceView({
  controller,
  canText,
  onType,
}: {
  controller: ReturnType<typeof useTextController>
  canText: boolean
  onType: () => void
}) {
  const { turns, voiceState, voiceLoadPct, levelRef, error, retry } = controller
  const scrollRef = useRef<HTMLDivElement>(null)
  const loadingVoice = voiceState === 'loading' || voiceState === 'cold'

  useEffect(() => {
    const el = scrollRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [turns, voiceState])

  const status = loadingVoice
    ? { text: 'Loading voice', busy: true }
    : voiceState === 'thinking'
      ? { text: 'Thinking', busy: true }
      : voiceState === 'speaking'
        ? { text: 'Speaking', busy: true }
        : { text: 'Listening, just speak', busy: false }

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex flex-col items-center gap-2.5 px-4 pt-5 pb-3">
        <SonarPing state={loadingVoice ? 'thinking' : voiceState} levelRef={levelRef} className="size-24" />
        <StatusLabel text={status.text} busy={status.busy} />
        {loadingVoice && (
          <div className="h-1.5 w-40 overflow-hidden rounded-full bg-secondary">
            <div
              className="h-full rounded-full bg-primary transition-[width] duration-200"
              style={{ width: `${Math.max(4, voiceLoadPct * 100)}%` }}
            />
          </div>
        )}
      </div>

      <div
        ref={scrollRef}
        className="convo-scroll flex min-h-0 flex-1 flex-col gap-2.5 overflow-y-auto px-3.5 pb-2"
      >
        {turns.map((t) => (
          <Bubble key={t.id} role={t.role} text={t.text} />
        ))}
      </div>

      {error && <ErrorBar error={error} onRetry={retry} />}

      {canText && (
        <div className="flex justify-center border-t border-border bg-card/60 p-2.5">
          <button
            type="button"
            onClick={onType}
            className="inline-flex items-center gap-1.5 rounded-xl border border-input px-3.5 py-2 text-xs font-medium text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
          >
            <Keyboard className="size-3.5" /> Type instead
          </button>
        </div>
      )}
    </div>
  )
}

// ── Shared bits ─────────────────────────────────────────────────────────────────
function Header({ title, embedded, onClose }: { title: string; embedded?: boolean; onClose?: () => void }) {
  return (
    <header className="flex items-center gap-2.5 border-b border-border bg-card/70 px-3.5 py-2.5">
      <AidekinMark className="size-5 text-foreground" coreClassName="fill-primary" />
      <span className="flex-1 truncate font-display text-sm font-semibold">{title}</span>
      {embedded && (
        <button
          type="button"
          onClick={onClose}
          aria-label="Close chat"
          className="grid size-7 place-items-center rounded-lg text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
        >
          <X className="size-4" />
        </button>
      )}
    </header>
  )
}

function EmptyState({ greeting }: { greeting?: string }) {
  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-3 px-6 text-center">
      <AidekinMark className="size-9 text-foreground" coreClassName="fill-primary" />
      <p className="text-sm text-muted-foreground">
        {greeting || 'Hi! Ask me anything. I run entirely in your browser.'}
      </p>
    </div>
  )
}

function Bubble({ role, text }: { role: WidgetTurn['role']; text: string }) {
  const isAssistant = role === 'assistant'
  return (
    <div
      className={cn(
        'max-w-[85%] rounded-xl px-3.5 py-2.5 text-left text-sm leading-relaxed animate-in fade-in slide-in-from-bottom-1',
        !isAssistant && 'whitespace-pre-wrap',
        role === 'user' && 'self-end rounded-br-sm bg-secondary',
        isAssistant && 'self-start rounded-bl-sm border border-primary/25 bg-primary/[0.07]',
        role === 'error' &&
          'self-start rounded-bl-sm border border-destructive/40 bg-destructive/10 text-destructive',
      )}
    >
      {isAssistant ? (text ? <Markdown text={text} /> : '…') : text || '…'}
    </div>
  )
}

function TypingDots() {
  return (
    <div className="flex items-center gap-1 self-start rounded-xl rounded-bl-sm border border-primary/25 bg-primary/[0.07] px-3.5 py-3">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="size-1.5 animate-bounce rounded-full bg-primary/70"
          style={{ animationDelay: `${i * 0.15}s` }}
        />
      ))}
    </div>
  )
}

// Inline bouncing dots, inheriting the current text color (bg-current).
function Dots({ className }: { className?: string }) {
  return (
    <span className={cn('inline-flex items-center gap-[3px]', className)}>
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="size-1 animate-bounce rounded-full bg-current"
          style={{ animationDelay: `${i * 0.15}s` }}
        />
      ))}
    </span>
  )
}

// Animated status line (Loading / Thinking / Speaking): a gentle pulse + trailing
// bouncing dots so transitional states feel alive instead of being flat text.
function StatusLabel({ text, busy }: { text: string; busy: boolean }) {
  return (
    <p className={cn('flex items-center gap-1.5 text-sm font-medium text-foreground', busy && 'animate-pulse')}>
      {text}
      {busy && <Dots className="text-primary" />}
    </p>
  )
}

function formatEta(s: number): string {
  if (!isFinite(s) || s <= 0) return ''
  if (s < 60) return `${Math.max(1, Math.round(s))}s`
  return `${Math.round(s / 60)} min`
}

function LoadBar({ pct, detail, cached }: { pct: number; detail: string; cached: boolean }) {
  const [eta, setEta] = useState<string | null>(null)
  // Anchor at the first real progress sample and extrapolate a smoothed remaining time
  // from the average rate so far (stable, "somewhat correct" — network speed varies).
  const anchor = useRef<{ t: number; pct: number } | null>(null)
  const smooth = useRef<number | null>(null)

  useEffect(() => {
    const now = performance.now()
    if (!anchor.current || pct < anchor.current.pct) {
      anchor.current = { t: now, pct }
      return
    }
    const dt = (now - anchor.current.t) / 1000
    const dp = pct - anchor.current.pct
    if (pct >= 0.99) {
      setEta(null)
      return
    }
    if (dt < 1 || dp < 0.01) return // too little signal yet to estimate
    const remain = ((1 - pct) * dt) / dp
    smooth.current = smooth.current == null ? remain : smooth.current * 0.6 + remain * 0.4
    setEta(formatEta(smooth.current) || null)
  }, [pct])

  const label = cached
    ? 'Loading the assistant'
    : eta
      ? `Downloading the assistant · about ${eta} left`
      : 'Downloading the assistant (one-time)'

  return (
    <div className="border-t border-border bg-card/60 px-3.5 py-2">
      <div className="mb-1 flex items-center justify-between text-[11px] text-muted-foreground">
        <span className="flex min-w-0 items-center gap-1.5">
          <span className="truncate">{label}</span>
          <Dots className="shrink-0 text-muted-foreground" />
        </span>
        <span className="shrink-0 pl-2 tabular-nums">{Math.round(pct * 100)}%</span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-secondary">
        <div
          className="h-full rounded-full bg-primary transition-[width] duration-200"
          style={{ width: `${Math.max(4, pct * 100)}%` }}
        />
      </div>
      {detail && <p className="mt-1 truncate text-[10px] text-muted-foreground">{detail}</p>}
    </div>
  )
}

function ErrorBar({ error, onRetry }: { error: string; onRetry: () => void }) {
  return (
    <div className="flex items-center justify-between gap-2 border-t border-destructive/30 bg-destructive/10 px-3.5 py-2 text-xs text-destructive">
      <span className="truncate">{error}</span>
      <button
        type="button"
        onClick={onRetry}
        className="shrink-0 rounded-lg bg-destructive/15 px-2.5 py-1 font-medium transition-colors hover:bg-destructive/25"
      >
        Retry
      </button>
    </div>
  )
}

function Unsupported({ reason }: { reason?: string }) {
  return (
    <Centered>
      <div className="max-w-xs px-6 text-center">
        <p className="mb-1.5 text-sm font-medium">This device can't run the assistant yet</p>
        <p className="text-xs text-muted-foreground">
          {reason ?? 'WebGPU is required.'} Try the latest Chrome or Edge on desktop, or Safari 26+.
        </p>
      </div>
    </Centered>
  )
}

// WebGPU is present but the device looks too small for a ~1.7B model (phone/tablet, low memory).
// We warn before the multi-hundred-MB download instead of letting it OOM or reload, but still
// let the visitor proceed, since the check is a heuristic.
function ConstrainedNotice({ reason, onProceed }: { reason?: string; onProceed: () => void }) {
  return (
    <Centered>
      <div className="max-w-xs px-6 text-center">
        <p className="mb-1.5 text-sm font-medium">This device may not have enough memory</p>
        <p className="text-xs text-muted-foreground">
          {reason ?? 'aidekin runs the model on your device and works best on desktop.'} For the best
          experience, open it in Chrome or Edge on a desktop.
        </p>
        <button
          type="button"
          onClick={onProceed}
          className="mt-4 rounded-md border border-border px-4 py-2 text-xs font-medium text-foreground transition-colors hover:bg-secondary"
        >
          Try anyway
        </button>
      </div>
    </Centered>
  )
}

// Per-visitor light/dark toggle. Defaults to the owner's data-theme (auto = OS), but a
// visitor's choice here wins and is remembered (see widget/main.tsx pre-paint).
function ThemeToggle() {
  const [dark, setDark] = useState(
    () => typeof document !== 'undefined' && document.documentElement.classList.contains('dark'),
  )
  const toggle = () => {
    const next = !dark
    setDark(next)
    document.documentElement.classList.toggle('dark', next)
    try {
      localStorage.setItem('aidekin:widget-theme', next ? 'dark' : 'light')
    } catch {
      /* storage unavailable */
    }
  }
  return (
    <button
      type="button"
      onClick={toggle}
      aria-label="Toggle light or dark theme"
      className="grid size-7 place-items-center rounded-lg text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
    >
      {dark ? <Sun className="size-4" /> : <Moon className="size-4" />}
    </button>
  )
}

function SettingsMenu({
  onClearChat,
  hasChat,
  canRemove,
  onForgetModel,
}: {
  onClearChat: () => void
  hasChat: boolean
  canRemove: boolean
  onForgetModel: () => Promise<{ ok: boolean }>
}) {
  const [open, setOpen] = useState(false)
  const [storage, setStorage] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [note, setNote] = useState<string | null>(null)

  const refreshStorage = () => {
    void import('@/core/storage')
      .then(({ estimateStorage }) => estimateStorage())
      .then((s) => {
        if (s) setStorage(`${Math.round(s.usageBytes / 1048576)} MB used on this site`)
      })
      .catch(() => undefined)
  }

  useEffect(() => {
    if (open) refreshStorage()
  }, [open])

  const removeModel = () => {
    setBusy(true)
    setNote(null)
    void onForgetModel()
      .then((r) => {
        setNote(r.ok ? 'Removed. It will re-download next time.' : 'Partly removed; some files were still in use.')
        refreshStorage()
      })
      .catch(() => setNote('Could not remove model data.'))
      .finally(() => setBusy(false))
  }

  const itemCls =
    'flex w-full items-center gap-2 rounded-lg px-2 py-2 text-left text-sm transition-colors hover:bg-secondary disabled:opacity-40'

  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-label="Settings"
        className="grid size-7 place-items-center rounded-lg text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
      >
        <Settings className="size-4" />
      </button>
      {open && (
        <>
          <div className="fixed inset-0 z-10" onClick={() => setOpen(false)} />
          <div className="absolute right-0 z-20 mt-1 w-60 rounded-xl border border-border bg-popover p-2 shadow-xl">
            <p className="flex items-center gap-1.5 px-2 py-1.5 text-xs text-muted-foreground">
              <HardDrive className="size-3.5" /> {storage ?? 'Checking storage…'}
            </p>
            <button
              type="button"
              onClick={() => {
                onClearChat()
                setOpen(false)
              }}
              disabled={!hasChat}
              className={itemCls}
            >
              <Trash2 className="size-4" /> Clear chat
            </button>
            <button type="button" onClick={removeModel} disabled={busy || !canRemove} className={itemCls}>
              <HardDrive className="size-4" /> {busy ? 'Removing…' : 'Remove downloaded model'}
            </button>
            {note && <p className="px-2 pt-1 text-[11px] text-muted-foreground">{note}</p>}
            <a
              href="https://aidekin.com"
              target="_blank"
              rel="noreferrer"
              className="mt-1.5 block border-t border-border px-2 pt-2 text-center text-[11px] text-muted-foreground transition-colors hover:text-foreground"
            >
              Powered by aidekin
            </a>
          </div>
        </>
      )}
    </div>
  )
}

function Centered({ children }: { children: ReactNode }) {
  return <div className="grid flex-1 place-items-center">{children}</div>
}
