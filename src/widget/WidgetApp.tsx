// The widget panel UI. Lives inside the iframe (embedded) OR inline on the site (demo /
// configurator preview). Probes capabilities, then shows either a Type/Talk picker
// (text+voice widgets), the text chat, or the immersive voice view. Falls back to a
// friendly notice when WebGPU is missing.

import {
  memo,
  useEffect,
  useRef,
  useState,
  type CSSProperties,
  type KeyboardEvent,
  type ReactNode,
} from 'react'
import { Send, Square, Trash2, X, Loader2, Settings, HardDrive, Mic, MicOff, Keyboard, Sun, Moon, MessageSquareWarning } from 'lucide-react'
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

/** Keep a conversation log pinned to its newest entry unless the user has scrolled up to read
 *  history (re-sticks when they return to the bottom). A distance-only check on append breaks
 *  as soon as one update grows the log past the threshold (a multi-line ASR bubble, a whole
 *  spoken clause), after which auto-scroll never recovers - the voice view hit this constantly. */
function useStickyAutoScroll(...deps: unknown[]) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const stick = useRef(true)
  const onScroll = () => {
    const el = scrollRef.current
    if (el) stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 120
  }
  useEffect(() => {
    const el = scrollRef.current
    if (el && stick.current) el.scrollTop = el.scrollHeight
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)
  // Virtual keyboard: when it opens, the host loader shrinks our iframe to the visible rect, and
  // WebKit may also force-scroll THIS document to reveal the focused composer. Undo that scroll
  // (the app is its own scroll container) and keep the log pinned to the newest message.
  useEffect(() => {
    const onResize = (): void => {
      window.scrollTo(0, 0)
      const el = scrollRef.current
      if (el && stick.current) el.scrollTop = el.scrollHeight
    }
    window.addEventListener('resize', onResize)
    window.visualViewport?.addEventListener('resize', onResize)
    return () => {
      window.removeEventListener('resize', onResize)
      window.visualViewport?.removeEventListener('resize', onResize)
    }
  }, [])
  return { scrollRef, onScroll }
}

interface Props {
  config: WidgetConfig
  /** Inside the iframe → show a close button that asks the host to hide us. */
  embedded?: boolean
  persistKey?: string
  /** Begin model download on mount. Default true; configurator preview passes false. */
  loadOnMount?: boolean
  onMessage?: (role: 'user' | 'assistant', text: string) => void
  onClose?: () => void
  /** Hide the visitor theme toggle (configurator preview: the theme is driven by the
   *  "Default theme" control there, and the toggle would flip the whole host site). */
  lockTheme?: boolean
}

export function WidgetApp({ config, embedded, persistKey, loadOnMount, onMessage, onClose, lockTheme }: Props) {
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
          lockTheme={lockTheme}
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
  lockTheme,
  voiceAvailable,
}: Pick<Props, 'config' | 'persistKey' | 'loadOnMount' | 'onMessage' | 'lockTheme'> & { voiceAvailable?: boolean }) {
  const c = useTextController(config, { persistKey, loadOnMount, onMessage })
  const canText = config.mode !== 'voice'
  const canVoice = !!voiceAvailable
  // text+voice widgets open on a quick Type/Talk picker; single-mode widgets skip it
  // (text → straight to chat; voice → a one-tap start so the mic gesture + download are
  // intentional). `picked` flips once the user has chosen how to start.
  const [picked, setPicked] = useState(canText && !canVoice)
  const view = !picked ? 'picker' : c.voiceActive ? 'voice' : 'text'
  const greeting = config.greeting?.trim()

  // Load the brain the moment the user is in the text conversation: text-only opens straight
  // here (so it loads on open, same as before), while text+voice arrives only after picking
  // Type. Voice loads via toggleVoice. Sitting on the picker triggers no download.
  useEffect(() => {
    if (loadOnMount !== false && view === 'text') c.preload()
    // eslint-disable-next-line react-hooks/exhaustive-deps -- `c` is a fresh object every render; c.preload is the stable (useCallback) dependency
  }, [view, loadOnMount, c.preload])

  return (
    <>
      <div className="flex items-center justify-end gap-1 px-3 pt-2">
        {!lockTheme && <ThemeToggle />}
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
      <p className="max-w-60 text-sm text-muted-foreground">
        {greeting || 'How would you like to chat?'}
      </p>
      <div className="flex w-full max-w-65 flex-col gap-2.5">
        {/* Text is the recommended default (lighter, works everywhere) → it's the primary CTA.
            Voice is a heavier opt-in (large speech download, beta) → secondary. */}
        {canText && (
          <button
            type="button"
            onClick={onType}
            aria-label="Type - recommended, no microphone needed"
            className="inline-flex flex-col items-center justify-center gap-0.5 rounded-xl bg-primary px-4 py-3 font-semibold text-primary-foreground transition-opacity hover:opacity-90"
          >
            <span className="inline-flex items-center gap-2 text-sm">
              <Keyboard className="size-4" /> Type
            </span>
            <span className="text-[10px] font-normal text-primary-foreground">recommended · no microphone needed</span>
          </button>
        )}
        {canVoice && (
          <button
            type="button"
            onClick={onTalk}
            aria-label="Talk (beta) - uses your microphone, about 1.6 GB downloaded on first use"
            className="inline-flex flex-col items-center justify-center gap-0.5 rounded-xl border border-input px-4 py-3 font-semibold transition-colors hover:bg-secondary"
          >
            <span className="inline-flex items-center gap-2 text-sm">
              <Mic className="size-4" /> Talk
              <span className="rounded bg-primary px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-primary-foreground">
                beta
              </span>
            </span>
            <span className="text-[10px] font-normal text-muted-foreground">uses your mic · ~1.6 GB on first use</span>
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
  const { turns, status, loadPct, cached, error, send, stop, retry, slowDevice } = controller
  const [draft, setDraft] = useState('')
  const { scrollRef, onScroll } = useStickyAutoScroll(turns, status)
  const loading = status === 'loading'
  const thinking = status === 'thinking'

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
      <div
        ref={scrollRef}
        onScroll={onScroll}
        role="log"
        aria-live="polite"
        aria-relevant="additions"
        aria-label="Conversation"
        className="convo-scroll flex flex-1 flex-col gap-2.5 overflow-y-auto px-3.5 py-3"
      >
        {turns.length === 0 && <EmptyState greeting={greeting} />}
        {slowDevice && turns.length > 0 && (
          <p className="mono-kicker self-center py-1 text-center text-[10px] normal-case tracking-normal">
            this device is on the slower side for on-device AI · replies may take a while
          </p>
        )}
        {turns.map((t) => (
          <Bubble key={t.id} role={t.role} text={t.text} />
        ))}
        {thinking && turns[turns.length - 1]?.role !== 'assistant' && <TypingDots />}
      </div>

      {loading && <LoadBar pct={loadPct} cached={cached} />}
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
          className="max-h-28 min-h-10 flex-1 resize-none rounded-xl border border-input bg-background px-3.5 py-2.5 text-sm outline-none placeholder:text-muted-foreground focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-60 pointer-coarse:text-base"
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
      <p className="px-3 pt-1.5 pb-2 text-center text-[10px] leading-tight text-muted-foreground">
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
  const { turns, voiceState, voiceLoadPct, voiceCached, levelRef, error, retry, muted, toggleMute } = controller
  const { scrollRef, onScroll } = useStickyAutoScroll(turns, voiceState)
  const loadingVoice = voiceState === 'loading' || voiceState === 'cold'

  const status = loadingVoice
    ? { text: 'Loading voice', busy: true }
    : voiceState === 'requesting-mic'
      ? { text: 'Allow microphone access to talk', busy: true }
      : voiceState === 'thinking'
        ? { text: 'Thinking', busy: true }
        : voiceState === 'speaking'
          ? { text: 'Speaking', busy: true }
          : muted
            ? { text: 'Muted', busy: false }
            : { text: 'Listening, just speak', busy: false }

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex flex-col items-center gap-2.5 px-4 pt-5 pb-3">
        <SonarPing state={loadingVoice ? 'thinking' : voiceState} levelRef={levelRef} className="size-24" />
        <StatusLabel text={status.text} busy={status.busy} />
        {loadingVoice && (
          <>
            <div className="h-1.5 w-40 overflow-hidden rounded-full bg-secondary">
              <div
                className="h-full rounded-full bg-primary transition-[width] duration-200"
                style={{ width: `${Math.max(4, voiceLoadPct * 100)}%` }}
              />
            </div>
            {/* Honest expectation-setting. When the speech weights are already cached, this is a
                fast read from disk (no download); otherwise it's the one-time ~1.6 GB fetch. The
                "Use text instead" button below actually cancels + cleans up (see toggleVoice). */}
            <p className="max-w-60 text-center text-[11px] leading-snug text-muted-foreground">
              {voiceCached === null
                ? 'Setting up voice...'
                : voiceCached
                  ? 'Loading the voice models from your device (already downloaded).'
                  : 'One-time ~1.6 GB download. This can take a few minutes, and you can switch to text anytime.'}
            </p>
          </>
        )}
      </div>

      <div
        ref={scrollRef}
        onScroll={onScroll}
        role="log"
        aria-live="polite"
        aria-relevant="additions"
        aria-label="Conversation"
        className="convo-scroll flex min-h-0 flex-1 flex-col gap-2.5 overflow-y-auto px-3.5 pb-2"
      >
        {turns.map((t) => (
          <Bubble key={t.id} role={t.role} text={t.text} />
        ))}
      </div>

      {error && <ErrorBar error={error} onRetry={retry} />}

      {(!loadingVoice || canText) && (
        <div className="flex items-center justify-center gap-2 border-t border-border bg-card/60 p-2.5">
          {!loadingVoice && (
            <button
              type="button"
              onClick={toggleMute}
              aria-pressed={muted}
              aria-label={muted ? 'Unmute microphone' : 'Mute microphone'}
              className="inline-flex items-center gap-1.5 rounded-xl border border-input px-3.5 py-2 text-xs font-medium text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
            >
              {muted ? <MicOff className="size-3.5" /> : <Mic className="size-3.5" />}
              {muted ? 'Unmute' : 'Mute'}
            </button>
          )}
          {canText && (
            <button
              type="button"
              onClick={onType}
              className="inline-flex items-center gap-1.5 rounded-xl border border-input px-3.5 py-2 text-xs font-medium text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
            >
              <Keyboard className="size-3.5" /> {loadingVoice ? 'Use text instead' : 'Type instead'}
            </button>
          )}
        </div>
      )}
      {!loadingVoice && (
        <p className="px-3 pt-1.5 pb-2 text-center text-[10px] leading-tight text-muted-foreground">
          AI can make mistakes. Double-check important info.
        </p>
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

const Bubble = memo(function Bubble({ role, text }: { role: WidgetTurn['role']; text: string }) {
  const isAssistant = role === 'assistant'
  return (
    <div
      className={cn(
        'max-w-[85%] rounded-xl px-3.5 py-2.5 text-left text-sm leading-relaxed animate-in fade-in slide-in-from-bottom-1',
        !isAssistant && 'whitespace-pre-wrap',
        role === 'user' && 'self-end rounded-br-sm bg-secondary',
        isAssistant && 'self-start rounded-bl-sm border border-primary/25 bg-primary/7',
        role === 'error' &&
          'self-start rounded-bl-sm border border-destructive/40 bg-destructive/10 text-destructive',
      )}
    >
      {isAssistant ? (text ? <Markdown text={text} /> : '…') : text || '…'}
    </div>
  )
})

function TypingDots() {
  return (
    <div className="flex items-center gap-1 self-start rounded-xl rounded-bl-sm border border-primary/25 bg-primary/7 px-3.5 py-3">
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
    <span className={cn('inline-flex items-center gap-0.75', className)}>
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
    <p
      role="status"
      aria-live="polite"
      className={cn('flex items-center gap-1.5 text-sm font-medium text-foreground', busy && 'animate-pulse')}
    >
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

function LoadBar({ pct, cached }: { pct: number; cached: boolean }) {
  const [eta, setEta] = useState<string | null>(null)
  // Anchor at the first real progress sample and extrapolate a smoothed remaining time
  // from the average rate so far (stable, "somewhat correct" - network speed varies).
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
      // eslint-disable-next-line react-hooks/set-state-in-effect -- progress-tick estimator: eta derives from ref history across ticks, not from render state
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
    </div>
  )
}

function ErrorBar({ error, onRetry }: { error: string; onRetry: () => void }) {
  return (
    <div
      role="alert"
      className="flex items-center justify-between gap-2 border-t border-destructive/30 bg-destructive/10 px-3.5 py-2 text-xs text-destructive"
    >
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
        <p className="mb-1.5 text-sm font-medium">Runs on this device</p>
        <p className="text-xs text-muted-foreground">
          {reason ?? 'aidekin runs the model on your device and works best on desktop or a recent phone.'} The
          fastest experience is on a computer.
        </p>
        <button
          type="button"
          onClick={onProceed}
          className="mt-4 rounded-md border border-border px-4 py-2 text-xs font-medium text-foreground transition-colors hover:bg-secondary"
        >
          Continue
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
    const theme = next ? 'dark' : 'light'
    try {
      localStorage.setItem('aidekin:widget-theme', theme)
    } catch {
      /* storage unavailable */
    }
    // Tell the host loader so its launcher + loading overlay match on the next open. Target the
    // embedding page's exact origin (from the referrer), not '*'. Skipped when run inline.
    try {
      const hostOrigin = (() => {
        try {
          return new URL(document.referrer).origin
        } catch {
          return ''
        }
      })()
      if (hostOrigin && window.parent && window.parent !== window) {
        window.parent.postMessage({ kind: 'aidekin:theme-changed', theme }, hostOrigin)
      }
    } catch {
      /* parent unavailable */
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

// Inline two-step confirm for a destructive menu action: the action's button is swapped for
// this row (label + Cancel + a colored confirm), so nothing fires on the first click.
function ConfirmRow({
  label,
  description,
  confirmLabel,
  onConfirm,
  onCancel,
  destructive,
}: {
  label: string
  description?: string
  confirmLabel: string
  onConfirm: () => void
  onCancel: () => void
  destructive?: boolean
}) {
  const confirmCls = destructive
    ? 'bg-destructive/15 text-destructive hover:bg-destructive/25'
    : 'bg-primary/15 text-primary hover:bg-primary/25'
  // Stacked, not a single cramped row: the question wraps in full and a one-line consequence sits
  // under it, so the user can actually read what they're confirming inside the narrow menu.
  return (
    <div className="rounded-lg bg-secondary/50 p-2.5">
      <p className="text-sm font-medium leading-snug">{label}</p>
      {description && <p className="mt-1 text-xs leading-snug text-muted-foreground">{description}</p>}
      <div className="mt-2.5 flex justify-end gap-1.5">
        <button
          type="button"
          onClick={onCancel}
          className="rounded-md px-2.5 py-1 text-xs font-medium text-muted-foreground transition-colors hover:bg-secondary"
        >
          Cancel
        </button>
        <button
          type="button"
          onClick={onConfirm}
          className={`rounded-md px-2.5 py-1 text-xs font-medium transition-colors ${confirmCls}`}
        >
          {confirmLabel}
        </button>
      </div>
    </div>
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
  // Which destructive action is awaiting a confirm tap (null = none). Both Clear chat and
  // Remove model ask first instead of firing on the initial click.
  const [confirming, setConfirming] = useState<'clear' | 'remove' | null>(null)

  const refreshStorage = () => {
    void import('@/core/storage')
      .then(({ estimateStorage }) => estimateStorage())
      .then((s) => {
        if (s) setStorage(`${Math.round(s.usageBytes / 1048576)} MB used on this site`)
      })
      .catch(() => undefined)
  }

  // Every close path funnels through here so a pending confirmation never survives a
  // close/reopen (event-driven; no state-sync effect needed).
  const close = () => {
    setOpen(false)
    setConfirming(null)
  }

  useEffect(() => {
    if (!open) return
    refreshStorage()
    const onKey = (e: globalThis.KeyboardEvent) => {
      if (e.key === 'Escape') close()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
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
        aria-haspopup="menu"
        aria-expanded={open}
        className="grid size-7 place-items-center rounded-lg text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
      >
        <Settings className="size-4" />
      </button>
      {open && (
        <>
          <div className="fixed inset-0 z-10" onClick={close} />
          <div className="absolute right-0 z-20 mt-1 w-60 rounded-xl border border-border bg-popover p-2 shadow-xl">
            <p className="flex items-center gap-1.5 px-2 py-1.5 text-xs text-muted-foreground">
              <HardDrive className="size-3.5" /> {storage ?? 'Checking storage…'}
            </p>
            {confirming === 'clear' ? (
              <ConfirmRow
                label="Clear this conversation?"
                description="Removes the chat history saved on this device. This can’t be undone."
                confirmLabel="Clear"
                destructive
                onCancel={() => setConfirming(null)}
                onConfirm={() => {
                  onClearChat()
                  close()
                }}
              />
            ) : (
              <button type="button" onClick={() => setConfirming('clear')} disabled={!hasChat} className={itemCls}>
                <Trash2 className="size-4" /> Clear chat
              </button>
            )}
            {confirming === 'remove' ? (
              <ConfirmRow
                label="Remove the downloaded model?"
                description="Frees the cached model storage. It re-downloads the next time you open the assistant."
                confirmLabel="Remove"
                destructive
                onCancel={() => setConfirming(null)}
                onConfirm={() => {
                  setConfirming(null)
                  removeModel()
                }}
              />
            ) : (
              <button type="button" onClick={() => setConfirming('remove')} disabled={busy || !canRemove} className={itemCls}>
                <HardDrive className="size-4" /> {busy ? 'Removing…' : 'Remove downloaded model'}
              </button>
            )}
            {note && <p className="px-2 pt-1 text-[11px] text-muted-foreground">{note}</p>}
            {/* Zero telemetry means user reports are the ONLY error signal - keep this easy to find. */}
            <a href="https://github.com/stfurkan/aidekin/issues" target="_blank" rel="noreferrer" className={itemCls}>
              <MessageSquareWarning className="size-4" /> Report a problem
            </a>
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
