import { useEffect, useRef, useState } from 'react'
import { Mic, Send, Keyboard } from 'lucide-react'
import { WidgetFrame } from '@/widget/WidgetFrame'
import { SonarPing } from '@/widget/SonarPing'
import type { AgentState } from '@/pipeline/orchestrator'
import { AidekinMark } from './icons'

// A scripted, model-free preview of the real widget for the landing hero. It uses the
// actual widget chrome (Bracket Core header, the real bubble + voice styling) so it reads
// as the genuine product, not a stock mock. It alternates a TEXT cycle (types a Q&A) and
// a VOICE cycle (the immersive ping + live caption) so the hero shows both modes and
// their distinct designs. No model downloads here; the real thing is behind /demo.

interface Line {
  role: 'assistant' | 'user'
  text: string
}
const TEXT_SCRIPT: Line[] = [
  { role: 'assistant', text: 'Hi! Ask me anything. I run entirely in your browser.' },
  { role: 'user', text: 'What are your support hours?' },
  { role: 'assistant', text: 'We are around 9 to 5 Pacific, Monday to Friday. Want me to leave a note for the team?' },
]
const VOICE_USER = 'What is your return policy?'
const VOICE_REPLY = 'You can return anything within 30 days, no receipt needed.'

export function HeroChat() {
  const [phase, setPhase] = useState<'text' | 'voice'>('text')
  const [shown, setShown] = useState<Line[]>([])
  const [dots, setDots] = useState(false)
  const [vState, setVState] = useState<AgentState>('listening')
  const [vLabel, setVLabel] = useState('Listening, just speak')
  const levelRef = useRef(0)
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) {
      setShown(TEXT_SCRIPT.map((l) => ({ ...l })))
      return
    }
    let cancelled = false
    const timers: ReturnType<typeof setTimeout>[] = []
    const wait = (ms: number) => new Promise<void>((r) => timers.push(setTimeout(r, ms)))

    const typeInto = async (line: Line, speed: number) => {
      setShown((p) => [...p, { role: line.role, text: '' }])
      for (let i = 1; i <= line.text.length; i++) {
        if (cancelled) return
        const slice = line.text.slice(0, i)
        setShown((p) => p.map((m, idx) => (idx === p.length - 1 ? { ...m, text: slice } : m)))
        await wait(speed)
      }
    }

    const textCycle = async () => {
      setPhase('text')
      setShown([])
      await wait(600)
      for (const line of TEXT_SCRIPT) {
        if (cancelled) return
        if (line.role === 'assistant') {
          setDots(true)
          await wait(700)
          setDots(false)
        }
        await typeInto(line, line.role === 'user' ? 34 : 18)
        await wait(line.role === 'user' ? 450 : 1200)
      }
      await wait(1800)
    }

    const voiceCycle = async () => {
      setPhase('voice')
      setShown([])
      levelRef.current = 0
      setVState('listening')
      setVLabel('Listening, just speak')
      await wait(900)
      levelRef.current = 0.55
      await typeInto({ role: 'user', text: VOICE_USER }, 40)
      levelRef.current = 0
      await wait(400)
      setVState('thinking')
      setVLabel('Thinking…')
      await wait(900)
      setVState('speaking')
      setVLabel('Speaking…')
      levelRef.current = 0.6
      await typeInto({ role: 'assistant', text: VOICE_REPLY }, 26)
      levelRef.current = 0
      await wait(1900)
    }

    const run = async () => {
      for (;;) {
        await textCycle()
        if (cancelled) return
        await voiceCycle()
        if (cancelled) return
      }
    }
    void run()
    return () => {
      cancelled = true
      timers.forEach(clearTimeout)
    }
  }, [])

  useEffect(() => {
    const el = scrollRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [shown, dots, phase])

  const bubbles = shown.map((m, i) => (
    <div
      key={i}
      className={
        m.role === 'user'
          ? 'max-w-[85%] self-end rounded-xl rounded-br-sm bg-secondary px-3.5 py-2.5 text-sm leading-relaxed'
          : 'max-w-[88%] self-start rounded-xl rounded-bl-sm border border-primary/25 bg-primary/[0.07] px-3.5 py-2.5 text-sm leading-relaxed'
      }
    >
      {m.text || '…'}
    </div>
  ))

  return (
    <WidgetFrame className="h-[clamp(380px,54vh,480px)]">
      <header className="flex items-center gap-2.5 border-b border-border bg-card/70 px-3.5 py-2.5">
        <AidekinMark className="size-5 text-foreground" coreClassName="fill-primary" />
        <span className="flex-1 truncate font-display text-sm font-semibold">aidekin</span>
        <span className="mono-kicker text-[10px] normal-case tracking-normal">
          {phase === 'voice' ? 'voice' : 'text'}
        </span>
      </header>

      {phase === 'text' ? (
        <>
          <div ref={scrollRef} className="convo-scroll flex flex-1 flex-col gap-2.5 overflow-y-auto px-3.5 py-3">
            {bubbles}
            {dots && (
              <div className="flex items-center gap-1 self-start rounded-xl rounded-bl-sm border border-primary/25 bg-primary/[0.07] px-3.5 py-3">
                {[0, 1, 2].map((i) => (
                  <span
                    key={i}
                    className="size-1.5 animate-bounce rounded-full bg-primary/70"
                    style={{ animationDelay: `${i * 0.15}s` }}
                  />
                ))}
              </div>
            )}
          </div>
          <div className="flex items-center gap-2 border-t border-border bg-card/60 p-2.5">
            <div className="flex-1 rounded-xl border border-input bg-background px-3.5 py-2.5 text-sm text-muted-foreground">
              Ask anything…
            </div>
            <span className="grid size-10 shrink-0 place-items-center rounded-xl border border-input text-muted-foreground">
              <Mic className="size-4" />
            </span>
            <span className="grid size-10 shrink-0 place-items-center rounded-xl bg-primary text-primary-foreground">
              <Send className="size-4" />
            </span>
          </div>
        </>
      ) : (
        <>
          <div className="flex flex-col items-center gap-2 px-4 pt-5 pb-2">
            <SonarPing state={vState} levelRef={levelRef} className="size-20" />
            <p className="text-sm font-medium text-foreground">{vLabel}</p>
          </div>
          <div ref={scrollRef} className="convo-scroll flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto px-3.5 pb-2">
            {bubbles}
          </div>
          <div className="flex justify-center border-t border-border bg-card/60 p-2.5">
            <span className="inline-flex items-center gap-1.5 rounded-xl border border-input px-3.5 py-2 text-xs font-medium text-muted-foreground">
              <Keyboard className="size-3.5" /> Type instead
            </span>
          </div>
        </>
      )}
    </WidgetFrame>
  )
}
