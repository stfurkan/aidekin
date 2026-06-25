// The sonar-ping voice visualization - Aidekin's signature motif. A source core with
// waves emanating outward; the core scales with the live mic level. Sized by the parent
// (pass a size via className, e.g. "size-28").

import { useEffect, useRef, type RefObject } from 'react'
import { cn } from '@/lib/utils'
import type { AgentState } from '@/pipeline/orchestrator'

const STATE_CLASS: Partial<Record<AgentState, string>> = {
  listening: 'sonar-listening',
  thinking: 'sonar-thinking',
  speaking: 'sonar-speaking',
}

export function SonarPing({
  state,
  levelRef,
  className,
}: {
  state: AgentState
  levelRef?: RefObject<number>
  className?: string
}) {
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!levelRef) return
    let raf = 0
    const tick = (): void => {
      const el = ref.current
      if (el) el.style.setProperty('--level', String(Math.min(1, levelRef.current * 3.2)))
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [levelRef])

  return (
    <div ref={ref} className={cn('sonar', STATE_CLASS[state] ?? 'sonar-idle', className)}>
      <span className="sonar-wave" />
      <span className="sonar-wave" />
      <span className="sonar-wave" />
      <span className="sonar-core" />
    </div>
  )
}
