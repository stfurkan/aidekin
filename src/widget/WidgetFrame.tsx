import { type ReactNode } from 'react'
import { cn } from '@/lib/utils'

// The shared widget bezel - used everywhere the assistant is shown inline (the demo,
// the configurator preview, the landing hero). One frame so the chat looks identical
// across the site, and it mirrors the floating panel the embed loader draws on a host
// page (14px radius, hairline border, soft shadow, ~384px wide).
export function WidgetFrame({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div
      className={cn(
        'mx-auto flex h-[min(70vh,560px)] w-[min(384px,100%)] flex-col overflow-hidden rounded-[14px] border border-border bg-card shadow-2xl',
        className,
      )}
    >
      {children}
    </div>
  )
}
