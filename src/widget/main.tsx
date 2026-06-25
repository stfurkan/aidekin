// Widget iframe entry. Reads its config from the URL hash (the loader puts it there
// for instant, race-free render), pre-paints the theme, posts `ready` to the host,
// and bridges widget→host events (message/close) + host→widget control (theme).

import { StrictMode, useEffect } from 'react'
import { createRoot } from 'react-dom/client'
import { installErrorCapture } from '@/core/diagnostics'
import { WidgetApp } from './WidgetApp'
import {
  isAidekinMessage,
  parseConfigFromHash,
  withDefaults,
  type HostMessage,
  type WidgetMessage,
} from './protocol'
import '@/index.css'

installErrorCapture()

const config = withDefaults(parseConfigFromHash(location.hash))

// The embedding page's origin - used to target postMessage and scope persistence.
const hostOrigin = (() => {
  try {
    return new URL(document.referrer).origin
  } catch {
    return ''
  }
})()
const allowed = new Set([hostOrigin, ...(config.allowedOrigins ?? [])].filter(Boolean))
const persistKey = config.persist ? `aidekin-chat:${hostOrigin || 'local'}` : undefined

// Pre-paint theme. The visitor's own in-widget choice (if they've toggled it) wins;
// otherwise follow the owner's data-theme (default 'auto' = the visitor's OS scheme).
const storedTheme = (() => {
  try {
    return localStorage.getItem('aidekin:widget-theme')
  } catch {
    return null
  }
})()
const dark = storedTheme
  ? storedTheme === 'dark'
  : config.theme === 'dark' ||
    (config.theme === 'auto' && matchMedia('(prefers-color-scheme: dark)').matches)
if (dark) document.documentElement.classList.add('dark')

// Post to the embedding page ONLY (its exact origin). If the referrer is stripped so the
// origin is unknown, drop the message rather than broadcast widget/chat data with '*'.
const post = (m: WidgetMessage): void => {
  if (hostOrigin) window.parent?.postMessage(m, hostOrigin)
}

function Bridge() {
  useEffect(() => {
    const onMsg = (e: MessageEvent) => {
      // Only the embedding page (or explicitly allowed origins) may drive the widget. Fail
      // CLOSED: if the allow-set is empty (unknown origin), reject every message.
      if (!allowed.has(e.origin)) return
      if (!isAidekinMessage(e.data)) return
      const m = e.data as HostMessage
      if (m.kind === 'aidekin:theme') {
        document.documentElement.classList.toggle('dark', m.theme === 'dark')
      }
    }
    window.addEventListener('message', onMsg)
    post({ kind: 'aidekin:ready' })
    return () => window.removeEventListener('message', onMsg)
  }, [])

  return (
    <WidgetApp
      config={config}
      embedded
      persistKey={persistKey}
      onMessage={(role, text) => post({ kind: 'aidekin:message', role, text })}
      onClose={() => post({ kind: 'aidekin:close-request' })}
    />
  )
}

const root = document.getElementById('root')
if (!root) throw new Error('Aidekin widget: #root mount point missing')

createRoot(root).render(
  <StrictMode>
    <Bridge />
  </StrictMode>,
)
