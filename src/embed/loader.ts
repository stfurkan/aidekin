// Aidekin embed loader — the ~2 KB script a site owner drops in. On page load it does
// ONE thing: draw a floating launcher (inside a Shadow DOM, so host page CSS can't
// touch it). The widget iframe — and therefore all widget JS + the model download —
// is created only on the FIRST open. Nothing heavy touches the host page's load.

import type { WidgetConfig } from '@/widget/protocol'

type Cfg = Partial<WidgetConfig>

interface AidekinApi {
  open(): void
  close(): void
  toggle(): void
  /** Force the widget's theme (e.g. to sync with the host site's own light/dark toggle). */
  setTheme(theme: 'light' | 'dark'): void
  on(event: 'open' | 'close' | 'ready' | 'message', cb: (detail?: unknown) => void): void
}

declare global {
  interface Window {
    AidekinConfig?: Cfg
    Aidekin?: AidekinApi
  }
}

const DEFAULT_ACCENT = '#29a383'
const Z = 2147483000

function readDataConfig(el: HTMLScriptElement): Cfg {
  const d = el.dataset
  const cfg: Cfg = {}
  if (d.mode) cfg.mode = d.mode as Cfg['mode']
  if (d.systemPrompt) cfg.systemPrompt = d.systemPrompt
  if (d.greeting) cfg.greeting = d.greeting
  if (d.knowledgeUrl) cfg.knowledgeUrl = d.knowledgeUrl
  if (d.ragTopK) cfg.ragTopK = Number(d.ragTopK)
  if (d.reasoning) cfg.reasoning = d.reasoning !== 'false'
  if (d.theme) cfg.theme = d.theme as Cfg['theme']
  if (d.accent) cfg.accent = d.accent
  if (d.position) cfg.position = d.position as Cfg['position']
  if (d.launcherLabel) cfg.launcherLabel = d.launcherLabel
  if (d.title) cfg.title = d.title
  if (d.persist) cfg.persist = d.persist !== 'false'
  if (d.allowedOrigins) cfg.allowedOrigins = d.allowedOrigins.split(',').map((s) => s.trim()).filter(Boolean)
  if (d.widgetOrigin) cfg.widgetOrigin = d.widgetOrigin
  return cfg
}

function currentScript(): HTMLScriptElement {
  const el = document.currentScript as HTMLScriptElement | null
  if (el?.tagName === 'SCRIPT') return el
  const all = document.getElementsByTagName('script')
  for (let i = all.length - 1; i >= 0; i--) {
    if (/loader(\.min)?\.js/.test(all[i].src) || all[i].dataset.aidekin != null) return all[i]
  }
  return all[all.length - 1]
}

function init(): void {
  const script = currentScript()
  const config: Cfg = { ...(window.AidekinConfig ?? {}), ...readDataConfig(script) }
  const accent = config.accent || DEFAULT_ACCENT
  const left = config.position === 'bottom-left'
  const side = left ? 'left' : 'right'
  const wantsMic = config.mode === 'voice' || config.mode === 'both'

  const widgetOrigin =
    config.widgetOrigin?.replace(/\/$/, '') ||
    (() => {
      try {
        return new URL(script.src).origin
      } catch {
        return location.origin
      }
    })()
  const iframeSrc = `${widgetOrigin}/widget/#${encodeURIComponent(JSON.stringify(config))}`

  // ── isolated UI root (Shadow DOM) ─────────────────────────────────────────
  const host = document.createElement('div')
  host.id = 'aidekin-widget-root'
  const shadow = host.attachShadow({ mode: 'open' })
  shadow.innerHTML = `
    <style>
      :host { all: initial; }
      .launcher {
        position: fixed; ${side}: 20px; bottom: 20px; z-index: ${Z};
        display: inline-flex; align-items: center; gap: 8px;
        height: 52px; padding: 0 20px; border: 0; border-radius: 16px;
        background: ${accent}; color: #fff; cursor: pointer;
        font: 600 14px/1 ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
        box-shadow: 0 8px 28px -6px rgba(0,0,0,.35), 0 0 0 1px rgba(255,255,255,.08) inset;
        transition: transform .18s ease, box-shadow .18s ease, opacity .18s ease;
      }
      .launcher:hover { transform: translateY(-1px); box-shadow: 0 12px 34px -6px rgba(0,0,0,.45); }
      .launcher:active { transform: translateY(0); }
      .launcher svg { width: 20px; height: 20px; }
      .panel {
        position: fixed; ${side}: 20px; bottom: 20px; z-index: ${Z};
        width: min(384px, calc(100vw - 40px));
        height: min(620px, calc(100vh - 40px));
        border: 0; border-radius: 14px; overflow: hidden; background: transparent;
        box-shadow: 0 24px 60px -12px rgba(0,0,0,.45), 0 0 0 1px rgba(0,0,0,.06);
        opacity: 0; transform: translateY(12px) scale(.98); transform-origin: bottom ${side};
        transition: opacity .2s ease, transform .2s ease; pointer-events: none;
      }
      .panel.open { opacity: 1; transform: translateY(0) scale(1); pointer-events: auto; }
      .panel iframe { width: 100%; height: 100%; border: 0; display: block; background: transparent; }
      .hidden { display: none !important; }
      @media (max-width: 480px) {
        .panel { ${side}: 12px; left: 12px; right: 12px; bottom: 12px; top: 12px;
                 width: auto; height: auto; border-radius: 16px; }
        .launcher { ${side}: 16px; bottom: 16px; }
      }
      @media (prefers-reduced-motion: reduce) {
        .launcher, .panel { transition: none; }
      }
    </style>
  `

  const launcher = document.createElement('button')
  launcher.className = 'launcher'
  launcher.setAttribute('aria-label', config.launcherLabel || 'Open chat')
  launcher.innerHTML =
    '<svg viewBox="0 0 24 24" fill="none"><path d="M8 4 H6 a2 2 0 0 0 -2 2 V18 a2 2 0 0 0 2 2 H8" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M16 4 H18 a2 2 0 0 1 2 2 V18 a2 2 0 0 1 -2 2 H16" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><circle cx="12" cy="12" r="3" fill="currentColor"/></svg>' +
    `<span>${(config.launcherLabel || 'Chat').replace(/[<>&]/g, '')}</span>`
  shadow.appendChild(launcher)

  const panel = document.createElement('div')
  panel.className = 'panel hidden'
  shadow.appendChild(panel)

  let iframe: HTMLIFrameElement | null = null
  let open = false
  const listeners: Record<string, ((d?: unknown) => void)[]> = {}
  const emit = (ev: string, d?: unknown) => (listeners[ev] ?? []).forEach((cb) => cb(d))

  function ensureIframe(): HTMLIFrameElement {
    if (iframe) return iframe
    iframe = document.createElement('iframe')
    iframe.src = iframeSrc // creating it here = first open → widget JS + model load starts
    iframe.setAttribute('title', config.title || 'aidekin assistant')
    // allow-popups-to-escape-sandbox: links the widget opens with target="_blank" (e.g. the
    // "Powered by aidekin" footer) become NORMAL top-level tabs instead of inheriting the
    // sandbox. Without it, a popup stays sandboxed and Chrome blocks it from loading any page
    // served with Cross-Origin-Opener-Policy (our whole site is) → ERR_BLOCKED_BY_RESPONSE.
    iframe.setAttribute(
      'sandbox',
      'allow-scripts allow-same-origin allow-forms allow-popups allow-popups-to-escape-sandbox allow-downloads',
    )
    if (wantsMic) iframe.setAttribute('allow', `microphone; cross-origin-isolated`)
    panel.appendChild(iframe)
    return iframe
  }

  function doOpen(): void {
    ensureIframe()
    panel.classList.remove('hidden')
    requestAnimationFrame(() => panel.classList.add('open'))
    launcher.classList.add('hidden')
    open = true
    emit('open')
  }
  function doClose(): void {
    panel.classList.remove('open')
    launcher.classList.remove('hidden')
    open = false
    setTimeout(() => {
      if (!open) panel.classList.add('hidden')
    }, 220)
    emit('close')
  }

  launcher.addEventListener('click', doOpen)

  window.addEventListener('message', (e: MessageEvent) => {
    if (iframe && e.source !== iframe.contentWindow) return
    const data = e.data as { kind?: string } | null
    if (!data || typeof data.kind !== 'string' || !data.kind.startsWith('aidekin:')) return
    if (data.kind === 'aidekin:ready') emit('ready')
    else if (data.kind === 'aidekin:close-request') doClose()
    else if (data.kind === 'aidekin:message') emit('message', data)
  })

  const mount = () => document.body.appendChild(host)
  if (document.body) mount()
  else document.addEventListener('DOMContentLoaded', mount, { once: true })

  window.Aidekin = {
    open: doOpen,
    close: doClose,
    toggle: () => (open ? doClose() : doOpen()),
    // No-op until the panel has opened (the iframe is created lazily). Until then the
    // initial theme comes from data-theme (default 'auto' = the visitor's OS scheme).
    setTheme: (theme) => iframe?.contentWindow?.postMessage({ kind: 'aidekin:theme', theme }, widgetOrigin),
    on: (event, cb) => {
      ;(listeners[event] ??= []).push(cb)
    },
  }
}

init()
