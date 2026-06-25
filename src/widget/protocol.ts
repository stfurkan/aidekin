// Shared contract between the host page (embed loader) and the widget iframe.
// Types only + a few tiny pure helpers, so the loader can import it with zero
// runtime dependencies (everything type-only is erased).

import type { WidgetMode } from '@/core/capabilities'

export interface WidgetConfig {
  /** text | voice | both — voice is an in-widget opt-in toggle. */
  mode: WidgetMode
  /** Owner-set persona. Delivered to the visitor's browser → NOT a secret. */
  systemPrompt?: string
  /** First assistant bubble shown before the user types anything. */
  greeting?: string
  /** URL of a precomputed knowledge.bin for RAG (omit = no RAG, nothing loads). */
  knowledgeUrl?: string
  ragTopK?: number
  /** Reason internally on every turn (slower, more accurate). RAG turns always reason. */
  reasoning?: boolean
  theme?: 'light' | 'dark' | 'auto'
  /** CSS color for the accent (launcher + send button + assistant bubble). */
  accent?: string
  position?: 'bottom-right' | 'bottom-left'
  /** Text shown on the floating launcher. */
  launcherLabel?: string
  /** Title shown in the panel header. */
  title?: string
  /** Persist the conversation across reloads (scoped per host origin). */
  persist?: boolean
  /** Extra origins (besides the embedding page) allowed to drive the widget. */
  allowedOrigins?: string[]
  /** Where the widget iframe is served from (defaults to the loader's own origin). */
  widgetOrigin?: string
}

/** The effective system prompt: the owner's custom one if set, otherwise a friendly
 *  default that adopts the widget's title as the assistant's name — so it introduces
 *  itself as "Aidekin" (or the owner's chosen name), never the underlying model name. */
export function resolveSystemPrompt(config: Pick<WidgetConfig, 'systemPrompt' | 'title'>): string {
  const custom = config.systemPrompt?.trim()
  if (custom) return custom
  const name = config.title?.trim() || 'aidekin'
  return (
    `You are ${name}, a friendly, helpful assistant embedded on this website, running entirely in ` +
    `the user's browser. Your name is ${name}; always introduce yourself as ${name}. Answer only the ` +
    `question that was asked, directly and in 1-2 sentences. Never volunteer unrelated information.`
  )
}

export const WIDGET_DEFAULTS = {
  mode: 'text' as WidgetMode,
  theme: 'auto' as const,
  position: 'bottom-right' as const,
  title: 'Assistant',
  launcherLabel: 'Chat with us',
  persist: true,
  ragTopK: 3,
}

/** Merge owner config over the built-in defaults. */
export function withDefaults(config: Partial<WidgetConfig>): WidgetConfig {
  return { ...WIDGET_DEFAULTS, ...config, mode: config.mode ?? WIDGET_DEFAULTS.mode }
}

// ── postMessage protocol ─────────────────────────────────────────────────────
// host → widget
export type HostMessage =
  | { readonly kind: 'aidekin:open' }
  | { readonly kind: 'aidekin:close' }
  | { readonly kind: 'aidekin:theme'; readonly theme: 'light' | 'dark' }

// widget → host
export type WidgetMessage =
  | { readonly kind: 'aidekin:ready' }
  | { readonly kind: 'aidekin:close-request' }
  | { readonly kind: 'aidekin:message'; readonly role: 'user' | 'assistant'; readonly text: string }
  | { readonly kind: 'aidekin:error'; readonly where: string; readonly message: string }
  // Visitor toggled the in-widget theme → the host loader saves it so the launcher/loading
  // overlay match on the next open (the loader can't read the iframe's own storage cross-origin).
  | { readonly kind: 'aidekin:theme-changed'; readonly theme: 'light' | 'dark' }

export function isAidekinMessage(data: unknown): data is { kind: string } {
  return (
    typeof data === 'object' &&
    data !== null &&
    typeof (data as { kind?: unknown }).kind === 'string' &&
    (data as { kind: string }).kind.startsWith('aidekin:')
  )
}

/** Decode a WidgetConfig passed to the iframe via the URL hash (#<encoded JSON>). */
export function parseConfigFromHash(hash: string): Partial<WidgetConfig> {
  const raw = hash.replace(/^#/, '')
  if (!raw) return {}
  try {
    return JSON.parse(decodeURIComponent(raw)) as Partial<WidgetConfig>
  } catch {
    return {}
  }
}
