import { useEffect } from 'react'

// Dogfood the REAL embed: inject the same loader <script> a customer would paste, so
// aidekin.com runs the actual product (loader → sandboxed iframe → widget), grounded in
// our own knowledge file. This is byte-for-byte what an embedder gets - same launcher,
// same panel. Dev serves the loader from source; the production build emits /loader.js.
// Because the launcher lives outside React (its own Shadow DOM element), it persists
// across page navigation, exactly as it would on a customer's site.

// Our own site's widget config - the exact shape the Configure page emits, including a custom
// systemPrompt (an option offered to every embedder). It sets aidekin's identity and answer style;
// product specifics (page names, URLs, features) are deliberately NOT enumerated here - those come
// from the knowledge file via RAG, so they stay correct as the site changes.
const SYSTEM_PROMPT =
  'You are aidekin, the assistant for aidekin.com - an on-device voice and text AI widget that runs ' +
  'entirely in the browser. Always introduce yourself as aidekin. Answer in 1-2 sentences. Base every ' +
  "factual answer on the information provided to you; if it isn't there, say you're not sure rather than " +
  'guessing. Never output HTML, markdown, or raw URLs; refer to things by name.'

export function SiteWidget() {
  useEffect(() => {
    if (document.getElementById('aidekin-embed')) return
    const s = document.createElement('script')
    s.id = 'aidekin-embed'
    if (import.meta.env.DEV) {
      s.type = 'module'
      s.src = '/src/embed/loader.ts' // Vite serves the loader source in dev
    } else {
      s.src = '/loader.js' // the separately-built IIFE loader in production
    }
    s.defer = true
    s.dataset.aidekin = '' // marks the script for the loader's currentScript fallback
    // Match the site's theme instead of the visitor's OS: the site is light by default (only an
    // explicit toggle sets dark), so the widget should be too. The pre-paint script in index.html has
    // already applied the `dark` class by now, so this reads the site's actual current theme. The
    // toggle in Layout keeps them in sync afterward via window.Aidekin.setTheme.
    s.dataset.theme = document.documentElement.classList.contains('dark') ? 'dark' : 'light'
    s.dataset.mode = 'both'
    s.dataset.title = 'aidekin'
    s.dataset.launcherLabel = 'Ask aidekin'
    s.dataset.greeting = 'Hi! Ask me anything about aidekin.'
    s.dataset.systemPrompt = SYSTEM_PROMPT
    s.dataset.knowledgeUrl = '/aidekin-knowledge.bin'
    document.body.appendChild(s)
  }, [])
  return null
}
