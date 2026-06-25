import { useEffect } from 'react'

// Dogfood the REAL embed: inject the same loader <script> a customer would paste, so
// aidekin.com runs the actual product (loader → sandboxed iframe → widget), grounded in
// our own knowledge file. This is byte-for-byte what an embedder gets - same launcher,
// same panel. Dev serves the loader from source; the production build emits /loader.js.
// Because the launcher lives outside React (its own Shadow DOM element), it persists
// across page navigation, exactly as it would on a customer's site.

const SYSTEM_PROMPT =
  'You are aidekin, an on-device voice and text AI assistant, answering on aidekin.com (the site for ' +
  'the aidekin product). Always introduce yourself as aidekin. Help visitors understand and use ' +
  'aidekin: answer in 1-2 sentences and point them to the right page (configure, builder, docs) when ' +
  "useful. If you don't know, say so."

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
