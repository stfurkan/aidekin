import { useEffect } from 'react'
import { useLocation } from 'react-router-dom'

// Keep the document title + key meta/canonical in sync with the active route. The static tags
// in index.html are the homepage defaults; this updates them on client navigation so every
// route reports its own title to browser tabs, history, and JS-executing crawlers. (A non-JS
// crawler still sees the static homepage tags — build-time prerender would be the deeper fix.)

const SITE = 'aidekin'
const ORIGIN = 'https://aidekin.com'
const HOME_TITLE = 'aidekin: your own on-device voice and text agent'
const DEFAULT_DESC =
  'A private voice and text AI agent that runs entirely in the browser. No backend, no API keys, no per-message cost.'

interface RouteMeta {
  title: string
  description: string
}

function metaForPath(pathname: string): RouteMeta {
  if (pathname === '/') return { title: HOME_TITLE, description: DEFAULT_DESC }
  if (pathname.startsWith('/docs'))
    return {
      title: `Docs · ${SITE}`,
      description: 'Embed, configure, and ground aidekin: the data attributes, the JavaScript API, RAG, and the privacy model.',
    }
  if (pathname === '/demo')
    return { title: `Live demo · ${SITE}`, description: 'Try aidekin for real: the actual widget, running on-device in your browser.' }
  if (pathname === '/configure')
    return { title: `Configure · ${SITE}`, description: 'Set your options and copy a one-line embed snippet for your own site.' }
  if (pathname === '/builder')
    return {
      title: `Knowledge builder · ${SITE}`,
      description: 'Build a knowledge file in your browser to ground aidekin in your own content (RAG).',
    }
  if (pathname === '/privacy')
    return { title: `Privacy · ${SITE}`, description: 'How aidekin handles data: it does not. Everything runs on the visitor device.' }
  if (pathname === '/terms') return { title: `Terms · ${SITE}`, description: 'Terms for using aidekin.' }
  return { title: `Page not found · ${SITE}`, description: DEFAULT_DESC }
}

function setMeta(attr: 'name' | 'property', key: string, content: string): void {
  let el = document.head.querySelector<HTMLMetaElement>(`meta[${attr}="${key}"]`)
  if (!el) {
    el = document.createElement('meta')
    el.setAttribute(attr, key)
    document.head.appendChild(el)
  }
  el.setAttribute('content', content)
}

export function useRouteMeta(): void {
  const { pathname } = useLocation()
  useEffect(() => {
    const { title, description } = metaForPath(pathname)
    document.title = title
    setMeta('property', 'og:title', title)
    setMeta('name', 'twitter:title', title)
    setMeta('name', 'description', description)
    setMeta('property', 'og:description', description)
    setMeta('name', 'twitter:description', description)
    const url = ORIGIN + pathname
    setMeta('property', 'og:url', url)
    let canonical = document.head.querySelector<HTMLLinkElement>('link[rel="canonical"]')
    if (!canonical) {
      canonical = document.createElement('link')
      canonical.rel = 'canonical'
      document.head.appendChild(canonical)
    }
    canonical.href = url
  }, [pathname])
}
