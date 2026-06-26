import { lazy, Suspense } from 'react'
import { Navigate, Route, Routes } from 'react-router-dom'
import { Loader2 } from 'lucide-react'
import { Layout } from './Layout'
import { Landing } from './pages/Landing'
import { NotFound } from './pages/NotFound'
import { useRouteMeta } from './useRouteMeta'

// Code-split the heavier / future routes so the landing stays light (lean by default).
const Demo = lazy(() => import('./pages/Demo'))
const Configure = lazy(() => import('./pages/Configure'))
const Builder = lazy(() => import('./pages/Builder'))
const Docs = lazy(() => import('./pages/Docs'))
const Privacy = lazy(() => import('./pages/Privacy'))
const Terms = lazy(() => import('./pages/Terms'))

export function SiteApp() {
  useRouteMeta()
  return (
    <Layout>
      <Suspense fallback={<PageSpinner />}>
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/demo" element={<Demo />} />
          <Route path="/configure" element={<Configure />} />
          <Route path="/knowledge" element={<Builder />} />
          {/* Keep the old /builder path as a redirect so existing links/bookmarks still work. */}
          <Route path="/builder" element={<Navigate to="/knowledge" replace />} />
          <Route path="/docs/*" element={<Docs />} />
          <Route path="/privacy" element={<Privacy />} />
          <Route path="/terms" element={<Terms />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </Suspense>
    </Layout>
  )
}

function PageSpinner() {
  return (
    <div className="grid min-h-[60vh] place-items-center">
      <Loader2 className="size-6 animate-spin text-muted-foreground" />
    </div>
  )
}
