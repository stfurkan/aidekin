import { lazy, Suspense } from 'react'
import { Navigate, Route, Routes } from 'react-router-dom'
import { Loader2 } from 'lucide-react'
import { DEMO_URL, Layout } from './Layout'
import { Landing } from './pages/Landing'
import { NotFound } from './pages/NotFound'
import { useRouteMeta } from './useRouteMeta'

// Code-split the heavier / future routes so the landing stays light (lean by default).
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
          {/* /demo now lives on its own origin (a real third-party embed); redirect old links. */}
          <Route path="/demo" element={<ExternalRedirect to={DEMO_URL} />} />
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

function ExternalRedirect({ to }: { to: string }) {
  window.location.replace(to)
  return <PageSpinner />
}

function PageSpinner() {
  return (
    <div className="grid min-h-[60vh] place-items-center">
      <Loader2 className="size-6 animate-spin text-muted-foreground" />
    </div>
  )
}
