import { Link } from 'react-router-dom'
import { Play } from 'lucide-react'
import { DEMO_URL } from '../Layout'

// 404 page, in the Ledger style (mono-kicker + display heading + jade primary CTA). Rendered
// by the catch-all route inside Layout, so it keeps the site header and footer.
export function NotFound() {
  return (
    <section className="mx-auto flex max-w-2xl flex-col items-center px-5 py-24 text-center">
      <p className="mono-kicker">Error 404</p>
      <h1 className="mt-3 text-balance font-display text-4xl font-semibold tracking-tight">
        Page not found
      </h1>
      <p className="mx-auto mt-3 max-w-md text-lg text-muted-foreground">
        That page doesn&rsquo;t exist or has moved. Everything aidekin does still runs entirely in
        your browser.
      </p>
      <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
        <Link
          to="/"
          className="inline-flex items-center gap-2 rounded-md bg-primary px-5 py-2.5 text-sm font-semibold text-primary-foreground transition-transform hover:-translate-y-px"
        >
          Back home
        </Link>
        <a
          href={DEMO_URL}
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-2 rounded-md border border-border px-5 py-2.5 text-sm font-semibold transition-colors hover:bg-secondary"
        >
          <Play className="size-4" /> Try the demo
        </a>
      </div>
    </section>
  )
}
