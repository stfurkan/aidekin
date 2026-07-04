import { useState, useEffect, type ReactNode } from 'react'
import { Link, NavLink, useLocation } from 'react-router-dom'
import { Menu, Moon, Sun, X } from 'lucide-react'
import { cn } from '@/lib/utils'
import { GithubIcon, AidekinMark } from './icons'
import { SiteWidget } from './SiteWidget'

const GITHUB_URL = 'https://github.com/stfurkan/aidekin'
// The demo is a REAL third-party embed (a fictional cafe on its own origin), which is a far more
// honest demonstration than a page on our own site - so Demo opens it in a new tab.
export const DEMO_URL = 'https://stfurkan.github.io/aidekin-demo/'

const NAV: Array<{ to?: string; href?: string; label: string }> = [
  { href: DEMO_URL, label: 'Demo' },
  { to: '/configure', label: 'Configure' },
  { to: '/knowledge', label: 'Knowledge' },
  { to: '/docs', label: 'Docs' },
]

export function Layout({ children }: { children: ReactNode }) {
  return (
    <div className="flex min-h-full flex-col">
      <SiteHeader />
      <main className="flex-1">{children}</main>
      <SiteFooter />
      <SiteWidget />
    </div>
  )
}

function Logo() {
  return (
    <Link to="/" className="group flex items-center gap-2.5">
      <AidekinMark
        className="size-6 text-foreground transition-transform group-hover:-translate-y-px"
        coreClassName="fill-primary"
      />
      <span className="font-display text-[17px] font-semibold tracking-tight">aidekin</span>
    </Link>
  )
}

const navLinkClass = ({ isActive }: { isActive: boolean }) =>
  cn(
    'rounded-md px-3 py-1.5 text-sm font-medium text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground',
    isActive && 'bg-secondary text-foreground',
  )

function SiteHeader() {
  const [open, setOpen] = useState(false)
  const { pathname } = useLocation()

  // Close the mobile menu whenever the route changes (e.g. after tapping a link).
  useEffect(() => setOpen(false), [pathname])

  // While the mobile menu is open: close on Escape and lock background scroll (the backdrop
  // overlays the page), restoring both on close.
  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setOpen(false)
    window.addEventListener('keydown', onKey)
    document.documentElement.style.overflow = 'hidden'
    return () => {
      window.removeEventListener('keydown', onKey)
      document.documentElement.style.overflow = ''
    }
  }, [open])

  return (
    <header className="glass sticky top-0 z-40 border-x-0 border-t-0">
      <div className="mx-auto flex h-16 max-w-6xl items-center gap-4 px-5">
        <Logo />
        <nav className="ml-4 hidden items-center gap-1 md:flex">
          {NAV.map((n) =>
            n.href ? (
              <a key={n.label} href={n.href} target="_blank" rel="noreferrer" className={navLinkClass({ isActive: false })}>
                {n.label}
              </a>
            ) : (
              <NavLink key={n.label} to={n.to!} className={navLinkClass}>
                {n.label}
              </NavLink>
            ),
          )}
        </nav>
        <div className="ml-auto flex items-center gap-1.5">
          <ThemeToggle />
          <a
            href={GITHUB_URL}
            target="_blank"
            rel="noreferrer"
            className="grid size-9 place-items-center rounded-md text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
            aria-label="GitHub repository"
          >
            <GithubIcon className="size-[18px]" />
          </a>
          <Link
            to="/configure"
            className="ml-1 hidden rounded-md bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground transition-opacity hover:opacity-90 md:inline-block"
          >
            Get your snippet
          </Link>
          <button
            type="button"
            onClick={() => setOpen((v) => !v)}
            aria-label={open ? 'Close menu' : 'Open menu'}
            aria-expanded={open}
            aria-controls="mobile-nav"
            className="grid size-9 place-items-center rounded-md text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground md:hidden"
          >
            {open ? <X className="size-5" /> : <Menu className="size-5" />}
          </button>
        </div>
      </div>

      {open && (
        <>
          {/* Backdrop: overlays the page (does not shift it) and closes on an outside tap. */}
          <div
            aria-hidden="true"
            onClick={() => setOpen(false)}
            className="fixed inset-x-0 bottom-0 top-16 bg-black/20 md:hidden"
          />
          {/* Menu: absolutely positioned just under the bar, so it floats over content. */}
          <nav
            id="mobile-nav"
            className="absolute inset-x-0 top-full border-b border-border bg-background shadow-xl md:hidden"
          >
            <div className="mx-auto flex max-w-6xl flex-col gap-1 px-5 py-3">
              {NAV.map((n) =>
                n.href ? (
                  <a
                    key={n.label}
                    href={n.href}
                    target="_blank"
                    rel="noreferrer"
                    className="rounded-md px-3 py-2.5 text-sm font-medium text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
                  >
                    {n.label}
                  </a>
                ) : (
                  <NavLink
                    key={n.label}
                    to={n.to!}
                    className={({ isActive }) =>
                      cn(
                        'rounded-md px-3 py-2.5 text-sm font-medium text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground',
                        isActive && 'bg-secondary text-foreground',
                      )
                    }
                  >
                    {n.label}
                  </NavLink>
                ),
              )}
              <Link
                to="/configure"
                className="mt-1 rounded-md bg-primary px-4 py-2.5 text-center text-sm font-semibold text-primary-foreground transition-opacity hover:opacity-90"
              >
                Get your snippet
              </Link>
            </div>
          </nav>
        </>
      )}
    </header>
  )
}

function ThemeToggle() {
  const [dark, setDark] = useState(() => document.documentElement.classList.contains('dark'))
  const toggle = () => {
    const next = !dark
    setDark(next)
    document.documentElement.classList.toggle('dark', next)
    try {
      localStorage.setItem('aidekin-theme', next ? 'dark' : 'light')
    } catch {
      /* ignore */
    }
  }
  return (
    <button
      type="button"
      onClick={toggle}
      aria-label="Toggle theme"
      className="grid size-9 place-items-center rounded-md text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
    >
      {dark ? <Sun className="size-[18px]" /> : <Moon className="size-[18px]" />}
    </button>
  )
}

function SiteFooter() {
  return (
    <footer className="border-t border-border">
      <div className="mx-auto flex max-w-6xl flex-col items-center justify-between gap-4 px-5 py-8 text-sm text-muted-foreground sm:flex-row">
        <div className="flex items-center gap-2.5">
          <AidekinMark className="size-4 shrink-0 text-foreground" coreClassName="fill-primary" />
          <span>
            <span className="font-medium text-foreground">aidekin</span>
            <span className="hidden sm:inline">
              , your own voice and text agent, running 100% in the browser.
            </span>
          </span>
        </div>
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
          <Link to="/docs" className="hover:text-foreground">
            Docs
          </Link>
          <Link to="/privacy" className="hover:text-foreground">
            Privacy
          </Link>
          <Link to="/terms" className="hover:text-foreground">
            Terms
          </Link>
          <a href={GITHUB_URL} target="_blank" rel="noreferrer" className="hover:text-foreground">
            GitHub
          </a>
          <span className="opacity-60">MIT licensed</span>
        </div>
      </div>
    </footer>
  )
}
