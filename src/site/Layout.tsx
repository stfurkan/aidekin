import { useState, type ReactNode } from 'react'
import { Link, NavLink } from 'react-router-dom'
import { Moon, Sun } from 'lucide-react'
import { cn } from '@/lib/utils'
import { GithubIcon, AidekinMark } from './icons'
import { SiteWidget } from './SiteWidget'

const GITHUB_URL = 'https://github.com/stfurkan/aidekin'

const NAV = [
  { to: '/demo', label: 'Demo' },
  { to: '/configure', label: 'Configure' },
  { to: '/builder', label: 'Knowledge' },
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

function SiteHeader() {
  return (
    <header className="glass sticky top-0 z-40 border-x-0 border-t-0">
      <div className="mx-auto flex h-16 max-w-6xl items-center gap-4 px-5">
        <Logo />
        <nav className="ml-4 hidden items-center gap-1 md:flex">
          {NAV.map((n) => (
            <NavLink
              key={n.to}
              to={n.to}
              className={({ isActive }) =>
                cn(
                  'rounded-md px-3 py-1.5 text-sm font-medium text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground',
                  isActive && 'bg-secondary text-foreground',
                )
              }
            >
              {n.label}
            </NavLink>
          ))}
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
            className="ml-1 hidden rounded-md bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground transition-opacity hover:opacity-90 sm:inline-block"
          >
            Get your snippet
          </Link>
        </div>
      </div>
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
          <AidekinMark className="size-4 text-foreground" coreClassName="fill-primary" />
          <span>aidekin, your own voice and text agent, running 100% in the browser.</span>
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
