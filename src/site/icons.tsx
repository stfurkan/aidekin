// Brand mark - "Bracket Core": two brackets clasping a solid core, [ • ]. Encodes the
// product truth - a private, self-contained agent that lives INSIDE someone else's page
// (the brackets read as containment AND as a literal <script> embed; the dot is the
// on-device core). Deliberately NOT a wifi/sonar arc, sparkle, or chat bubble. The solid
// core keeps it legible at 16px. currentColor so it inherits theme/accent; the core can
// be tinted via the `coreClassName` slot when an accent dot is wanted.
export function AidekinMark({
  className,
  coreClassName,
}: {
  className?: string
  coreClassName?: string
}) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M8 4 H6 a2 2 0 0 0 -2 2 V18 a2 2 0 0 0 2 2 H8"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M16 4 H18 a2 2 0 0 1 2 2 V18 a2 2 0 0 1 -2 2 H16"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <circle cx="12" cy="12" r="3" fill="currentColor" className={coreClassName} />
    </svg>
  )
}

// Brand marks not covered by lucide (which dropped brand icons).

export function GithubIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <path d="M12 .5C5.73.5.5 5.73.5 12a11.5 11.5 0 0 0 7.86 10.92c.575.106.785-.25.785-.555 0-.274-.01-1-.015-1.965-3.196.695-3.87-1.54-3.87-1.54-.523-1.33-1.277-1.685-1.277-1.685-1.044-.714.08-.7.08-.7 1.154.082 1.762 1.185 1.762 1.185 1.026 1.758 2.692 1.25 3.347.955.104-.743.402-1.25.73-1.538-2.553-.29-5.236-1.276-5.236-5.68 0-1.255.448-2.28 1.183-3.084-.119-.29-.513-1.46.112-3.043 0 0 .965-.31 3.163 1.178a10.96 10.96 0 0 1 5.76 0c2.196-1.488 3.16-1.178 3.16-1.178.626 1.583.232 2.753.114 3.043.736.804 1.18 1.83 1.18 3.084 0 4.415-2.687 5.387-5.247 5.672.413.355.78 1.057.78 2.13 0 1.538-.014 2.778-.014 3.157 0 .308.207.667.79.554A11.5 11.5 0 0 0 23.5 12C23.5 5.73 18.27.5 12 .5z" />
    </svg>
  )
}
