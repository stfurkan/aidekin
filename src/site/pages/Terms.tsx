import { Link } from 'react-router-dom'
import { type ReactNode } from 'react'

export default function Terms() {
  return (
    <section className="mx-auto max-w-2xl px-5 py-14">
      <p className="mono-kicker">Terms</p>
      <h1 className="mt-3 font-display text-4xl font-semibold tracking-tight">Terms of use</h1>
      <p className="mt-2 text-sm text-muted-foreground">Last updated June 2026</p>

      <div className="mt-8 space-y-6 text-[15px] leading-relaxed text-muted-foreground">
        <p>
          aidekin is free, open-source software released under the MIT license. It is provided &ldquo;as
          is&rdquo;, without warranty of any kind. By using or embedding it, you agree to these terms.
        </p>

        <H>AI output</H>
        <p>
          The assistant runs a small language model on the visitor&rsquo;s device. Its answers can be
          inaccurate, incomplete, or inappropriate. Do not rely on them for important decisions, and
          verify anything that matters. Treat the assistant as a helpful starting point, not an
          authority.
        </p>

        <H>Your responsibilities</H>
        <p>
          If you embed aidekin, you are responsible for how you deploy it, the persona and knowledge
          you configure, and compliance with the underlying model licenses and applicable law. Because
          the system prompt and any knowledge file are delivered to every visitor&rsquo;s browser,
          never put secrets or private data in them.
        </p>

        <H>No warranty and no liability</H>
        <p>
          To the maximum extent permitted by law, the software is provided without warranties, and the
          authors are not liable for any claim, damages, or other liability arising from its use, as
          set out in the MIT license.
        </p>

        <H>Changes</H>
        <p>These terms may be updated; the current version on this page governs.</p>
      </div>

      <p className="mt-10 border-t border-border pt-6 text-sm">
        See the{' '}
        <Link to="/privacy" className="text-primary hover:underline">
          privacy policy
        </Link>{' '}
        or the{' '}
        <a
          href="https://github.com/stfurkan/aidekin/blob/main/LICENSE"
          target="_blank"
          rel="noreferrer"
          className="text-primary hover:underline"
        >
          MIT license
        </a>
        .
      </p>
    </section>
  )
}

function H({ children }: { children: ReactNode }) {
  return <h2 className="font-display text-xl font-semibold tracking-tight text-foreground">{children}</h2>
}
