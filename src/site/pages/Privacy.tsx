import { Link } from 'react-router-dom'
import { type ReactNode } from 'react'

export default function Privacy() {
  return (
    <section className="mx-auto max-w-2xl px-5 py-14">
      <p className="mono-kicker">Privacy</p>
      <h1 className="mt-3 font-display text-4xl font-semibold tracking-tight">Privacy policy</h1>
      <p className="mt-2 text-sm text-muted-foreground">Last updated June 2026</p>

      <div className="mt-8 space-y-6 text-[15px] leading-relaxed text-muted-foreground">
        <p>
          aidekin is built so that conversations never leave the visitor&rsquo;s device. The widget
          runs entirely in the browser. There is no aidekin backend, and the software sends no
          messages, transcripts, or audio to us or anyone else.
        </p>

        <H>What stays on your device</H>
        <p>
          Chat history (only if the embedding site enables it), your theme preference, and the
          downloaded AI model are stored locally in your browser (localStorage, Cache Storage, and
          OPFS). You can clear them at any time from the widget&rsquo;s settings or your browser, which
          removes that data.
        </p>

        <H>Model downloads</H>
        <p>
          The first time the assistant runs, the model files are downloaded from third-party content
          delivery networks (such as the Hugging Face Hub and jsDelivr). Those providers receive the
          network request needed to serve the files (for example, your IP address) under their own
          privacy policies; aidekin is not affiliated with them. After the first download the model is
          cached and runs offline.
        </p>

        <H>Knowledge files</H>
        <p>
          If a site grounds the assistant in its own content, that knowledge file is downloaded to your
          browser and used locally for retrieval. It is provided by the site owner, not by aidekin.
        </p>

        <H>Microphone</H>
        <p>
          Voice mode requests microphone access only when you turn it on, and audio is processed
          entirely on your device for on-device speech recognition. It is never uploaded.
        </p>

        <H>Sites that embed aidekin</H>
        <p>
          aidekin is a tool that website owners add to their own sites. Each site is responsible for
          its own deployment and its own privacy practices. This policy covers the aidekin software and
          aidekin.com only.
        </p>

      </div>

      <p className="mt-10 border-t border-border pt-6 text-sm">
        See the{' '}
        <Link to="/docs" className="text-primary hover:underline">
          docs
        </Link>{' '}
        for the technical details, or the{' '}
        <Link to="/terms" className="text-primary hover:underline">
          terms
        </Link>
        .
      </p>
    </section>
  )
}

function H({ children }: { children: ReactNode }) {
  return <h2 className="font-display text-xl font-semibold tracking-tight text-foreground">{children}</h2>
}
