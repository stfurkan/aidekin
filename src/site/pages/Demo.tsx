import { useEffect, useState } from 'react'
import { Play, MonitorSmartphone } from 'lucide-react'
import { WidgetApp } from '@/widget/WidgetApp'
import { WidgetFrame } from '@/widget/WidgetFrame'
import { SonarPing } from '@/widget/SonarPing'
import { withDefaults } from '@/widget/protocol'
import { probeCapabilities } from '@/core/capabilities'

// Preview + opt-in: nothing heavy loads until the visitor clicks "Launch". Then the
// real text widget mounts inline and downloads the ~290 MB model (one-time, cached).
export default function Demo() {
  const [launched, setLaunched] = useState(false)
  const [webgpu, setWebgpu] = useState<boolean | null>(null)
  const config = withDefaults({
    mode: 'both',
    title: 'aidekin',
    greeting: 'Hi! Ask me anything about aidekin.',
    knowledgeUrl: '/aidekin-knowledge.bin',
  })

  // Proactively detect WebGPU so unsupported visitors see what they need up front.
  useEffect(() => {
    let alive = true
    void probeCapabilities().then((r) => {
      if (alive) setWebgpu(r.webgpu.supported)
    })
    return () => {
      alive = false
    }
  }, [])

  return (
    <section className="mx-auto max-w-3xl px-5 py-16 text-center">
      <p className="mono-kicker">Live demo</p>
      <h1 className="mt-3 text-balance font-display text-4xl font-semibold tracking-tight">
        See it for real
      </h1>
      <p className="mx-auto mt-3 max-w-xl text-lg text-muted-foreground">
        This is the actual widget, running in your browser right here. No servers involved.
      </p>

      <div className="mt-10 flex justify-center">
        <WidgetFrame>
          {webgpu === false ? (
            <div className="flex h-full flex-col items-center justify-center gap-4 px-8 text-center">
              <MonitorSmartphone className="size-7 text-muted-foreground" />
              <div>
                <p className="text-sm font-medium">This browser can’t run the demo</p>
                <p className="mt-1.5 text-xs text-muted-foreground">
                  This demo needs WebGPU. Try the latest Chrome or Edge on desktop, Safari 26+, or
                  Firefox 145+.
                </p>
              </div>
            </div>
          ) : launched ? (
            <WidgetApp config={config} loadOnMount />
          ) : (
            <div className="flex h-full flex-col items-center justify-center gap-5 px-8 text-center">
              <SonarPing state="listening" className="size-24" />
              <button
                type="button"
                onClick={() => setLaunched(true)}
                className="inline-flex items-center gap-2 rounded-md bg-primary px-5 py-2.5 text-sm font-semibold text-primary-foreground transition-transform hover:-translate-y-px"
              >
                <Play className="size-4" /> Launch live demo
              </button>
              <p className="max-w-[260px] text-center text-xs leading-relaxed text-muted-foreground">
                Runs in your browser, downloads on first use. Needs WebGPU.
              </p>
            </div>
          )}
        </WidgetFrame>
      </div>

      <p className="mx-auto mt-8 max-w-md text-center text-sm text-muted-foreground">
        Choose text (recommended) or voice when it opens — the same assistant either way, running
        entirely in your browser. Voice is a beta opt-in and adds a one-time ~1.6&nbsp;GB download.
      </p>
    </section>
  )
}
