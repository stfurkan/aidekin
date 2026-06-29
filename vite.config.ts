import { readFileSync, rmSync } from 'node:fs'
import { join } from 'node:path'
import { fileURLToPath, URL } from 'node:url'
import type { Plugin } from 'vite'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Cross-origin isolation is REQUIRED for SharedArrayBuffer + WASM threads, which the threaded
// onnxruntime-web (speech models) depends on. Safari does NOT support COEP: credentialless, so we use
// require-corp. All model/runtime assets load from CORS+CORP-clean CDNs (Hugging Face, jsDelivr), so
// they pass under require-corp.
const COI_HEADERS: Record<string, string> = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
}

function crossOriginIsolation(): Plugin {
  const apply = (res: { setHeader(k: string, v: string): void }) => {
    for (const [k, v] of Object.entries(COI_HEADERS)) res.setHeader(k, v)
  }
  return {
    name: 'aidekin:cross-origin-isolation',
    configureServer(server) {
      server.middlewares.use((_req, res, next) => {
        apply(res)
        next()
      })
    },
    configurePreviewServer(server) {
      server.middlewares.use((_req, res, next) => {
        apply(res)
        next()
      })
    },
  }
}

// Read the EXACT installed version of a dependency whose wasm/assets we load from a versioned
// jsDelivr URL at runtime, so that CDN pin can never drift from the bundled JS. Single source
// of truth = package.json / the lockfile. Injected as globals consumed in src/models/registry.ts.
function installedVersion(pkg: string): string {
  const url = new URL(`./node_modules/${pkg}/package.json`, import.meta.url)
  return (JSON.parse(readFileSync(url, 'utf8')) as { version: string }).version
}

// The gitignored local dev mirrors (public/llm/model_q1.onnx_data, public/models/*) are served
// same-origin only for offline dev. In production the LLM data + embedder + speech models all stream
// from HF/CDN, so these must NOT ship. A git-based deploy never has them, but strip them from the
// build output too, so a local build can't accidentally ship the 277MB weights.
function stripDevModelMirrors(): Plugin {
  return {
    name: 'aidekin:strip-dev-model-mirrors',
    apply: 'build',
    closeBundle() {
      const out = fileURLToPath(new URL('./dist', import.meta.url))
      for (const p of ['llm/model_q1.onnx_data', 'models']) rmSync(join(out, p), { recursive: true, force: true })
    },
  }
}

export default defineConfig({
  plugins: [react(), tailwindcss(), crossOriginIsolation(), stripDevModelMirrors()],
  define: {
    __ORT_VERSION__: JSON.stringify(installedVersion('onnxruntime-web')),
    __VAD_VERSION__: JSON.stringify(installedVersion('@ricky0123/vad-web')),
  },
  resolve: {
    alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) },
  },
  worker: { format: 'es' },
  optimizeDeps: {
    // Pre-scan the speech workers at startup so their npm deps (fft.js, vad-web) are
    // bundled up front. Without this, the dep scanner never crawls `new Worker(new
    // URL(...))` workers, so the FIRST mic tap discovers new deps, re-optimizes, and
    // force-reloads the page (dev only). Listing the workers here scans them at boot.
    entries: ['index.html', 'widget/index.html', 'src/workers/*.worker.ts'],
    exclude: ['onnxruntime-web'],
  },
  build: {
    // The large chunks (transformers, onnxruntime, pdf.js, the embedder) are all loaded via
    // dynamic import() and never on the initial path, so the default 500 kB warning is noise.
    chunkSizeWarningLimit: 700,
    rollupOptions: {
      // Two entries share /src: the product site (index.html) and the embeddable
      // widget served at /widget/ (its own minimal bundle, no site code).
      input: {
        main: fileURLToPath(new URL('./index.html', import.meta.url)),
        widget: fileURLToPath(new URL('./widget/index.html', import.meta.url)),
      },
    },
    // The AudioWorklet must be a standalone file for audioWorklet.addModule() -
    // never inline it as a data: URL (fragile under COEP cross-origin isolation).
    assetsInlineLimit: (filePath: string) =>
      filePath.endsWith('pcmWorklet.js') ? false : undefined,
  },
  server: { headers: { ...COI_HEADERS } },
  preview: { headers: { ...COI_HEADERS } },
})
