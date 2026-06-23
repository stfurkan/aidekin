# Deploying aidekin

aidekin is a fully static site plus an embeddable widget. There is no backend to run. A
production build is just files in `dist/`, and every model weight streams from a public CDN
into the visitor's browser on first use, then caches. This guide targets **Cloudflare Workers
(Static Assets)**, Cloudflare's recommended path for new projects (free, unlimited requests,
custom headers), with Cloudflare Pages documented as an alternative. Any static host that lets
you set response headers works.

## What gets built

```bash
npm install
npm run build      # typecheck + app build + loader build -> dist/
```

`dist/` then contains:

| Path | What it is |
|---|---|
| `index.html` | the marketing + app site (single-page app) |
| `widget/index.html` | the embeddable widget, loaded in an iframe |
| `loader.js` | the ~2 KB script customers drop into their site |
| `assets/*` | content-hashed JS, CSS, and WASM |
| `aidekin-knowledge.bin` | the RAG index for the on-site assistant |
| `_headers` | response headers incl. COOP/COEP (copied from `public/`) |
| `og.png`, `favicon.svg`, `embed-example.html` | static assets |

> `_headers` lives in `public/` so the build copies it to the `dist/` root, where Cloudflare
> reads it (supported natively by both Workers and Pages). Do not move it out of `public/`.
> The SPA fallback is configured in `wrangler.jsonc` (`not_found_handling`), not a `_redirects`
> file: Workers rejects the Pages-style `/* /index.html 200` rule as an infinite loop.

## Deploy to Cloudflare Workers (Static Assets)

The repo ships a `wrangler.jsonc` that serves `dist/` as static assets and uses
`not_found_handling: "single-page-application"` so unknown paths fall back to `index.html`
(200) for the client-side router:

```jsonc
{
  "name": "aidekin",
  "compatibility_date": "2026-06-23",
  "assets": {
    "directory": "./dist/",
    "not_found_handling": "single-page-application"
  }
}
```

### Option A: connect the Git repo (recommended)

1. Push this repo to GitHub: `https://github.com/stfurkan/aidekin`
2. In the Cloudflare dashboard: **Workers & Pages -> Create -> Workers -> Import a repository**, and pick the repo.
3. Cloudflare auto-detects the Vite build (`npm run build`) and serves `dist/` per `wrangler.jsonc`, so you do not set build/output manually. Node is pinned by the committed `.node-version` (`24`, the current Active LTS); set a `NODE_VERSION` build variable only to override.
4. Deploy. Every push to the production branch redeploys.

### Option B: direct deploy from your machine

```bash
npm run build
npx wrangler deploy
```

`aidekin-knowledge.bin` is committed to the repo, so the build does **not** regenerate it.
When you change anything in `content/`, rebuild and commit it:

```bash
npm run build-knowledge -- --in content --out public/aidekin-knowledge.bin
npm run verify-knowledge -- public/aidekin-knowledge.bin
```

## Alternative: Cloudflare Pages

Pages is still fully supported (free, unlimited bandwidth, custom domains). To use it instead
of Workers: create a **Pages** project connected to the repo with build command
`npm run build` and output directory `dist`, then **delete `wrangler.jsonc`** (Pages ignores
it) and add a `public/_redirects` file for the SPA fallback (Pages does not use
`not_found_handling`):

```
/*    /index.html   200
```

`_headers` works identically on both targets.

## Custom domains

Add both of these as custom domains on the same project (Workers or Pages), so they serve
identical content:

- **`aidekin.com`**: the site.
- **`cdn.aidekin.com`**: the loader. The snippet uses `https://cdn.aidekin.com/loader.js`,
  and the loader derives the widget origin from its own `src`, so the iframe loads from
  `https://cdn.aidekin.com/widget/`. Both must resolve to this deployment.

If you serve the loader only from the apex, change the snippet to
`https://aidekin.com/loader.js` (the configurator at `/configure` emits whatever origin you
point it at). Customers can always override with `data-widget-origin`. (Workers serves custom
domains only for zones whose nameservers are on Cloudflare, which aidekin already uses.)

## Headers: why the site is cross-origin isolated

`public/_headers` sends `Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp` across the whole site. This is required for
voice: the VAD and the ASR decoder use `SharedArrayBuffer` + threaded WASM, which only work
on a cross-origin-isolated page. An iframe can be isolated only if its top-level page is too,
so the marketing site itself must carry these headers for the on-site `/demo` and widget to
do voice. Text mode needs only WebGPU and is unaffected either way.

Every cross-origin resource the site loads sends `Cross-Origin-Resource-Policy`, so they all
pass under `require-corp`: Google Fonts (`fonts.googleapis.com`, `fonts.gstatic.com`),
Hugging Face model weights (`huggingface.co`), and the onnxruntime-web + VAD runtime
(`cdn.jsdelivr.net`). If you add any new third-party resource, confirm it sends
`Cross-Origin-Resource-Policy: cross-origin` (or `Access-Control-Allow-Origin`) or it will be
blocked.

`/widget/*` and `/loader.js` additionally send `Cross-Origin-Resource-Policy: cross-origin`
so a customer page that is itself cross-origin isolated can still embed the iframe and load
the loader.

## Verifying a deploy

After the first deploy, confirm the headers are live:

```bash
curl -sI https://aidekin.com/ | grep -i cross-origin
# expect: cross-origin-opener-policy: same-origin
#         cross-origin-embedder-policy: require-corp

curl -sI https://cdn.aidekin.com/loader.js | grep -i 'cross-origin-resource\|cache-control'
# expect: cross-origin-resource-policy: cross-origin
```

Confirm the SPA fallback is not masking missing files: these must return their real content
type, not `text/html` (an HTML body means the single-page-application fallback served the app
shell instead of the real asset, which then fails as "Failed to load module script"):

```bash
curl -sI https://aidekin.com/widget/      | grep -i content-type   # expect: text/html
curl -sI https://cdn.aidekin.com/loader.js | grep -i content-type   # expect: javascript
```

Then in a WebGPU-capable browser:

1. Open `https://aidekin.com/` and click the launcher. Text chat should stream a reply.
2. Open `https://aidekin.com/demo`, launch it, and try the mic. Voice should work end to end.
3. DevTools console should show no COOP/COEP warnings, and `crossOriginIsolated` should be
   `true` on both the page and the widget iframe.

## Customers embedding the widget

A customer needs only the one-line snippet from `/configure`. They get the same launcher
button and the same experience as the on-site widget. Two notes worth documenting for them:

- **Voice on a non-isolated page runs degraded.** If the customer's own page is not
  cross-origin isolated, the widget iframe cannot use threaded WASM, so voice falls back to
  single-threaded (or text only). Text always works. For full-speed voice, the customer sets
  `COOP: same-origin` + `COEP: require-corp` on their own page.
- **The model caches per site.** Browsers partition storage by top-level site, so a visitor
  downloads the model once per site that embeds aidekin, then loads from cache on return.

## Hosting a knowledge file

The on-site `aidekin-knowledge.bin` is served from this deployment. Customers host their own
`knowledge.bin` anywhere that allows cross-origin reads (their own site with
`Access-Control-Allow-Origin: *`, a GitHub repo via jsDelivr `/gh/`, or Cloudflare R2) and
point the widget at it with `data-knowledge-url`. It is downloaded by every visitor, so it is
effectively public: never put secrets in it.

## Model-hosting resilience (optional)

By default all weights stream from the Hugging Face Hub. For very high-traffic deployments
you can mirror them and rebuild with `VITE_MODEL_CDN=https://your-bucket.example`, or fork
the model repos and change the `hfModelId` values in `src/models/registry.ts`. See the README
"Models" table for the full list.
