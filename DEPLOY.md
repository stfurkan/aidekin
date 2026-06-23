# Deploying aidekin

aidekin is a fully static site plus an embeddable widget. There is no backend to run. A
production build is just files in `dist/`, and every model weight streams from a public CDN
into the visitor's browser on first use, then caches. This guide uses **Cloudflare Pages**
(free, unlimited bandwidth, custom headers), but any static host that lets you set response
headers works.

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
| `_headers`, `_redirects` | Cloudflare Pages routing + headers (copied from `public/`) |
| `og.png`, `favicon.svg`, `embed-example.html` | static assets |

> `_headers` and `_redirects` live in `public/` so the build copies them to the `dist/`
> root, where Cloudflare reads them (Pages and Workers both support these files natively). Do
> not move them out of `public/`.

## Cloudflare Pages

> Cloudflare now points new projects at **Workers (Static Assets)** and focuses new investment
> there. Pages is **not** deprecated: it stays fully supported (free plan, unlimited
> bandwidth, custom domains), and for a pure static site it is the lowest-friction option, so
> it is the default here. The recommended Workers path is documented below; the build output,
> `_headers`, and `_redirects` are identical either way.

### Option A: connect the Git repo (recommended)

1. Push this repo to GitHub: `https://github.com/stfurkan/aidekin`
2. In the Cloudflare dashboard: **Workers & Pages -> Create -> Pages -> Connect to Git**.
3. Pick the repo and set:
   - **Build command:** `npm run build`
   - **Build output directory:** `dist`
   - **Node version:** 24, the current Active LTS. The repo ships a `.node-version` file pinned to `24`, which Cloudflare reads automatically, so you normally need to set nothing. To override, add a `NODE_VERSION` environment variable. This matches `engines` in package.json (`>=24.0.0`).
4. Deploy. Every push to the production branch redeploys.

`aidekin-knowledge.bin` is committed to the repo, so the Cloudflare build does **not**
rebuild it. When you change anything in `content/`, regenerate and commit it:

```bash
npm run build-knowledge -- --in content --out public/aidekin-knowledge.bin
npm run verify-knowledge -- public/aidekin-knowledge.bin
```

### Option B: direct upload

```bash
npm run build
npx wrangler pages deploy dist --project-name aidekin
```

## Cloudflare Workers (Static Assets)

Workers Static Assets is Cloudflare's recommended target for new projects. This site runs on
it unchanged: same `npm run build`, same `dist/`, same `_headers`. Add a `wrangler.jsonc` at
the repo root:

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

Then build and deploy:

```bash
npm run build
npx wrangler deploy
```

`not_found_handling: "single-page-application"` is the native SPA fallback, equivalent to the
`/* /index.html 200` rule in `_redirects` (you can drop `_redirects` on this path; keep
`_headers`, which Workers reads natively). Custom domains work as on Pages, with one caveat:
Workers only serves custom domains whose nameservers are managed by Cloudflare, which aidekin
already uses.

## Custom domains

Add both of these as custom domains on the same Pages project, so they serve identical
content:

- **`aidekin.com`**: the site.
- **`cdn.aidekin.com`**: the loader. The snippet uses `https://cdn.aidekin.com/loader.js`,
  and the loader derives the widget origin from its own `src`, so the iframe loads from
  `https://cdn.aidekin.com/widget/`. Both must resolve to this deployment.

If you serve the loader only from the apex, change the snippet to
`https://aidekin.com/loader.js` (the configurator at `/configure` emits whatever origin you
point it at). Customers can always override with `data-widget-origin`.

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

Confirm the SPA catch-all is not masking missing files: these must return their real content
type, not `text/html` (an HTML body means the `/* -> /index.html 200` fallback served the app
shell instead of the real asset, which then fails as "Failed to load module script").

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
