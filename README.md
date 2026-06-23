# aidekin

**Your own voice + text AI agent, embedded with one script tag, running 100% in the visitor's browser.**

aidekin is an open-source, client-side AI assistant you drop onto any website. Your visitors get a private voice and text agent that runs entirely on their own device via WebGPU. There is no backend, no API keys, and no per-message cost. Nothing they type or say leaves their machine.

[aidekin.com](https://aidekin.com) · [Configure](https://aidekin.com/configure) · [Knowledge builder](https://aidekin.com/builder) · [Docs](https://aidekin.com/docs) · [Demo](https://aidekin.com/demo) · [GitHub](https://github.com/stfurkan/aidekin) · MIT licensed

---

## Add it to your site

Paste one line, just before the closing `</body>` tag:

```html
<script src="https://cdn.aidekin.com/loader.js" data-title="Acme" defer></script>
```

On page load the loader (~2 KB) only draws a floating launcher. The widget and the model load on the **first open**, so there is zero impact on your page load. Generate a snippet tailored to your settings at [aidekin.com/configure](https://aidekin.com/configure).

## Why aidekin

- **On-device.** The language model, retrieval, and speech all run in the visitor's browser. No servers.
- **Free per message.** The visitor's device does the work. No tokens, no metering, no bills.
- **Private by design.** Nothing leaves the device. Works offline after the first load, and behind firewalls.
- **Voice + text, one brain.** The same model answers whether the visitor types or talks.
- **Your own knowledge (RAG).** Feed it your docs; retrieval runs in the browser.
- **One script tag.** No build step, no backend, no keys. MIT licensed; self-host if you prefer.

## How it works

```
host page ──<script> loader (~2 KB, Shadow DOM launcher)──► on first open: <iframe> widget
                                                                  │ origin-checked postMessage
                                  text (default) ── mic toggle ─► voice (speech models load lazily)
                                        both run on-device:
   VAD ─► Smart-Turn ─► ASR (Nemotron) ─► LLM (Bonsai-1.7B) ─► TTS (Supertonic), all WebGPU/WASM
                                              └─ optional RAG over your knowledge.bin
```

Text needs only WebGPU. Voice adds on-device speech recognition and synthesis, loaded only when a visitor first taps the mic. Replies stream clause-by-clause into TTS so audio starts while the model is still writing, and barge-in lets the visitor interrupt.

## Configuration

Configure with `data-*` attributes on the script tag (or a `window.AidekinConfig` object). The common ones:

| attribute | values | default |
|---|---|---|
| `data-mode` | `text` · `voice` · `both` | `text` |
| `data-title` | string (name + header) | `Assistant` |
| `data-greeting` | string | none |
| `data-accent` | CSS color | brand jade |
| `data-knowledge-url` | URL of a `knowledge.bin` | none |
| `data-reasoning` | `true` · `false` | `false` |
| `data-theme` | `light` · `dark` · `auto` | `auto` |

Full reference + the JavaScript API (`window.Aidekin.open/close/toggle/setTheme/on`) at [aidekin.com/docs](https://aidekin.com/docs).

## Ground it in your own content (RAG)

Build a small `knowledge.bin` and point the widget at it with `data-knowledge-url`. Retrieval runs in the browser; at query time only the visitor's question is embedded.

- **In the browser:** [aidekin.com/builder](https://aidekin.com/builder): drag in PDF, Word, text, Markdown, HTML, CSV, or JSON, paste text, or add URLs, then download.
- **From the CLI:** `npm run build-knowledge -- --in ./content --out ./public/knowledge.bin` (same chunker/embedder/format, so the output is identical).

Host the file anywhere with cross-origin reads (a GitHub repo via a CDN, Cloudflare R2, etc.). It is downloaded by every visitor, so treat it as **public**. Never put secrets in it.

## Browser support

aidekin needs **WebGPU**: recent desktop Chrome/Edge, Safari 26+, Firefox 145+. Unsupported browsers see a short notice instead of a crash. One-time download: **~290 MB** for text, **~1.6 GB more** for voice (the first time it is used), cached afterwards. Browsers partition cache by site, so a visitor downloads once per site, then loads from cache.

---

## Develop / self-host

```bash
npm install
npm run dev        # serves the site + /widget/ with the required COOP/COEP headers
npm run build      # typecheck + app build + loader build → dist/
```

Open the printed URL in a WebGPU-capable browser. Model weights stream from CDNs on first use (Hugging Face for the models, jsDelivr for the onnxruntime-web runtime + VAD), then cache to OPFS / Cache Storage. The repo ships **no weights**; a production build is just the app bundle (tens of MB).

**Cross-origin isolation.** WASM threads / `SharedArrayBuffer` (used by voice) require the page to be cross-origin isolated. Vite dev/preview send `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` (see `vite.config.ts`); send the same two headers in production for full-speed voice. Text is unaffected.

**Self-host the widget:** deploy `dist/`, then point the loader at your copy with `data-widget-origin`. Mirror the model weights and set `VITE_MODEL_CDN=https://your-bucket.example` at build time to serve them yourself; `VITE_MODEL_CDN=/models` + `npm run fetch-models` serves a local mirror for offline dev.

**Swap a model:** everything lives behind one entry in [`src/models/registry.ts`](src/models/registry.ts) (the single source of truth) plus a small worker interface. Change `LLM.hfModelId` / `ASR.hfModelId` / `TTS` to swap.

### Models

| Role | Model | Runtime |
|---|---|---|
| LLM | `onnx-community/Bonsai-1.7B-ONNX` (`q1`) | transformers.js (WebGPU) |
| ASR | `soniqo/Nemotron-3.5-ASR-Streaming-Multilingual-0.6B-ONNX-FP16` | onnxruntime-web (WebGPU + WASM) |
| TTS | `Supertone/supertonic-3` | onnxruntime-web (WebGPU/WASM) |
| Turn detect | `onnx-community/smart-turn-v3-ONNX` | transformers.js (WASM) |
| VAD | Silero v5 (via `@ricky0123/vad-web`) | onnxruntime-web (WASM) |
| Embedder (RAG) | `Xenova/bge-small-en-v1.5` (`q8`, 384-dim) | transformers.js (WASM) |

### Project layout

```
src/
  site/        Landing · Configure · Builder · Docs · Demo · Privacy · Terms · Layout · SiteWidget · icons
  widget/      WidgetApp · WidgetFrame · useTextController · SonarPing · Markdown · protocol · main
  embed/       loader            (the ~2 KB script + iframe launcher)
  engine/      conversationEngine (shared brain: LLM turn + history + RAG)
  rag/         chunker · embedder · store · retriever
  pipeline/    orchestrator · sentenceChunker
  workers/     vad · asr · llm · tts · turn   (each = one model in its own Worker)
  asr/ tts/    streaming ASR core · Supertonic TTS
  audio/       micCapture · pcmWorklet · resampler · autoGain · playbackQueue
  core/        capabilities · storage · modelStore · diagnostics
  models/      registry (single source of truth)
  protocol/    messages (typed worker protocol)
content/       knowledge sources for the demo assistant
scripts/       fetch-models · build-knowledge (PDF/Word extraction) · verify-knowledge
```

## Privacy & security

Everything runs on the visitor's device, so there is nothing on your side to steal or run up. The system prompt and the knowledge file are delivered to the browser, so treat them as **public** (never put secrets in them). The widget is a sandboxed iframe, requests the mic only for voice, and only the embedding page (plus any `data-allowed-origins`) can drive it. Because it runs locally, a determined visitor can inspect the prompt and model on their own machine, so do not treat the prompt as a security boundary.

## License

MIT © Sait Furkan Teke. See [LICENSE](LICENSE).
