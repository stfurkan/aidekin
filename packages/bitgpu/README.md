# bitgpu

A fast, dependency-free WebGPU runtime for **low-bit LLMs** in the browser.

Today it runs **1-bit (binary-weight)** models.
Reference target is Bonsai-1.7B (Qwen3 architecture, sign-packed binary linear weights + 2/4-bit tied
embeddings). Bit-exact with the reference forward, GPU-resident decode (greedy or sampled), streaming,
EOS stop, `AbortSignal`, and cross-turn KV-cache reuse. Runs the fast subgroup path on Apple / NVIDIA /
recent AMD and falls back to a workgroup-reduction path everywhere else WebGPU is available.

## Install

```sh
npm install bitgpu
```

ESM-only, zero runtime dependencies.

## Usage

```ts
import { createEngine, WebGPUUnavailableError } from 'bitgpu'

let engine
try {
  engine = await createEngine({
    modelUrl: 'https://cdn.example.com/bonsai', // dir with manifest.json + data/aux files
    onProgress: (p) => console.log(p.phase),
  })
} catch (err) {
  if (err instanceof WebGPUUnavailableError) {
    // render a "WebGPU not supported" fallback
  } else throw err
}

// Greedy by default; stream tokens, stop on EOS, cancel with a signal.
const result = await engine.generate(promptTokenIds, {
  maxTokens: 256,
  stopTokens: [151645],
  onToken: (id) => process.stdout.write(String(id) + ' '),
})
console.log(result.tokens, result.tokensPerSecond)

// Sampling (matches transformers.js v4.2.0 exactly): set a temperature other than 0/1.
await engine.generate(promptTokenIds, { temperature: 0.5, topK: 20, repetitionPenalty: 1.15 })

engine.dispose()
```

Tokenization is intentionally out of scope: the engine operates on token ids, so you can pair it
with any tokenizer.

## API

- `createEngine(options: EngineOptions | string): Promise<Engine>` - load a model. A bare string is
  treated as `modelUrl`.
- `engine.generate(promptTokenIds, options?)` - generate tokens. Greedy by default; sampling, streaming
  (`onToken`), EOS (`stopTokens`), cancellation (`signal`) and cross-turn cache reuse (`reuseCache`) are
  all supported. `maxTokens` is clamped to the KV window. See the published `EngineOptions` /
  `GenerateOptions` types for the full option shapes.
- `engine.prefill(promptTokenIds)` - prefill a prompt prefix into the KV cache without decoding, so a
  later `generate(delta, { reuseCache: true })` starts from a warm cache (e.g. a static system prompt).
- `engine.forward(tokenIds)` - single forward pass (hidden states + logits) for correctness checks.
- `engine.resetCache()` - clear the cross-turn KV cache (start a fresh conversation).
- `engine.capabilities` - detected GPU path (`useSubgroups`, `subgroupSize`, adapter info, limits).
- `engine.lost` - promise that resolves if the GPU device is lost (also via `onDeviceLost` option);
  create a new engine to recover.
- `engine.dispose()` - release GPU resources.

Errors: `WebGPUUnavailableError` (no WebGPU / no adapter) and `GpuOutOfMemoryError` (weight upload or
KV growth failed) are exported so you can branch on them.

## Development

```sh
npm run gen:shaders   # inline shaders/*.wgsl -> src/shaders.generated.ts
npm run build         # tsdown -> dist (ESM + .d.ts)
npm run typecheck
npm run test:sampler  # sampler parity vs transformers.js v4.2.0
npm run check:publish # publint + are-the-types-wrong
```

The WGSL kernels live in `shaders/` and are inlined into the bundle at build time (no runtime
`fetch`). `scripts/gen-shaders.ts` does the inlining.

## License

MIT
