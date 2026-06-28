# @aidekin/webgpu-llm

A fast, dependency-free WebGPU runtime for **1-bit (binary-weight) LLMs** in the browser.

Built for Bonsai-1.7B (Qwen3 architecture, sign-packed binary linear weights + 2/4-bit tied
embeddings). Bit-exact with the reference forward, GPU-resident greedy decode, and a tiled GEMM
prefill path. Runs the fast subgroup path on Apple / NVIDIA / recent AMD and falls back to a
workgroup-reduction path everywhere else WebGPU is available.

> Status: early. The decode path is greedy (argmax). Sampling, streaming, EOS stop and
> `AbortSignal` are landing next; the `GenerateOptions` shape is already in place for them.

## Install

```sh
npm install @aidekin/webgpu-llm
```

ESM-only, zero runtime dependencies.

## Usage

```ts
import { createEngine, WebGPUUnavailableError } from '@aidekin/webgpu-llm'

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

const result = await engine.generate(promptTokenIds, { maxTokens: 128 })
console.log(result.tokens, result.tokensPerSecond)

engine.dispose()
```

Tokenization is intentionally out of scope: the engine operates on token ids, so you can pair it
with any tokenizer.

## API

- `createEngine(options: EngineOptions | string): Promise<Engine>` - load a model. A bare string is
  treated as `modelUrl`.
- `engine.generate(promptTokenIds, options?)` - generate tokens.
- `engine.forward(tokenIds)` - single forward pass (hidden states + logits) for correctness checks.
- `engine.capabilities` - detected GPU path (`useSubgroups`, `subgroupSize`, adapter info, limits).
- `engine.dispose()` - release GPU resources.

See `src/types.ts` for the full option and result shapes.

## Development

```sh
npm run gen:shaders   # inline shaders/*.wgsl -> src/shaders.generated.ts
npm run build         # tsdown -> dist (ESM + .d.ts)
npm run typecheck
npm run check:publish # publint + are-the-types-wrong
```

The WGSL kernels live in `shaders/` and are inlined into the bundle at build time (no runtime
`fetch`). `scripts/gen-shaders.ts` does the inlining.

## License

MIT
