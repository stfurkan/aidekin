// Public types for @aidekin/webgpu-llm.

/** Progress event emitted while a model loads. */
export interface LoadProgress {
  phase: 'manifest' | 'weights' | 'pipelines'
  /** Bytes fetched so far (weights phase only). */
  loaded?: number
  /** Total bytes to fetch (weights phase only). */
  total?: number
}

/** Options for {@link createEngine}. Pass a string to use defaults with just a model URL. */
export interface EngineOptions {
  /** Base URL of the model directory containing `manifest.json` and its data/aux files. */
  modelUrl: string
  /** GPU power preference. Default `'high-performance'` (picks the discrete GPU on multi-GPU machines). */
  powerPreference?: GPUPowerPreference
  /** Force the no-subgroup reduction path (for testing the fallback). Default `false`. */
  forceNoSubgroups?: boolean
  /** Workgroup size for the no-subgroup reduction kernels. Default `64`. */
  noSubgroupWorkgroupSize?: number
  /** Decode steps chained per CPU sync (deferred readback). Higher hides latency; default `4`. */
  syncSteps?: number
  /** Prefill GEMM tiling: `'auto'` tiles once a prompt fills the 64-row tiles, `'always'`/`'never'` force it. Default `'auto'`. */
  prefillTiling?: 'auto' | 'always' | 'never'
  /** Called as the model loads. */
  onProgress?: (progress: LoadProgress) => void
}

/** Options for a single {@link Engine.generate} call. Sampling fields are reserved for a
 *  later release; the current build decodes greedily (argmax). */
export interface GenerateOptions {
  /** Maximum number of new tokens to generate. Default `256`. */
  maxTokens?: number
  // --- the following are accepted but not yet honored (greedy decode only) ---
  /** Token ids that end generation when produced (e.g. EOS). */
  stopTokens?: number[]
  /** Called with each generated token id as it is read back. */
  onToken?: (tokenId: number) => void
  /** Aborts generation when signaled. */
  signal?: AbortSignal
  /** Softmax temperature. */
  temperature?: number
  /** Top-k sampling cutoff. */
  topK?: number
  /** Top-p (nucleus) sampling cutoff. */
  topP?: number
  /** Repetition penalty applied to already-generated tokens. */
  repetitionPenalty?: number
  /** Block any repeated n-gram of this size. */
  noRepeatNgramSize?: number
  /** Seed for the sampler RNG. */
  seed?: number
}

/** Result of a {@link Engine.generate} call. */
export interface GenerateResult {
  /** Generated token ids (excludes the prompt). */
  tokens: number[]
  /** Time to first token (prefill of the prompt), in milliseconds. */
  prefillMs: number
  /** Decode time for the remaining tokens, in milliseconds. */
  decodeMs: number
  /** Decode throughput (tokens / second), excluding prefill. */
  tokensPerSecond: number
  /** Per-token decode timing breakdown, in milliseconds. */
  timing: { recordMs: number; gpuMs: number; readbackMs: number }
}

/** Diagnostic result of {@link Engine.forward}: hidden states + logits for a single forward pass. */
export interface ForwardResult {
  embed: Float32Array
  layer0: Float32Array
  finalnorm: Float32Array
  logits: Float32Array
  vocab: number
  sequenceLength: number
}

/** What the engine detected about the host GPU and which code path it selected. */
export interface EngineCapabilities {
  /** Whether the fast subgroup path is in use (false = the workgroup-reduction fallback). */
  useSubgroups: boolean
  /** Subgroup width when the subgroup path is active. */
  subgroupSize: number
  /** Adapter identification, when the browser exposes it. */
  adapter: { vendor?: string; architecture?: string; device?: string; description?: string }
  /** Relevant adapter limits the engine codes against. */
  limits: { maxStorageBufferBindingSize: number; maxComputeWorkgroupStorageSize: number }
}

/** A loaded model ready to generate. Create one with {@link createEngine}. */
export interface Engine {
  /** Generate tokens from a prompt given as token ids. */
  generate(promptTokenIds: number[], options?: GenerateOptions): Promise<GenerateResult>
  /** Run a single forward pass and return hidden states + logits (diagnostic / correctness checks). */
  forward(tokenIds: number[]): Promise<ForwardResult>
  /** Detected GPU capabilities and selected code path. */
  readonly capabilities: EngineCapabilities
  /** Release GPU resources. The engine is unusable afterward. */
  dispose(): void
}
