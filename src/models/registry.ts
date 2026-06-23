// ─────────────────────────────────────────────────────────────────────────────
// Aidekin model + runtime registry — the SINGLE SOURCE OF TRUTH.
//
// Every repo ID and version below was VERIFIED against npm + the Hugging Face Hub
// on 2026-06-17 (see README → "Resolved versions"). To swap a model, edit one
// entry here; each model stays behind its interface in /pipeline/interfaces.ts so
// the swap is a one-liner everywhere else.
// ─────────────────────────────────────────────────────────────────────────────

export type Runtime =
  | 'transformers.js'
  | 'onnxruntime-web'
  | 'vad-web'

/** Pinned runtime versions (npm). */
export const RUNTIME_VERSIONS = {
  vite: '8.0.16',
  typescript: '6.0.3',
  transformers: '4.2.0', //  @huggingface/transformers (Transformers.js v4 — confirmed real)
  onnxruntimeWeb: '1.26.0',
  vadWeb: '0.0.30', //          @ricky0123/vad-web (bundles Silero VAD v4 default / v5 opt-in)
  fftJs: '4.0.3',
} as const

// ── LLM (the "brain"): PrismML Bonsai via transformers.js ─────────────────────
// Bonsai is a Qwen3-architecture model compressed to ternary weights, exported to
// ONNX by onnx-community and run on WebGPU by @huggingface/transformers (verified
// API; the live demo webml-community/bonsai-webgpu works). It keeps the Qwen3 chat
// template + <think> behaviour, so the worker's prompt handling is unchanged. The
// `q1` ONNX is actually 2-bit MatMulNBits (ternary packed) — a real ~290 MB download.
//
// CAVEAT (verified): ort-web has no 2-bit MatMulNBits WebGPU kernel on Apple GPUs, so
// the weights dequantize to fp16 in VRAM — a 4B → ~8 GB, which OOMs (std::bad_alloc)
// alongside the resident ASR+TTS. The 1.7B (~3.4 GB dequantized) is the largest Bonsai
// that fits with the speech stack, and is the proven demo size. Measure tokens/sec in
// the console.
export const LLM = {
  runtime: 'transformers.js' satisfies Runtime,
  hfModelId: 'onnx-community/Bonsai-1.7B-ONNX', // ↔ '…Bonsai-4B-ONNX' only if VRAM allows
  dtype: 'q1', //                                  matches the proven webml-community/bonsai-webgpu demo
  device: 'webgpu' as const,
  eosTokenId: 151645, //                           <|im_end|>
} as const

// ── ASR (Nemotron 3.5 streaming, FP16 → WebGPU — the ONE AND ONLY engine) ─────
// One model, one path. The heavy FastConformer-RNNT encoder runs FP16 on WebGPU
// (RTF≪1 → real-time streaming); decoder/joint run on WASM. We drive the three ONNX
// sessions ourselves (no onnxruntime-genai, no sherpa-onnx): log-mel features →
// cache-aware streaming encoder → greedy RNNT decode (see asr/soniqoAsr.ts).
//
// WebGPU is REQUIRED for ASR — as it already is for the LLM. The former int4/WASM CPU
// export was REMOVED: in the browser it ran ~20× slower than real-time (90s+/turn), so
// it could never be the live path; it only added a dual-engine code path for a tiny
// clean-audio WER edge that mic quality dwarfs anyway.
//
// Contract VERIFIED (asr-fp16-test) by inspecting the ONNX graph I/O + config.json:
//   encoder: audio_signal[1,128,32] mel-major · audio_length(i32) · language_mask
//            (one-hot[1,128]) · pre_cache[1,128,9] · cache_last_channel[24,1,56,1024]
//            · cache_last_time[24,1,1024,8] · cache_last_channel_len(i32)
//            → encoded_output[1,4,1024] · encoded_length(i32) · new_* caches
//   decoder: token(i64) · h · c → decoder_output[1,1,640] · h_out · c_out
//   joint:   encoder_output[1,1,1024] · decoder_output[1,1,640] → logits[1,1,13088]
// Streams 320 ms chunks. ~1.25 GB, streamed from the HF CDN and OPFS-cached on first
// use. English-only here (the model is multilingual; we always request English).
export const ASR = {
  runtime: 'onnxruntime-web' satisfies Runtime,
  hfModelId: 'soniqo/Nemotron-3.5-ASR-Streaming-Multilingual-0.6B-ONNX-FP16',
  files: {
    encoder: 'encoder.onnx',
    encoderData: 'encoder.onnx.data',
    decoder: 'decoder.onnx',
    decoderData: 'decoder.onnx.data',
    joiner: 'joint.onnx',
    joinerData: 'joint.onnx.data',
    config: 'config.json',
    vocab: 'vocab.json',
  },
  contract: {
    sampleRate: 16000,
    numMels: 128,
    nFft: 512,
    hopLength: 160,
    winLength: 400,
    preemph: 0.97,
    logEps: 5.96046448e-8,
    hidden: 1024,
    encoderLayers: 24,
    leftContext: 56,
    convContext: 8,
    preCacheFrames: 9,
    decoderHidden: 640,
    decoderLayers: 2,
    subsamplingFactor: 8,
    chunkSamples: 5120, //          320 ms @ 16 kHz
    melFramesPerChunk: 32, //       5120 / hop(160)
    blankId: 13087,
    vocabSize: 13088,
    numPrompts: 128,
    maxSymbolsPerStep: 10,
  },
  // `language_mask` one-hot index — canonical prompt_dictionary (the repo's languages.json).
  // English-only: index 0. (The model supports 100+ via the prompt dictionary.)
  langId: { en: 0 } as Record<string, number>,
} as const

// ── Turn detection (Smart Turn v3 — confirmed real) ──────────────────────────
export const TURN = {
  runtime: 'transformers.js' satisfies Runtime,
  hfModelId: 'onnx-community/smart-turn-v3-ONNX',
  rawModelId: 'pipecat-ai/smart-turn-v3', // raw .onnx (smart-turn-v3.2-cpu.onnx ~8MB int8) for the onnxruntime-web path
  // Whisper-Tiny encoder + linear head, ~8M params, semantic turn detection on raw
  // 16kHz waveform. CPU ~12ms.
  tailSeconds: 8, // run on the completed-turn audio, truncated to the last ~8s
} as const

// ── VAD (Silero, bundled in @ricky0123/vad-web) ──────────────────────────────
// vad.worker.ts loads Silero v5 directly (SileroV5 + silero_vad_v5.onnx) with
// 512-sample @16 kHz frames — matching the mic worklet's 512-sample frame size.
// (NOTE: @ricky0123/vad-web's own MicVAD defaults to 'legacy'/v4; we deliberately
// pin v5 for the smaller frame = lower detection latency.)
export const VAD = {
  runtime: 'vad-web' satisfies Runtime,
  sileroModel: 'v5' as const,
  frameSamples: 512,
} as const

// ── TTS (Supertonic-3 — confirmed real) ──────────────────────────────────────
// ~99M params, multilingual; we synthesize English (<en> wrapper). Official browser
// example uses onnxruntime-web + fft.js.
export const TTS = {
  runtime: 'onnxruntime-web' satisfies Runtime,
  hfModelId: 'Supertone/supertonic-3',
  // Files under /public/models/tts (HF repo Supertone/supertonic-3):
  files: {
    durationPredictor: 'onnx/duration_predictor.onnx', // ~3.7MB
    textEncoder: 'onnx/text_encoder.onnx', //              ~36.4MB
    vectorEstimator: 'onnx/vector_estimator.onnx', //      ~257MB
    vocoder: 'onnx/vocoder.onnx', //                       ~101MB
    config: 'onnx/tts.json',
    unicodeIndexer: 'onnx/unicode_indexer.json',
    modelConfig: 'config.json',
    voiceStyle: 'voice_styles/F1.json', // speaker embedding (F1..F5 / M1..M5 available)
  },
} as const

// ── Embeddings (local RAG): bge-small-en-v1.5 via transformers.js ─────────────
// 384-dim, q8 ONNX ≈34 MB (single file). Chosen over all-MiniLM-L6-v2 after a 2026
// review: same 384 dims (drop-in for the int8 store) but a large retrieval-quality jump
// (~42 → ~52 nDCG@10 on MTEB/BEIR), which is exactly what matters when grounding a small
// 1.7B LLM (a wrong top-1 chunk poisons the answer). Runs on the WASM (CPU) backend in
// the browser so it never competes with the LLM for the GPU, and identically in Node
// (onnxruntime-node) for the build CLI — the SAME model + dtype + pooling in both is what
// keeps query vectors compatible with the precomputed index (parity). q8 is safe on both
// runtimes (q4f16 has an onnxruntime-node fusion bug); int8 is applied only to the stored
// corpus vectors. bge-v1.5 needs NO query prefix; 'mean' pooling A/B-beat 'cls' here.
export const EMBED = {
  runtime: 'transformers.js' satisfies Runtime,
  hfModelId: 'Xenova/bge-small-en-v1.5',
  dim: 384,
  maxSeqTokens: 512,
  pooling: 'mean',
  normalize: true,
  dtype: 'q8',
} as const

/** Where self-hosted assets live, served same-origin (so they satisfy COEP: require-corp). */
export const ASSET_PATHS = {
  models: '/models', //         public/models/<role>/...
  vadAssets: '/models/vad', //  self-hosted @ricky0123/vad-web dist (.onnx + worklet)
} as const

const HF_RESOLVE = (repo: string): string => `https://huggingface.co/${repo}/resolve/main`

/** onnxruntime-web wasm runtime, served from jsDelivr (CORS + CORP clean → COEP-ok). */
export const ORT_WASM_CDN = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${RUNTIME_VERSIONS.onnxruntimeWeb}/dist/`

/**
 * Base URL for a model role's weights. NOTHING is self-hosted by default — dev and
 * prod behave identically (everything streams from a CDN, then caches to OPFS for
 * offline use). Keeps the deploy lean (no ~1.2 GB shipped) and dev clean + testable:
 *   • asr / tts → Hugging Face CDN (verified CORS-clean → passes COEP require-corp)
 *   • vad       → jsDelivr (the Silero model from @ricky0123/vad-web)
 *   • override  → set `VITE_MODEL_CDN` to your own bucket / a local `/models` mirror
 *                 (e.g. `VITE_MODEL_CDN=/models npm run dev` after `npm run fetch-models`)
 */
export function modelSource(role: 'asr' | 'tts' | 'vad'): string {
  const cdn = (import.meta.env.VITE_MODEL_CDN as string | undefined)?.replace(/\/$/, '')
  if (cdn) return `${cdn}/${role}`
  if (role === 'asr') return HF_RESOLVE(ASR.hfModelId)
  if (role === 'tts') return HF_RESOLVE(TTS.hfModelId)
  return `https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@${RUNTIME_VERSIONS.vadWeb}/dist`
}
