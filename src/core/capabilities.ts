// Local capability probe — decides which models can run and how to degrade.
// Everything here is detected locally; nothing is fetched over the network.

export interface WebGpuInfo {
  readonly supported: boolean
  readonly reason?: string
  readonly vendor?: string
  readonly architecture?: string
  readonly maxBufferSizeMB?: number
  readonly maxStorageBufferBindingSizeMB?: number
}

export interface CapabilityReport {
  readonly crossOriginIsolated: boolean
  readonly sharedArrayBuffer: boolean
  readonly wasmThreads: boolean
  readonly wasmSimd: boolean
  readonly audioWorklet: boolean
  readonly opfs: boolean
  readonly webgpu: WebGpuInfo
  readonly hardwareConcurrency: number
  readonly deviceMemoryGB: number | null
  /** Phone/tablet (UA-CH mobile flag, mobile UA, or coarse-pointer touch device). */
  readonly isMobile: boolean
}

// Canonical WASM SIMD feature-detect module (GoogleChromeLabs/wasm-feature-detect).
// A 30-byte module whose body uses an i8x16.splat → i32x4 lane op; validates only
// when SIMD is supported.
const SIMD_TEST = new Uint8Array([
  0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8,
  0, 65, 0, 253, 15, 253, 98, 11,
])

function detectWasmSimd(): boolean {
  try {
    return WebAssembly.validate(SIMD_TEST)
  } catch {
    return false
  }
}

async function detectWebGpu(): Promise<WebGpuInfo> {
  const gpu = (navigator as Navigator & { gpu?: GPU }).gpu
  if (!gpu) {
    return { supported: false, reason: 'navigator.gpu is undefined (WebGPU unavailable in this browser)' }
  }
  try {
    const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' })
    if (!adapter) {
      return { supported: false, reason: 'No GPUAdapter returned (hardware/driver blocklisted WebGPU)' }
    }
    const info = adapter.info
    const limits = adapter.limits
    const toMB = (bytes: number | bigint): number => Math.round(Number(bytes) / (1024 * 1024))
    return {
      supported: true,
      vendor: info?.vendor || undefined,
      architecture: info?.architecture || undefined,
      maxBufferSizeMB: toMB(limits.maxBufferSize),
      maxStorageBufferBindingSizeMB: toMB(limits.maxStorageBufferBindingSize),
    }
  } catch (err) {
    return { supported: false, reason: `requestAdapter() threw: ${(err as Error).message}` }
  }
}

export async function probeCapabilities(): Promise<CapabilityReport> {
  const coi = typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated === true
  const sab = typeof SharedArrayBuffer !== 'undefined'
  const opfs =
    typeof navigator !== 'undefined' &&
    'storage' in navigator &&
    typeof navigator.storage?.getDirectory === 'function'

  const ua = typeof navigator !== 'undefined' ? navigator.userAgent || '' : ''
  const coarseTouch =
    typeof navigator !== 'undefined' &&
    navigator.maxTouchPoints > 1 &&
    typeof matchMedia === 'function' &&
    matchMedia('(pointer: coarse)').matches
  const isMobile =
    (navigator as Navigator & { userAgentData?: { mobile?: boolean } }).userAgentData?.mobile === true ||
    /iPhone|iPad|iPod|Android/i.test(ua) ||
    coarseTouch

  return {
    crossOriginIsolated: coi,
    sharedArrayBuffer: sab,
    wasmThreads: coi && sab, // real WASM threads need a shared WebAssembly.Memory → needs SAB + COI
    wasmSimd: detectWasmSimd(),
    audioWorklet: typeof AudioWorkletNode !== 'undefined',
    opfs,
    webgpu: await detectWebGpu(),
    hardwareConcurrency: navigator.hardwareConcurrency || 1,
    deviceMemoryGB: (navigator as Navigator & { deviceMemory?: number }).deviceMemory ?? null,
    isMobile,
  }
}

export interface DegradationPlan {
  /** The on-device LLM (transformers.js) needs WebGPU; without it it cannot run. */
  readonly canRunLlm: boolean
  /** Streaming ASR (onnxruntime-web) wants WASM threads for usable latency. */
  readonly asrThreaded: boolean
  readonly warnings: readonly string[]
}

export function planDegradation(r: CapabilityReport): DegradationPlan {
  const warnings: string[] = []
  if (!r.webgpu.supported) {
    warnings.push(
      'WebGPU unavailable — the on-device LLM cannot run here. ASR/VAD/TTS may still work on WASM/CPU.',
    )
  }
  if (!r.crossOriginIsolated) {
    warnings.push(
      'Page is NOT cross-origin isolated — SharedArrayBuffer/WASM threads are disabled. ASR will be slow or fail. Check the COOP/COEP headers.',
    )
  }
  if (!r.wasmSimd) {
    warnings.push('WASM SIMD unavailable — speech models (ASR/turn/TTS) will be significantly slower.')
  }
  if (!r.opfs) {
    warnings.push('OPFS unavailable — falling back to IndexedDB/Cache Storage for model weights.')
  }
  if (r.webgpu.supported && r.webgpu.maxBufferSizeMB && r.webgpu.maxBufferSizeMB < 1024) {
    warnings.push(
      `Low GPU maxBufferSize (${r.webgpu.maxBufferSizeMB}MB) — the LLM may run slowly or fail to allocate.`,
    )
  }
  return {
    canRunLlm: r.webgpu.supported,
    asrThreaded: r.wasmThreads,
    warnings,
  }
}

// ── Widget mode resolution ───────────────────────────────────────────────────
// Text needs only WebGPU (the LLM); the embedder runs single-threaded WASM. Voice
// additionally needs threaded WASM (VAD + ASR decoder) → cross-origin isolation.
export type WidgetMode = 'text' | 'voice' | 'both'

export interface WidgetCapabilities {
  /** Text chat can run here (WebGPU present). */
  readonly textAvailable: boolean
  /** Full-speed voice can run here (WebGPU + cross-origin isolation + SAB threads). */
  readonly voiceAvailable: boolean
  /** What the widget should actually do, given the request + what's supported. */
  readonly effectiveMode: WidgetMode | 'unsupported'
  /** Human-readable note when something requested isn't available. */
  readonly reason?: string
  /**
   * WebGPU is present, but the device (phone/tablet, low memory, or low GPU limits) probably
   * can't actually load a ~1.7B model. The widget warns before the heavy download yet still
   * lets the visitor try, since the check is a heuristic, not a guarantee.
   */
  readonly constrained: boolean
  readonly constrainedReason?: string
}

/** Decide what the widget can run, given device capabilities + the owner's request. */
export function resolveWidgetCapabilities(r: CapabilityReport, requested: WidgetMode): WidgetCapabilities {
  const textAvailable = r.webgpu.supported
  const voiceAvailable = textAvailable && r.crossOriginIsolated && r.sharedArrayBuffer && r.wasmThreads

  if (!textAvailable) {
    return {
      textAvailable: false,
      voiceAvailable: false,
      effectiveMode: 'unsupported',
      reason: r.webgpu.reason ?? 'WebGPU is required to run the assistant on this device.',
      constrained: false,
    }
  }

  // WebGPU is present, but a phone/tablet or a low-memory device usually can't actually load a
  // ~1.7B model: it OOMs, and on iOS the tab is killed before any error fires. Flag it so the
  // widget warns before the heavy download, while still allowing "try anyway".
  // deviceMemory is only exposed by Chromium and caps at 8, so <= 4 means a genuinely small
  // machine; Safari/Firefox report null and are not flagged here (a desktop Mac should still
  // try). We deliberately do NOT use the WebGPU buffer limits: they default to the spec minimum
  // on many capable desktops, which would false-flag them. Mobile is the reliable signal.
  const lowMemory = r.deviceMemoryGB != null && r.deviceMemoryGB <= 4
  const constrained = r.isMobile || lowMemory
  const constrainedReason = constrained
    ? 'aidekin runs the AI model on your device, which needs a desktop browser with enough memory. Phones and tablets usually cannot load it.'
    : undefined

  if (requested === 'text') {
    return { textAvailable, voiceAvailable, effectiveMode: 'text', constrained, constrainedReason }
  }
  if (requested === 'voice') {
    return voiceAvailable
      ? { textAvailable, voiceAvailable, effectiveMode: 'voice', constrained, constrainedReason }
      : {
          textAvailable,
          voiceAvailable,
          effectiveMode: 'text',
          reason:
            'Voice needs the host page to enable cross-origin isolation (COOP/COEP). Falling back to text chat.',
          constrained,
          constrainedReason,
        }
  }
  // 'both' — text always works; voice is offered only where supported.
  return {
    textAvailable,
    voiceAvailable,
    effectiveMode: 'both',
    reason: voiceAvailable
      ? undefined
      : 'Voice needs cross-origin isolation on the host page; only text chat is available here.',
    constrained,
    constrainedReason,
  }
}
