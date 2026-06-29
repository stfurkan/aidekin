// Bonsai-1.7B-class WebGPU runtime. Loads binary (1-bit) Qwen3 weights via the manifest,
// runs the forward with the validated kernels, keeps a persistent KV cache, and generates
// autoregressively. Decode is dispatch-overhead-bound, so the matmuls are fused (q/k/v in one
// dispatch, gate/up in one, the residual add folded into o_proj/down_proj) and the decode loop
// is GPU-resident (GPU argmax + embedding gather) with deferred CPU sync + pooled resources.
//
// This is a faithful port of the validated engine.js: the kernel sequence and numerics are
// unchanged (bit-exact). Only the shader source (now inlined, not fetched), the configuration
// (now typed options, not URL params), and the public surface differ.
import { SHADERS } from './shaders.generated'
import { WebGPUUnavailableError } from './errors'
import { MT19937, affectedIds, ngramBans, sampleFromCandidates } from './sampler'
import type {
  Engine,
  EngineCapabilities,
  EngineOptions,
  ForwardResult,
  GenerateOptions,
  GenerateResult,
} from './types'

type Field = ['f' | 'u', number]

interface Ref {
  src?: string
  dtype: string
  off: number
  len: number
}
interface MTensor {
  kind?: string
  N?: number
  K?: number
  block?: number
  weight?: Ref
  scales?: Ref
  zp?: Ref
  // cos_cache / sin_cache are stored as bare refs:
  src?: string
  dtype?: string
  off?: number
  len?: number
}
interface Arch {
  layers: number
  hidden: number
  intermediate: number
  heads: number
  kv_heads: number
  head_dim: number
  rms_eps: number
  vocab: number
  eos: number
  act: string
}
interface Manifest {
  data_file: string
  aux_file: string
  arch: Arch
  luts: Record<string, Ref>
  tensors: Record<string, MTensor>
}

interface GpuWeight {
  buf?: GPUBuffer
  sign?: GPUBuffer
  scales?: GPUBuffer
  codes?: GPUBuffer
  N?: number
  K?: number
  nb?: number
  N0?: number
  N1?: number
  N2?: number
  zp?: number
}

interface RawGenResult {
  prefillMs: number
  decodeMs: number
  tokPerSec: number
  tokens: number[]
  firstArgmax: number
  recMs: number
  gpuMs: number
  rbMs: number
}

/** Internal engine handle: the public {@link Engine} surface plus diagnostics used by the
 *  correctness/benchmark harness. The diagnostics are intentionally not in the public type. */
interface EngineInternal extends Engine {
  device: GPUDevice
  adapter: GPUAdapter
  /** Raw decode with the per-kernel profiling switch (`full`) and sync depth exposed. */
  profileDecode(ids: number[], nTokens: number, full?: Set<string> | null, syncN?: number): Promise<RawGenResult>
  /** Differential debug: one decode step through the fast and slow paths, checkpoint by checkpoint. */
  debugDecode(prefillIds: number[]): Promise<{ fast: Record<string, Float32Array>; slow: Record<string, Float32Array> }>
  /** Debug: GPU base + penalized logits + top-K for a prefill, to diff the sampler kernels vs CPU math. */
  debugSampler(ids: number[], genOpts: GenerateOptions): Promise<{ base: Float32Array; penalized: Float32Array; candIds: Uint32Array; candVals: Float32Array }>
}

type TypedArrayCtor = Float32ArrayConstructor | Uint8ArrayConstructor | Uint16ArrayConstructor
const VIEW: Record<string, TypedArrayCtor> = { FLOAT: Float32Array, UINT8: Uint8Array, FLOAT16: Uint16Array }
const WGSLS = ['matmul_binary_vec4', 'matmul_split', 'matmul_resid', 'matmul_q2', 'rmsnorm', 'rope', 'swiglu', 'attention_cache', 'add', 'copy']
const MAXSEQ = 256

const PARAM_AB = new ArrayBuffer(64)
const PARAM_DV = new DataView(PARAM_AB)
const PARAM_U8 = new Uint8Array(PARAM_AB)
function makeParams(fields: Field[]): Uint8Array {
  // fills a reused buffer (no per-dispatch alloc)
  for (let i = 0; i < fields.length; i++) {
    const f = fields[i]
    if (f[0] === 'f') PARAM_DV.setFloat32(i * 4, f[1], true)
    else PARAM_DV.setUint32(i * 4, f[1] >>> 0, true)
  }
  return PARAM_U8.subarray(0, Math.ceil(fields.length / 4) * 16)
}
const eqBytes = (a: Uint8Array, b: Uint8Array): boolean => {
  for (let i = 0; i < b.length; i++) if (a[i] !== b[i]) return false
  return true
}
function concat<A extends Uint8Array | Float32Array>(Cls: { new (n: number): A }, arrs: A[]): A {
  let n = 0
  for (const a of arrs) n += a.length
  const o = new Cls(n)
  let p = 0
  for (const a of arrs) {
    o.set(a as unknown as ArrayLike<number>, p)
    p += a.length
  }
  return o
}

/** Load a 1-bit model and return an {@link Engine}. Pass a model URL string for defaults. */
export async function createEngine(options: EngineOptions | string): Promise<Engine> {
  const opts: EngineOptions = typeof options === 'string' ? { modelUrl: options } : options
  const modelDir = opts.modelUrl.replace(/\/$/, '')
  const powerPreference = opts.powerPreference ?? 'high-performance'

  if (typeof navigator === 'undefined' || !navigator.gpu) {
    throw new WebGPUUnavailableError('WebGPU is not available (no navigator.gpu). Use a WebGPU-capable browser over a secure context.')
  }

  opts.onProgress?.({ phase: 'manifest' })
  const manifest = (await (await fetch(`${modelDir}/manifest.json`)).json()) as Manifest
  opts.onProgress?.({ phase: 'weights' })
  const data = await (await fetch(`${modelDir}/${manifest.data_file}`)).arrayBuffer()
  const aux = await (await fetch(`${modelDir}/${manifest.aux_file}`)).arrayBuffer()
  const A = manifest.arch
  const T = manifest.tensors

  const readRef = (ref: Ref): Float32Array | Uint8Array | Uint16Array => {
    const src = ref.src === 'aux' ? aux : data
    const V = VIEW[ref.dtype]!
    if (V === Uint8Array) return new Uint8Array(src, ref.off, ref.len)
    const bpe = V.BYTES_PER_ELEMENT
    if (ref.off % bpe === 0) return new V(src, ref.off, ref.len / bpe)
    return new V(src.slice(ref.off, ref.off + ref.len))
  }
  const readU8 = (ref: Ref): Uint8Array => readRef(ref) as Uint8Array
  const readF32 = (ref: Ref): Float32Array => readRef(ref) as Float32Array

  const adapter = await navigator.gpu.requestAdapter({ powerPreference }) // pick the discrete GPU on multi-GPU machines, not the weak iGPU
  if (!adapter) throw new WebGPUUnavailableError('No suitable WebGPU adapter was found.')
  const hasSG = adapter.features.has('subgroups' as GPUFeatureName)
  const info = (adapter.info ?? {}) as GPUAdapterInfo & { subgroupMinSize?: number; subgroupMaxSize?: number } // subgroup sizes live on GPUAdapterInfo
  const sgMax = info.subgroupMaxSize ?? 32
  const sgMin = info.subgroupMinSize ?? sgMax
  const forceNoSG = opts.forceNoSubgroups ?? false
  const WG_NS = Math.min(256, opts.noSubgroupWorkgroupSize ?? 64) // no-subgroup reduction workgroup size
  const NOTILE = opts.prefillTiling === 'never' // force the scalar prefill GEMM (A/B)
  const FORCETILE = opts.prefillTiling === 'always' // use tiled even for short prompts (validation)
  const tiledPrefill = (S: number): boolean => FORCETILE || (!NOTILE && S >= 64) // tiled GEMM wins only once it fills its 64-row tiles
  const SYNC_N = Math.max(1, opts.syncSteps ?? 4) // decode: chain N steps per CPU sync
  const useSG = hasSG && sgMin === sgMax && (sgMax === 32 || sgMax === 64) && !forceNoSG // uniform >=32 -> head_dim/SG<=4
  const device = await adapter.requestDevice({ requiredFeatures: useSG ? (['subgroups'] as GPUFeatureName[]) : [] }) // no requiredLimits -> code to the guaranteed minimums (runs on low-end/mobile)
  const L = adapter.limits // awareness: largest binding (lm_head ~77MB) < 128MiB, shared mem <=8KB < 16KB min, WG <=256

  opts.onProgress?.({ phase: 'pipelines' })
  const pipelines: Record<string, GPUComputePipeline> = {}
  // async pipeline creation: compile in parallel, non-blocking -> faster, stall-free cold start (MDN-recommended)
  const mkPipe = async (name: string, constants?: Record<string, number>): Promise<void> => {
    const code = SHADERS[name]
    if (code === undefined) throw new Error(`shader not found: ${name}`)
    const module = device.createShaderModule({ code, label: name }) // label so errors name the shader
    const info = await module.getCompilationInfo()
    const err = info.messages.find((m) => m.type === 'error')
    if (err) throw new Error(`WGSL compile error in ${name} (L${err.lineNum}:${err.linePos}): ${err.message}`)
    pipelines[name] = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main', constants } })
  }
  const ROWS_MR = 4 // output rows per workgroup in the multi-row GEMV
  const specs: Array<[string, Record<string, number>?]> = [...WGSLS.map((n): [string] => [n]), ['matmul_split_tiled'], ['matmul_resid_tiled'], ['argmax'], ['embed_gather'], ['sampler_penalty'], ['argmax_masked']]
  if (useSG) {
    for (const n of ['rmsnorm_sg', 'attention_sg', 'matmul_split_sg', 'matmul_q2_sg', 'rmsnorm_rope_sg']) specs.push([n, { SG: sgMax }])
    for (const n of ['matmul_resid_mr_sg', 'matmul_swiglu_mr_sg']) specs.push([n, { SG: sgMax, ROWS: ROWS_MR }])
  } else {
    for (const n of ['matmul_split_wg', 'matmul_resid_wg', 'matmul_q2_wg']) specs.push([n, { WG: WG_NS }])
  }
  await Promise.all(specs.map(([n, c]) => mkPipe(n, c))) // parallel compile of all pipelines

  const S_ = GPUBufferUsage.STORAGE,
    CD = GPUBufferUsage.COPY_DST,
    CS = GPUBufferUsage.COPY_SRC,
    U = GPUBufferUsage.UNIFORM
  const upload = (typed: ArrayBufferView, usage: number = S_ | CD): GPUBuffer => {
    const b = device.createBuffer({ size: typed.byteLength, usage })
    device.queue.writeBuffer(b, 0, typed as BufferSource)
    return b
  }
  // Decode resource pool: in the decode loop the dispatch sequence is identical every batch, so reuse
  // the scratch + uniform buffers across batches (createBuffer is the dominant per-token record cost).
  // Counters increment per call and reset per batch, so within a batch every dispatch still gets its
  // own buffer (no aliasing of in-flight work); reuse happens only across batches (after the sync).
  let pooling = false
  const bufPool: GPUBuffer[] = []
  let bufIdx = 0
  interface DispSlot {
    uni: GPUBuffer
    bg: GPUBindGroup | null
    last: Uint8Array | null
  }
  const dispPool: DispSlot[] = []
  let dispIdx = 0
  const poolReset = (): void => {
    bufIdx = 0
    dispIdx = 0
  }
  // The cached bind groups reference this generate() call's buffers (tokBuf/lg/candIds/... are created
  // per call). A later call creates new buffers, so the cache MUST be rebuilt at each decode entry or it
  // would bind the previous call's (dead) buffers - or, across greedy<->sampled, a different pipeline's
  // auto-layout bind group (a validation error). Buffers are stable within a call, so one rebuild suffices.
  const poolInvalidate = (): void => {
    for (const s of dispPool) {
      s.bg = null
      s.last = null
    }
  }
  const actBuf = (n: number): GPUBuffer => {
    if (!pooling) return device.createBuffer({ size: n * 4, usage: S_ | CS | CD })
    let b = bufPool[bufIdx]
    if (!b || b.size !== n * 4) {
      b = device.createBuffer({ size: n * 4, usage: S_ | CS | CD })
      bufPool[bufIdx] = b
    }
    bufIdx++
    return b
  }
  const dummy = device.createBuffer({ size: 16, usage: S_ })

  const tgt2 = readU8(manifest.luts.tgt2),
    tgt4 = readU8(manifest.luts.tgt4)
  const signTable = new Uint8Array(256)
  for (let b = 0; b < 256; b++) {
    let bits = 0
    for (let j = 0; j < 8; j++) bits |= ((((tgt2[2 * b + (j >> 2)] >> (2 * (j & 3))) & 3) >> 1) & 1) << j
    signTable[b] = bits
  }
  const rawBin = (name: string): { sign: Uint8Array; scales: Float32Array; N: number; K: number; nb: number } => {
    const t = T[name]
    const wq = readU8(t.weight!)
    const sign = new Uint8Array(wq.length)
    for (let i = 0; i < wq.length; i++) sign[i] = signTable[wq[i]]
    return { sign, scales: readF32(t.scales!), N: t.N!, K: t.K!, nb: t.K! / 128 }
  }

  const W: Record<string, GpuWeight> = {}
  for (const [name, t] of Object.entries(T)) {
    if (t.kind === 'q2') {
      const wq = readU8(t.weight!),
        codes = new Uint8Array(wq.length * 2)
      for (let i = 0; i < wq.length; i++) {
        codes[2 * i] = tgt2[2 * wq[i]]
        codes[2 * i + 1] = tgt2[2 * wq[i] + 1]
      }
      W[name] = { N: t.N!, K: t.K!, nb: t.K! / 128, zp: 2, codes: upload(codes), scales: upload(readF32(t.scales!)) }
    } else if (t.kind === 'f32' && t.weight) {
      W[name] = { buf: upload(readRef(t.weight)) }
    }
  }
  // fuse per-layer matmul weights: qkv (3), gate/up (2); o_proj + down_proj stay individual (residual-folded)
  for (let li = 0; li < A.layers; li++) {
    const q = rawBin(`layers.${li}.attn.q_proj`),
      k = rawBin(`layers.${li}.attn.k_proj`),
      v = rawBin(`layers.${li}.attn.v_proj`)
    W[`layers.${li}.attn.qkv`] = {
      K: q.K,
      nb: q.nb,
      N0: q.N,
      N1: k.N,
      N2: v.N,
      sign: upload(concat(Uint8Array, [q.sign, k.sign, v.sign])),
      scales: upload(concat(Float32Array, [q.scales, k.scales, v.scales])),
    }
    const g = rawBin(`layers.${li}.mlp.gate_proj`),
      u = rawBin(`layers.${li}.mlp.up_proj`)
    W[`layers.${li}.mlp.gateup`] = {
      K: g.K,
      nb: g.nb,
      N0: g.N,
      N1: u.N,
      N2: 0,
      sign: upload(concat(Uint8Array, [g.sign, u.sign])),
      scales: upload(concat(Float32Array, [g.scales, u.scales])),
    }
    for (const nm of [`layers.${li}.attn.o_proj`, `layers.${li}.mlp.down_proj`]) {
      const r = rawBin(nm)
      W[nm] = { N: r.N, K: r.K, nb: r.nb, sign: upload(r.sign), scales: upload(r.scales) }
    }
  }

  const embWq = readU8(T.embed_tokens.weight!),
    embScales = readF32(T.embed_tokens.scales!),
    embZp = readU8(T.embed_tokens.zp!)
  const cosCache = readF32(T.cos_cache as Ref),
    sinCache = readF32(T.sin_cache as Ref)
  // GPU-resident embedding table (for the on-GPU embed gather in the async decode loop). uint8 arrays
  // are uploaded as bytes and read as u32 (byte-extracted) in embed_gather.wgsl. ~49MB VRAM.
  const embWqG = upload(embWq),
    tgt4G = upload(tgt4),
    embScalesG = upload(embScales),
    embZpG = upload(embZp)

  function embedDequant(ids: number[]): Float32Array {
    const Hh = A.hidden,
      out = new Float32Array(ids.length * Hh)
    for (let r = 0; r < ids.length; r++) {
      const id = ids[r]
      for (let i = 0; i < 256; i++)
        for (let qd = 0; qd < 4; qd++) {
          const byte = tgt4[4 * embWq[id * 256 + i] + qd],
            baseK = (i * 4 + qd) * 2
          for (let c = 0; c < 2; c++) {
            const k = baseK + c,
              code = (byte >> (4 * c)) & 15,
              blk = (k / 128) | 0
            const zp = (embZp[id * 8 + ((blk / 2) | 0)] >> (4 * (blk & 1))) & 15
            out[r * Hh + k] = (code - zp) * embScales[id * 16 + blk]
          }
        }
    }
    return out
  }
  function ropeBufs(posBase: number, S: number): { cos: GPUBuffer; sin: GPUBuffer } {
    const D = A.head_dim,
      cos = new Float32Array(S * D),
      sin = new Float32Array(S * D)
    for (let s = 0; s < S; s++)
      for (let d = 0; d < D; d++) {
        cos[s * D + d] = cosCache[(posBase + s) * 64 + (d % 64)]
        sin[s * D + d] = sinCache[(posBase + s) * 64 + (d % 64)]
      }
    const cb = actBuf(S * D),
      sb = actBuf(S * D)
    device.queue.writeBuffer(cb, 0, cos)
    device.queue.writeBuffer(sb, 0, sin)
    return { cos: cb, sin: sb }
  }

  const KV = A.kv_heads,
    Dh = A.head_dim,
    Hd = A.hidden,
    H = A.heads,
    F = A.intermediate
  const Kc: GPUBuffer[] = [],
    Vc: GPUBuffer[] = []
  for (let li = 0; li < A.layers; li++) {
    Kc.push(actBuf(MAXSEQ * KV * Dh))
    Vc.push(actBuf(MAXSEQ * KV * Dh))
  }

  async function readback(buf: GPUBuffer, n: number): Promise<Float32Array> {
    const rb = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.MAP_READ | CD })
    const enc = device.createCommandEncoder()
    enc.copyBufferToBuffer(buf, 0, rb, 0, n * 4)
    device.queue.submit([enc.finish()])
    await rb.mapAsync(GPUMapMode.READ)
    const out = new Float32Array(rb.getMappedRange().slice(0))
    rb.unmap()
    return out
  }
  async function readbackU32(buf: GPUBuffer, n: number): Promise<Uint32Array> {
    const rb = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.MAP_READ | CD })
    const enc = device.createCommandEncoder()
    enc.copyBufferToBuffer(buf, 0, rb, 0, n * 4)
    device.queue.submit([enc.finish()])
    await rb.mapAsync(GPUMapMode.READ)
    const out = new Uint32Array(rb.getMappedRange().slice(0))
    rb.unmap()
    return out
  }

  // diagnostic: FULL = null -> every kernel at real size; FULL = Set(names) -> only those at real size,
  // all others dispatched as 1 workgroup. Lets us measure each kernel type's true in-context cost.
  let FULL: Set<string> | null = null
  const isFull = (name: string): boolean => FULL === null || FULL.has(name)
  // differential debug: FORCE_SLOW routes S=1 through the prefill (known-good) path; DBG0 collects
  // layer-0 checkpoint buffers so a fused step and a slow step can be compared kernel by kernel.
  let FORCE_SLOW = false
  let DBG0: Record<string, GPUBuffer> | null = null
  const cap = (li: number, name: string, buf: GPUBuffer): void => {
    if (li === 0 && DBG0) DBG0[name] = buf
  }
  // set pipeline + bind group for a dispatch. When pooling (decode loop), the uniform buffer AND the
  // bind group are cached per dispatch slot, so only writeBuffer of the changed params runs per token.
  function setup(pass: GPUComputePassEncoder, name: string, fields: Field[], ins: GPUBuffer[], outs: GPUBuffer[]): void {
    pass.setPipeline(pipelines[name])
    if (pooling) {
      let slot = dispPool[dispIdx]
      if (!slot) {
        slot = { uni: device.createBuffer({ size: 64, usage: U | CD }), bg: null, last: null }
        dispPool[dispIdx] = slot
      }
      const data2 = makeParams(fields) // reused view; only writeBuffer when the params changed
      if (!slot.last || !eqBytes(slot.last, data2)) {
        device.queue.writeBuffer(slot.uni, 0, data2 as BufferSource)
        slot.last = data2.slice()
      }
      if (!slot.bg) {
        const entries: GPUBindGroupEntry[] = [{ binding: 0, resource: { buffer: slot.uni } }]
        ins.forEach((b, i) => entries.push({ binding: i + 1, resource: { buffer: b } }))
        outs.forEach((b, i) => entries.push({ binding: 1 + ins.length + i, resource: { buffer: b } }))
        slot.bg = device.createBindGroup({ layout: pipelines[name].getBindGroupLayout(0), entries })
      }
      pass.setBindGroup(0, slot.bg)
      dispIdx++
    } else {
      const entries: GPUBindGroupEntry[] = [{ binding: 0, resource: { buffer: upload(makeParams(fields), U | CD) } }]
      ins.forEach((b, i) => entries.push({ binding: i + 1, resource: { buffer: b } }))
      outs.forEach((b, i) => entries.push({ binding: 1 + ins.length + i, resource: { buffer: b } }))
      pass.setBindGroup(0, device.createBindGroup({ layout: pipelines[name].getBindGroupLayout(0), entries }))
    }
  }
  function runIO(pass: GPUComputePassEncoder, name: string, fields: Field[], ins: GPUBuffer[], outs: GPUBuffer[], threads: number): void {
    setup(pass, name, fields, ins, outs)
    pass.dispatchWorkgroups(isFull(name) ? Math.ceil(threads / 64) : 1)
  }
  const run = (pass: GPUComputePassEncoder, name: string, fields: Field[], ins: GPUBuffer[], out: GPUBuffer, threads: number): void =>
    runIO(pass, name, fields, ins, [out], threads)
  // dispatch exactly nWG workgroups (subgroup kernels: one workgroup per row / per (query,head))
  function runN(pass: GPUComputePassEncoder, name: string, fields: Field[], ins: GPUBuffer[], out: GPUBuffer, nWG: number): void {
    setup(pass, name, fields, ins, [out])
    pass.dispatchWorkgroups(isFull(name) ? nWG : 1)
  }
  // 2D workgroup dispatch (subgroup GEMV: one workgroup per output column)
  function runWG(pass: GPUComputePassEncoder, name: string, fields: Field[], ins: GPUBuffer[], outs: GPUBuffer[], wgX: number, wgY: number): void {
    setup(pass, name, fields, ins, outs)
    const f = isFull(name)
    pass.dispatchWorkgroups(f ? wgX : 1, f ? wgY : 1, 1)
  }
  const rms = (pass: GPUComputePassEncoder, x: GPUBuffer, g: string, R: number, Dn: number, out: GPUBuffer): void =>
    useSG
      ? runN(pass, 'rmsnorm_sg', [['u', R], ['u', Dn], ['f', A.rms_eps], ['u', 0]], [x, W[g].buf!], out, R)
      : run(pass, 'rmsnorm', [['u', R], ['u', Dn], ['f', A.rms_eps], ['u', 0]], [x, W[g].buf!], out, R)
  // fused q/k/v or gate/up matmul
  function fusedMM(pass: GPUComputePassEncoder, w: GpuWeight, inBuf: GPUBuffer, S: number, outs: GPUBuffer[]): void {
    const Ntot = w.N0! + w.N1! + w.N2!
    if (useSG && S === 1) {
      const gx = Math.min(Ntot, 65535)
      runWG(pass, 'matmul_split_sg', [['u', w.K!], ['u', w.nb!], ['u', w.N0!], ['u', w.N1!], ['u', w.N2!], ['u', gx]], [inBuf, w.sign!, w.scales!], outs, gx, Math.ceil(Ntot / gx))
    } else if (S === 1) {
      const gx = Math.min(Ntot, 65535) // no-subgroup decode: workgroup-reduction GEMV
      runWG(pass, 'matmul_split_wg', [['u', w.K!], ['u', w.nb!], ['u', w.N0!], ['u', w.N1!], ['u', w.N2!], ['u', gx]], [inBuf, w.sign!, w.scales!], outs, gx, Math.ceil(Ntot / gx))
    } else if (tiledPrefill(S)) {
      runWG(pass, 'matmul_split_tiled', [['u', S], ['u', w.K!], ['u', w.nb!], ['u', w.N0!], ['u', w.N1!], ['u', w.N2!]], [inBuf, w.sign!, w.scales!], outs, Math.ceil(Ntot / 64), Math.ceil(S / 64)) // long-prompt prefill: tiled GEMM
    } else {
      runIO(pass, 'matmul_split', [['u', S], ['u', w.K!], ['u', w.nb!], ['u', w.N0!], ['u', w.N1!], ['u', w.N2!]], [inBuf, w.sign!, w.scales!], outs, S * Ntot)
    }
  }
  // o_proj / down_proj matmul with fused residual add
  function residMM(pass: GPUComputePassEncoder, w: GpuWeight, inBuf: GPUBuffer, resid: GPUBuffer, S: number, out: GPUBuffer): void {
    if (useSG && S === 1) {
      const nwg = Math.ceil(w.N! / ROWS_MR) // multi-row GEMV: ROWS_MR output cols per workgroup
      const gx = Math.min(nwg, 65535)
      runWG(pass, 'matmul_resid_mr_sg', [['u', w.N!], ['u', w.K!], ['u', w.nb!], ['u', gx], ['u', 0], ['u', 0]], [inBuf, w.sign!, w.scales!, resid], [out], gx, Math.ceil(nwg / gx))
    } else if (S === 1) {
      const gx = Math.min(w.N!, 65535) // no-subgroup decode: workgroup-reduction GEMV + residual
      runWG(pass, 'matmul_resid_wg', [['u', w.N!], ['u', w.K!], ['u', w.nb!], ['u', gx], ['u', 0], ['u', 0]], [inBuf, w.sign!, w.scales!, resid], [out], gx, Math.ceil(w.N! / gx))
    } else if (tiledPrefill(S)) {
      runWG(pass, 'matmul_resid_tiled', [['u', S], ['u', w.N!], ['u', w.K!], ['u', w.nb!], ['u', 0], ['u', 0]], [inBuf, w.sign!, w.scales!, resid], [out], Math.ceil(w.N! / 64), Math.ceil(S / 64)) // long-prompt prefill: tiled GEMM
    } else {
      runIO(pass, 'matmul_resid', [['u', S], ['u', w.N!], ['u', w.K!], ['u', w.nb!], ['u', 128], ['u', 0]], [inBuf, w.sign!, w.scales!, resid], [out], S * w.N!)
    }
  }

  function layer(pass: GPUComputePassEncoder, li: number, h: GPUBuffer, S: number, posBase: number, cos: GPUBuffer, sin: GPUBuffer): GPUBuffer {
    const Ltot = posBase + S
    const n1 = actBuf(S * Hd)
    rms(pass, h, `layers.${li}.input_layernorm`, S, Hd, n1)
    const qkv = W[`layers.${li}.attn.qkv`]

    if (useSG && S === 1 && !FORCE_SLOW) {
      // fused decode path: fold copies and elementwise ops into the matmul/norm kernels.
      const q = actBuf(H * Dh),
        k = actBuf(KV * Dh),
        v = actBuf(KV * Dh)
      const Ntot = qkv.N0! + qkv.N1! + qkv.N2!,
        gx = Math.min(Ntot, 65535)
      runWG(pass, 'matmul_split_sg', [['u', qkv.K!], ['u', qkv.nb!], ['u', qkv.N0!], ['u', qkv.N1!], ['u', qkv.N2!], ['u', gx]], [n1, qkv.sign!, qkv.scales!], [q, k, v], gx, Math.ceil(Ntot / gx))
      run(pass, 'copy', [['u', KV * Dh], ['u', posBase * KV * Dh], ['u', 0], ['u', 0]], [v], Vc[li], KV * Dh)
      const qr = actBuf(H * Dh)
      runN(pass, 'rmsnorm_rope_sg', [['u', H], ['u', Dh], ['f', A.rms_eps], ['u', 0], ['u', Dh], ['u', 0]], [q, W[`layers.${li}.attn.q_norm`].buf!, cos, sin], qr, H)
      runN(pass, 'rmsnorm_rope_sg', [['u', KV], ['u', Dh], ['f', A.rms_eps], ['u', posBase * KV * Dh], ['u', Dh], ['u', 0]], [k, W[`layers.${li}.attn.k_norm`].buf!, cos, sin], Kc[li], KV)
      cap(li, 'qr', qr)
      const att = actBuf(H * Dh)
      runN(pass, 'attention_sg', [['u', 1], ['u', H], ['u', KV], ['u', Dh], ['u', posBase], ['u', Ltot]], [qr, Kc[li], Vc[li]], att, H)
      cap(li, 'att', att)
      const o = W[`layers.${li}.attn.o_proj`],
        h2 = actBuf(Hd)
      residMM(pass, o, att, h, 1, h2)
      const n2 = actBuf(Hd)
      rms(pass, h2, `layers.${li}.post_attention_layernorm`, 1, Hd, n2)
      const gu = W[`layers.${li}.mlp.gateup`],
        sw = actBuf(F),
        nwgF = Math.ceil(F / ROWS_MR),
        gxF = Math.min(nwgF, 65535)
      runWG(pass, 'matmul_swiglu_mr_sg', [['u', gu.K!], ['u', gu.nb!], ['u', F], ['u', gxF], ['u', 0], ['u', 0]], [n2, gu.sign!, gu.scales!], [sw], gxF, Math.ceil(nwgF / gxF))
      cap(li, 'sw', sw)
      const d = W[`layers.${li}.mlp.down_proj`],
        hn = actBuf(Hd)
      residMM(pass, d, sw, h2, 1, hn)
      return hn
    }

    // prefill / no-subgroup path: separate kernels (kept verbatim; validates correctness end to end)
    const q = actBuf(S * H * Dh),
      k = actBuf(S * KV * Dh),
      v = actBuf(S * KV * Dh)
    fusedMM(pass, qkv, n1, S, [q, k, v])
    const qn = actBuf(S * H * Dh),
      kn = actBuf(S * KV * Dh)
    rms(pass, q, `layers.${li}.attn.q_norm`, S * H, Dh, qn)
    rms(pass, k, `layers.${li}.attn.k_norm`, S * KV, Dh, kn)
    const qr = actBuf(S * H * Dh),
      kr = actBuf(S * KV * Dh)
    run(pass, 'rope', [['u', S], ['u', H], ['u', Dh], ['u', 0]], [qn, cos, sin], qr, S * H * Dh)
    run(pass, 'rope', [['u', S], ['u', KV], ['u', Dh], ['u', 0]], [kn, cos, sin], kr, S * KV * Dh)
    run(pass, 'copy', [['u', S * KV * Dh], ['u', posBase * KV * Dh], ['u', 0], ['u', 0]], [kr], Kc[li], S * KV * Dh)
    run(pass, 'copy', [['u', S * KV * Dh], ['u', posBase * KV * Dh], ['u', 0], ['u', 0]], [v], Vc[li], S * KV * Dh)
    cap(li, 'qr', qr)
    const att = actBuf(S * H * Dh)
    const attF: Field[] = [['u', S], ['u', H], ['u', KV], ['u', Dh], ['u', posBase], ['u', Ltot]]
    if (useSG) runN(pass, 'attention_sg', attF, [qr, Kc[li], Vc[li]], att, S * H)
    else run(pass, 'attention_cache', attF, [qr, Kc[li], Vc[li]], att, S * H)
    cap(li, 'att', att)
    const o = W[`layers.${li}.attn.o_proj`],
      h2 = actBuf(S * Hd)
    residMM(pass, o, att, h, S, h2)
    const n2 = actBuf(S * Hd)
    rms(pass, h2, `layers.${li}.post_attention_layernorm`, S, Hd, n2)
    const gu = W[`layers.${li}.mlp.gateup`],
      g = actBuf(S * F),
      u = actBuf(S * F)
    fusedMM(pass, gu, n2, S, [g, u, dummy])
    const sw = actBuf(S * F)
    run(pass, 'swiglu', [['u', S * F], ['u', 0], ['u', 0], ['u', 0]], [g, u], sw, S * F)
    cap(li, 'sw', sw)
    const d = W[`layers.${li}.mlp.down_proj`],
      hn = actBuf(S * Hd)
    residMM(pass, d, sw, h2, S, hn)
    return hn
  }
  function lmHead(pass: GPUComputePassEncoder, fn: GPUBuffer, M: number, out: GPUBuffer): void {
    const lm = W.lm_head
    if (useSG && M === 1) {
      const gx = Math.min(lm.N!, 65535)
      runWG(pass, 'matmul_q2_sg', [['u', lm.N!], ['u', lm.K!], ['u', lm.nb!], ['u', lm.zp!], ['u', gx], ['u', 0]], [fn, lm.codes!, lm.scales!], [out], gx, Math.ceil(lm.N! / gx))
    } else if (M === 1) {
      const gx = Math.min(lm.N!, 65535) // no-subgroup decode: workgroup-reduction 2-bit GEMV
      runWG(pass, 'matmul_q2_wg', [['u', lm.N!], ['u', lm.K!], ['u', lm.nb!], ['u', lm.zp!], ['u', gx], ['u', 0]], [fn, lm.codes!, lm.scales!], [out], gx, Math.ceil(lm.N! / gx))
    } else {
      run(pass, 'matmul_q2', [['u', M], ['u', lm.N!], ['u', lm.K!], ['u', lm.nb!], ['u', 128], ['u', lm.zp!]], [fn, lm.codes!, lm.scales!], out, M * lm.N!)
    }
  }

  function stack(enc: GPUCommandEncoder, h: GPUBuffer, S: number, posBase: number): { fn: GPUBuffer; layer0: GPUBuffer | null } {
    const { cos, sin } = ropeBufs(posBase, S)
    const pass = enc.beginComputePass()
    let cur = h,
      layer0: GPUBuffer | null = null
    for (let li = 0; li < A.layers; li++) {
      cur = layer(pass, li, cur, S, posBase, cos, sin)
      if (li === 0) layer0 = cur
    }
    const fn = actBuf(S * Hd)
    rms(pass, cur, 'layers.28.final_norm_layernorm', S, Hd, fn)
    pass.end()
    return { fn, layer0 }
  }

  async function forward(ids: number[]): Promise<ForwardResult> {
    const S = ids.length,
      embedOut = upload(embedDequant(ids), S_ | CD | CS)
    const enc = device.createCommandEncoder()
    const { fn, layer0 } = stack(enc, embedOut, S, 0)
    const logits = device.createBuffer({ size: S * W.lm_head.N! * 4, usage: S_ | CS })
    const pass = enc.beginComputePass()
    lmHead(pass, fn, S, logits)
    pass.end()
    device.queue.submit([enc.finish()])
    await device.queue.onSubmittedWorkDone()
    return {
      embed: await readback(embedOut, S * Hd),
      layer0: await readback(layer0!, S * Hd),
      finalnorm: await readback(fn, S * Hd),
      logits: await readback(logits, S * W.lm_head.N!),
      vocab: W.lm_head.N!,
      sequenceLength: S,
    }
  }

  // GPU-resident decode: argmax + embedding gather run on the GPU so the token id never leaves it;
  // chain syncN steps per CPU sync (deferred readback). Bit-exact: only the readback timing changes.
  async function generateImpl(ids: number[], nTokens: number, full: Set<string> | null = null, syncN: number = SYNC_N, ctl?: { stopTokens?: number[]; onToken?: (id: number) => void; signal?: AbortSignal }): Promise<RawGenResult> {
    FULL = full
    const vocab = W.lm_head.N!
    const tokBuf = device.createBuffer({ size: Math.max(1, nTokens) * 4, usage: S_ | CS }) // GPU-resident token ids
    const embG = actBuf(Hd) // GPU embedding of the current token
    const lg = device.createBuffer({ size: vocab * 4, usage: S_ | CS })

    const t0 = performance.now()
    // prefill (CPU embed of the known prompt) -> last hidden -> lm_head -> GPU argmax -> tokBuf[0]
    const encP = device.createCommandEncoder()
    const { fn } = stack(encP, upload(embedDequant(ids), S_ | CD), ids.length, 0)
    const lastP = actBuf(Hd)
    encP.copyBufferToBuffer(fn, (ids.length - 1) * Hd * 4, lastP, 0, Hd * 4)
    let pp = encP.beginComputePass()
    lmHead(pp, lastP, 1, lg)
    pp.end()
    pp = encP.beginComputePass()
    runN(pp, 'argmax', [['u', vocab], ['u', 0], ['u', 0], ['u', 0]], [lg], tokBuf, 1)
    pp.end()
    device.queue.submit([encP.finish()])
    await device.queue.onSubmittedWorkDone()
    const firstTok = (await readbackU32(tokBuf, 1))[0]
    const prefillMs = performance.now() - t0

    const gen = [firstTok]
    let recMs = 0,
      gpuMs = 0,
      rbMs = 0
    const t1 = performance.now()
    let total = 1 // tokens emitted (incl. prefill's first)
    const stopSet = ctl?.stopTokens ? new Set(ctl.stopTokens) : null
    if (ctl?.onToken && !stopSet?.has(firstTok)) ctl.onToken(firstTok)
    let stopped = stopSet?.has(firstTok) ?? false
    pooling = true // reuse decode scratch + uniform buffers across batches
    poolInvalidate() // rebuild cached bind groups against this call's buffers
    while (total < nTokens && !stopped) {
      if (ctl?.signal?.aborted) break
      const batch = Math.min(syncN, nTokens - total)
      poolReset()
      let t = performance.now()
      const enc = device.createCommandEncoder()
      for (let j = 0; j < batch; j++) {
        const idxOut = total + j,
          pos = ids.length + idxOut - 1
        let pass = enc.beginComputePass()
        runN(pass, 'embed_gather', [['u', Hd], ['u', idxOut - 1], ['u', 0], ['u', 0]], [tokBuf, embWqG, tgt4G, embScalesG, embZpG], embG, 1)
        pass.end()
        const r = stack(enc, embG, 1, pos)
        const last = actBuf(Hd)
        enc.copyBufferToBuffer(r.fn, 0, last, 0, Hd * 4)
        pass = enc.beginComputePass()
        lmHead(pass, last, 1, lg)
        runN(pass, 'argmax', [['u', vocab], ['u', idxOut], ['u', 0], ['u', 0]], [lg], tokBuf, 1)
        pass.end()
      }
      device.queue.submit([enc.finish()])
      recMs += performance.now() - t
      t = performance.now()
      await device.queue.onSubmittedWorkDone()
      gpuMs += performance.now() - t
      t = performance.now()
      const toks = await readbackU32(tokBuf, total + batch)
      rbMs += performance.now() - t
      for (let j = 0; j < batch; j++) {
        const tk = toks[total + j]
        if (stopSet?.has(tk)) { stopped = true; break } // EOS lands at the batch boundary (greedy)
        gen.push(tk)
        ctl?.onToken?.(tk)
      }
      total += batch
    }
    pooling = false
    const decodeMs = performance.now() - t1,
      nd = Math.max(1, gen.length - 1)
    FULL = null
    return { prefillMs, decodeMs, tokPerSec: nd / (decodeMs / 1000), tokens: gen, firstArgmax: gen[0], recMs: recMs / nd, gpuMs: gpuMs / nd, rbMs: rbMs / nd }
  }

  // Sampled decode (do_sample): the GPU pre-filters the logits in place (repetition_penalty +
  // no_repeat_ngram bans) and selects the top-K via K masked-argmax passes; only K (id, logit) pairs
  // are read back, and the CPU does temperature + softmax + MT19937 multinomial (exact transformers.js
  // semantics). Per-step (syncN=1) because the chosen token is picked on the CPU and feeds the next
  // step's embed gather. Greedy decode (generateImpl) is the separate, untouched GPU-resident path.
  async function generateSampledImpl(ids: number[], nTokens: number, genOpts: GenerateOptions): Promise<RawGenResult> {
    const vocab = W.lm_head.N!
    const K = Math.max(1, Math.min(genOpts.topK ?? 20, vocab))
    const temperature = genOpts.temperature ?? 1
    const penalty = genOpts.repetitionPenalty ?? 1
    const ngramN = genOpts.noRepeatNgramSize ?? 0
    const stopSet = genOpts.stopTokens ? new Set(genOpts.stopTokens) : null
    const onToken = genOpts.onToken
    const signal = genOpts.signal
    const rng = new MT19937(genOpts.seed)

    // persistent buffers (stable across steps for bind-group caching; not via actBuf)
    const tokBuf = device.createBuffer({ size: Math.max(1, nTokens) * 4, usage: S_ | CS | CD })
    const lg = device.createBuffer({ size: vocab * 4, usage: S_ | CS })
    const candIds = device.createBuffer({ size: K * 4, usage: S_ | CS })
    const candVals = device.createBuffer({ size: K * 4, usage: S_ | CS })
    const maxHist = ids.length + nTokens
    const affBuf = device.createBuffer({ size: Math.max(1, maxHist) * 4, usage: S_ | CD })
    const banBuf = device.createBuffer({ size: Math.max(1, maxHist) * 4, usage: S_ | CD })
    const rbBuf = device.createBuffer({ size: K * 8, usage: GPUBufferUsage.MAP_READ | CD })
    const embG = device.createBuffer({ size: Hd * 4, usage: S_ | CS | CD })

    // upload the CPU-computed deduped id set + ngram bans for the current history; return their lengths
    const writeAffBan = (history: number[]): { affLen: number; banLen: number } => {
      const aff = penalty !== 1 ? affectedIds(history) : new Uint32Array(0)
      if (aff.length) device.queue.writeBuffer(affBuf, 0, aff)
      const ban = ngramN > 0 ? ngramBans(history, ngramN) : []
      if (ban.length) device.queue.writeBuffer(banBuf, 0, Uint32Array.from(ban))
      return { affLen: aff.length, banLen: ban.length }
    }
    // penalty pre-filter + K masked-argmax, all in the given pass (after lm_head wrote lg)
    const samplerChain = (pass: GPUComputePassEncoder, affLen: number, banLen: number): void => {
      setup(pass, 'sampler_penalty', [['u', affLen], ['u', banLen], ['f', penalty], ['u', 0xff800000]], [affBuf, banBuf], [lg])
      pass.dispatchWorkgroups(1)
      for (let r = 0; r < K; r++) {
        setup(pass, 'argmax_masked', [['u', vocab], ['u', r], ['u', 0], ['u', 0]], [lg], [candIds, candVals])
        pass.dispatchWorkgroups(1)
      }
    }
    const readCands = async (): Promise<{ ci: Uint32Array; cv: Float32Array }> => {
      await rbBuf.mapAsync(GPUMapMode.READ)
      const mapped = rbBuf.getMappedRange()
      const ci = new Uint32Array(mapped.slice(0, K * 4))
      const cv = new Float32Array(mapped.slice(K * 4, K * 8))
      rbBuf.unmap()
      return { ci, cv }
    }

    const t0 = performance.now()
    const history = [...ids]
    // prefill -> last hidden -> lm_head -> sampler chain (non-pooling), CPU samples the first token
    const encP = device.createCommandEncoder()
    const { fn } = stack(encP, upload(embedDequant(ids), S_ | CD), ids.length, 0)
    const lastP = actBuf(Hd)
    encP.copyBufferToBuffer(fn, (ids.length - 1) * Hd * 4, lastP, 0, Hd * 4)
    const pf = writeAffBan(history)
    let pass = encP.beginComputePass()
    lmHead(pass, lastP, 1, lg)
    samplerChain(pass, pf.affLen, pf.banLen)
    pass.end()
    encP.copyBufferToBuffer(candIds, 0, rbBuf, 0, K * 4)
    encP.copyBufferToBuffer(candVals, 0, rbBuf, K * 4, K * 4)
    device.queue.submit([encP.finish()])
    await device.queue.onSubmittedWorkDone()
    const first = await readCands()
    const firstTok = sampleFromCandidates(first.ci, first.cv, temperature, rng)
    const prefillMs = performance.now() - t0

    const gen = [firstTok]
    history.push(firstTok)
    let stopped = stopSet?.has(firstTok) ?? false
    if (onToken && !stopped) onToken(firstTok)
    device.queue.writeBuffer(tokBuf, 0, new Uint32Array([firstTok]))

    let recMs = 0, gpuMs = 0, rbMs = 0
    const t1 = performance.now()
    let total = 1
    pooling = true
    poolInvalidate() // rebuild cached bind groups against this call's buffers
    while (total < nTokens && !stopped) {
      if (signal?.aborted) break
      poolReset()
      const idxOut = total, pos = ids.length + idxOut - 1
      let t = performance.now()
      const { affLen, banLen } = writeAffBan(history)
      const enc = device.createCommandEncoder()
      let p2 = enc.beginComputePass()
      runN(p2, 'embed_gather', [['u', Hd], ['u', idxOut - 1], ['u', 0], ['u', 0]], [tokBuf, embWqG, tgt4G, embScalesG, embZpG], embG, 1)
      p2.end()
      const r = stack(enc, embG, 1, pos)
      const last = actBuf(Hd)
      enc.copyBufferToBuffer(r.fn, 0, last, 0, Hd * 4)
      p2 = enc.beginComputePass()
      lmHead(p2, last, 1, lg)
      samplerChain(p2, affLen, banLen)
      p2.end()
      enc.copyBufferToBuffer(candIds, 0, rbBuf, 0, K * 4)
      enc.copyBufferToBuffer(candVals, 0, rbBuf, K * 4, K * 4)
      device.queue.submit([enc.finish()])
      recMs += performance.now() - t
      t = performance.now()
      await device.queue.onSubmittedWorkDone()
      gpuMs += performance.now() - t
      t = performance.now()
      const { ci, cv } = await readCands()
      rbMs += performance.now() - t
      const tk = sampleFromCandidates(ci, cv, temperature, rng)
      total += 1
      if (stopSet?.has(tk)) { stopped = true; break } // EOS: stop without emitting the stop token
      gen.push(tk)
      history.push(tk)
      onToken?.(tk)
      device.queue.writeBuffer(tokBuf, idxOut * 4, new Uint32Array([tk])) // feed the next step's embed gather
    }
    pooling = false
    const decodeMs = performance.now() - t1
    const nd = Math.max(1, gen.length - 1)
    return { prefillMs, decodeMs, tokPerSec: nd / (decodeMs / 1000), tokens: gen, firstArgmax: gen[0], recMs: recMs / nd, gpuMs: gpuMs / nd, rbMs: rbMs / nd }
  }

  // Run ONE decode step at the same position through the fused path and the slow (known-good) path
  // and return layer-0 checkpoints + final norm + logits for each, so a divergence pinpoints the
  // first fused kernel that differs.
  async function debugDecode(prefillIds: number[]): Promise<{ fast: Record<string, Float32Array>; slow: Record<string, Float32Array> }> {
    const encP = device.createCommandEncoder()
    stack(encP, upload(embedDequant(prefillIds), S_ | CD), prefillIds.length, 0)
    device.queue.submit([encP.finish()])
    await device.queue.onSubmittedWorkDone()
    const pos = prefillIds.length,
      tok = prefillIds[prefillIds.length - 1]
    const runStep = async (forceSlow: boolean): Promise<Record<string, Float32Array>> => {
      FORCE_SLOW = forceSlow
      DBG0 = {}
      const enc = device.createCommandEncoder()
      const r = stack(enc, upload(embedDequant([tok]), S_ | CD), 1, pos)
      const lg = device.createBuffer({ size: W.lm_head.N! * 4, usage: S_ | CS })
      const pass = enc.beginComputePass()
      lmHead(pass, r.fn, 1, lg)
      pass.end()
      device.queue.submit([enc.finish()])
      await device.queue.onSubmittedWorkDone()
      const ck: Record<string, Float32Array> = {}
      for (const [name, b] of Object.entries(DBG0)) ck[name] = await readback(b, b.size / 4)
      const off = pos * KV * Dh
      ck.kc = (await readback(Kc[0], MAXSEQ * KV * Dh)).slice(off, off + KV * Dh)
      ck.vc = (await readback(Vc[0], MAXSEQ * KV * Dh)).slice(off, off + KV * Dh)
      ck.fn = await readback(r.fn, Hd)
      ck.logits = await readback(lg, W.lm_head.N!)
      FORCE_SLOW = false
      DBG0 = null
      return ck
    }
    const fast = await runStep(false),
      slow = await runStep(true)
    return { fast, slow }
  }

  // Debug hook for the browser harness: run a prefill for `ids` (history = ids), then return the GPU
  // lm_head logits (base, pre-penalty), the GPU penalized logits, and the GPU top-K. The page penalizes
  // `base` on the CPU and diffs vs `penalized` (exact, same input), and compares its top-K vs candIds,
  // validating sampler_penalty.wgsl and argmax_masked.wgsl in isolation against the headless-checked math.
  async function debugSampler(ids: number[], genOpts: GenerateOptions): Promise<{ base: Float32Array; penalized: Float32Array; candIds: Uint32Array; candVals: Float32Array }> {
    const vocab = W.lm_head.N!
    const K = Math.max(1, Math.min(genOpts.topK ?? 20, vocab))
    const penalty = genOpts.repetitionPenalty ?? 1
    const ngramN = genOpts.noRepeatNgramSize ?? 0
    const lg = device.createBuffer({ size: vocab * 4, usage: S_ | CS })
    const candIds = device.createBuffer({ size: K * 4, usage: S_ | CS })
    const candVals = device.createBuffer({ size: K * 4, usage: S_ | CS })
    const aff = penalty !== 1 ? affectedIds(ids) : new Uint32Array(0)
    const ban = ngramN > 0 ? ngramBans(ids, ngramN) : []
    const affBuf = upload(aff.length ? aff : new Uint32Array(1), S_ | CD)
    const banBuf = upload(ban.length ? Uint32Array.from(ban) : new Uint32Array(1), S_ | CD)
    // pass 1: prefill -> lm_head -> base logits
    const enc1 = device.createCommandEncoder()
    const { fn } = stack(enc1, upload(embedDequant(ids), S_ | CD), ids.length, 0)
    const lastP = device.createBuffer({ size: Hd * 4, usage: S_ | CS | CD })
    enc1.copyBufferToBuffer(fn, (ids.length - 1) * Hd * 4, lastP, 0, Hd * 4)
    let pass = enc1.beginComputePass()
    lmHead(pass, lastP, 1, lg)
    pass.end()
    device.queue.submit([enc1.finish()])
    await device.queue.onSubmittedWorkDone()
    const base = await readback(lg, vocab)
    // pass 2: penalty (in place on lg) + K masked-argmax
    const enc2 = device.createCommandEncoder()
    pass = enc2.beginComputePass()
    setup(pass, 'sampler_penalty', [['u', aff.length], ['u', ban.length], ['f', penalty], ['u', 0xff800000]], [affBuf, banBuf], [lg])
    pass.dispatchWorkgroups(1)
    for (let r = 0; r < K; r++) {
      setup(pass, 'argmax_masked', [['u', vocab], ['u', r], ['u', 0], ['u', 0]], [lg], [candIds, candVals])
      pass.dispatchWorkgroups(1)
    }
    pass.end()
    device.queue.submit([enc2.finish()])
    await device.queue.onSubmittedWorkDone()
    return { base, penalized: await readback(lg, vocab), candIds: await readbackU32(candIds, K), candVals: await readback(candVals, K) }
  }

  const capabilities: EngineCapabilities = {
    useSubgroups: useSG,
    subgroupSize: sgMax,
    adapter: { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description },
    limits: {
      maxStorageBufferBindingSize: Number(L.maxStorageBufferBindingSize),
      maxComputeWorkgroupStorageSize: L.maxComputeWorkgroupStorageSize,
    },
  }

  // Public generate: routes to sampled decode when a sampling temperature is set, else greedy.
  // Both honor stopTokens / onToken / signal and map to the typed result shape.
  async function generate(promptTokenIds: number[], genOpts: GenerateOptions = {}): Promise<GenerateResult> {
    const maxTokens = genOpts.maxTokens ?? 256
    const sampled = genOpts.temperature != null && genOpts.temperature > 0 && genOpts.temperature !== 1
    const r = sampled
      ? await generateSampledImpl(promptTokenIds, maxTokens, genOpts)
      : await generateImpl(promptTokenIds, maxTokens, null, SYNC_N, { stopTokens: genOpts.stopTokens, onToken: genOpts.onToken, signal: genOpts.signal })
    return {
      tokens: r.tokens,
      prefillMs: r.prefillMs,
      decodeMs: r.decodeMs,
      tokensPerSecond: r.tokPerSec,
      timing: { recordMs: r.recMs, gpuMs: r.gpuMs, readbackMs: r.rbMs },
    }
  }

  const api: EngineInternal = {
    generate,
    forward,
    capabilities,
    dispose: () => device.destroy(),
    device,
    adapter,
    profileDecode: (ids, nTokens, full = null, syncN = SYNC_N) => generateImpl(ids, nTokens, full, syncN),
    debugDecode,
    debugSampler,
  }
  return api
}
