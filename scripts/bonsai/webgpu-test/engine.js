// Bonsai-1.7B WebGPU runtime (P2). Loads weights via the manifest, runs the Qwen3 forward
// with the validated kernels, keeps a persistent KV cache, generates autoregressively, and
// measures decode tok/s. Decode is dispatch-overhead-bound, so the matmuls are fused: q/k/v
// in one dispatch, gate/up in one, and the residual add folded into o_proj/down_proj.

const VIEW = { FLOAT: Float32Array, UINT8: Uint8Array, FLOAT16: Uint16Array };
const WGSLS = ['matmul_binary_vec4', 'matmul_split', 'matmul_resid', 'matmul_q2', 'rmsnorm', 'rope', 'swiglu', 'attention_cache', 'add', 'copy'];
const MAXSEQ = 256;

function makeParams(fields) {
  const ab = new ArrayBuffer(Math.ceil(fields.length / 4) * 16);
  const dv = new DataView(ab);
  fields.forEach(([t, v], i) => t === 'f' ? dv.setFloat32(i * 4, v, true) : dv.setUint32(i * 4, v >>> 0, true));
  return ab;
}
const concat = (Cls, arrs) => { let n = 0; for (const a of arrs) n += a.length; const o = new Cls(n); let p = 0; for (const a of arrs) { o.set(a, p); p += a.length; } return o; };

export async function createEngine(modelDir) {
  const manifest = await (await fetch(`${modelDir}/manifest.json`)).json();
  const data = await (await fetch(`${modelDir}/${manifest.data_file}`)).arrayBuffer();
  const aux = await (await fetch(`${modelDir}/${manifest.aux_file}`)).arrayBuffer();
  const A = manifest.arch, T = manifest.tensors;

  const readRef = (ref) => {
    const src = ref.src === 'aux' ? aux : data;
    const V = VIEW[ref.dtype];
    if (V === Uint8Array) return new Uint8Array(src, ref.off, ref.len);
    if (ref.off % V.BYTES_PER_ELEMENT === 0) return new V(src, ref.off, ref.len / V.BYTES_PER_ELEMENT);
    return new V(src.slice(ref.off, ref.off + ref.len));
  };

  const adapter = await navigator.gpu.requestAdapter();
  const hasSG = adapter.features.has('subgroups');
  const info = adapter.info ?? {};                          // subgroup sizes live on GPUAdapterInfo
  const sgMax = info.subgroupMaxSize ?? 32, sgMin = info.subgroupMinSize ?? sgMax;
  const forceNoSG = typeof location !== 'undefined' && new URLSearchParams(location.search).has('nosg');
  const useSG = hasSG && sgMin === sgMax && (sgMax === 32 || sgMax === 64) && !forceNoSG;  // uniform >=32 -> head_dim/SG<=4; ?nosg forces the v1 fallback
  const device = await adapter.requestDevice({ requiredFeatures: useSG ? ['subgroups'] : [] });
  const pipelines = {};
  const mkPipe = async (name, constants) => {
    const code = await (await fetch(`./${name}.wgsl`)).text();
    pipelines[name] = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'main', constants } });
  };
  const ROWS_MR = 4;                                        // output rows per workgroup in the multi-row GEMV
  for (const name of WGSLS) await mkPipe(name);
  if (useSG) for (const n of ['rmsnorm_sg', 'attention_sg', 'matmul_split_sg', 'matmul_q2_sg', 'rmsnorm_rope_sg']) await mkPipe(n, { SG: sgMax });
  if (useSG) for (const n of ['matmul_resid_mr_sg', 'matmul_swiglu_mr_sg']) await mkPipe(n, { SG: sgMax, ROWS: ROWS_MR });
  if (!useSG) for (const n of ['matmul_split_wg', 'matmul_resid_wg', 'matmul_q2_wg']) await mkPipe(n, { WG: 64 });  // no-subgroup fallback: workgroup-reduction GEMV

  const S_ = GPUBufferUsage.STORAGE, CD = GPUBufferUsage.COPY_DST, CS = GPUBufferUsage.COPY_SRC, U = GPUBufferUsage.UNIFORM;
  const upload = (typed, usage = S_ | CD) => { const b = device.createBuffer({ size: typed.byteLength, usage }); device.queue.writeBuffer(b, 0, typed); return b; };
  const actBuf = (n) => device.createBuffer({ size: n * 4, usage: S_ | CS | CD });
  const dummy = device.createBuffer({ size: 16, usage: S_ });

  const tgt2 = readRef(manifest.luts.tgt2), tgt4 = readRef(manifest.luts.tgt4);
  const signTable = new Uint8Array(256);
  for (let b = 0; b < 256; b++) { let bits = 0; for (let j = 0; j < 8; j++) bits |= (((tgt2[2 * b + (j >> 2)] >> (2 * (j & 3))) & 3) >> 1 & 1) << j; signTable[b] = bits; }
  const rawBin = (name) => { const t = T[name], wq = readRef(t.weight), sign = new Uint8Array(wq.length); for (let i = 0; i < wq.length; i++) sign[i] = signTable[wq[i]]; return { sign, scales: readRef(t.scales), N: t.N, K: t.K, nb: t.K / 128 }; };

  const W = {};
  for (const [name, t] of Object.entries(T)) {
    if (t.kind === 'q2') {
      const wq = readRef(t.weight), codes = new Uint8Array(wq.length * 2);
      for (let i = 0; i < wq.length; i++) { codes[2 * i] = tgt2[2 * wq[i]]; codes[2 * i + 1] = tgt2[2 * wq[i] + 1]; }
      W[name] = { N: t.N, K: t.K, nb: t.K / 128, zp: 2, codes: upload(codes), scales: upload(readRef(t.scales)) };
    } else if (t.kind === 'f32' && t.weight) {
      W[name] = { buf: upload(readRef(t.weight)) };
    }
  }
  // fuse per-layer matmul weights: qkv (3), gate/up (2); o_proj + down_proj stay individual (residual-folded)
  for (let li = 0; li < A.layers; li++) {
    const q = rawBin(`layers.${li}.attn.q_proj`), k = rawBin(`layers.${li}.attn.k_proj`), v = rawBin(`layers.${li}.attn.v_proj`);
    W[`layers.${li}.attn.qkv`] = { K: q.K, nb: q.nb, N0: q.N, N1: k.N, N2: v.N,
      sign: upload(concat(Uint8Array, [q.sign, k.sign, v.sign])), scales: upload(concat(Float32Array, [q.scales, k.scales, v.scales])) };
    const g = rawBin(`layers.${li}.mlp.gate_proj`), u = rawBin(`layers.${li}.mlp.up_proj`);
    W[`layers.${li}.mlp.gateup`] = { K: g.K, nb: g.nb, N0: g.N, N1: u.N, N2: 0,
      sign: upload(concat(Uint8Array, [g.sign, u.sign])), scales: upload(concat(Float32Array, [g.scales, u.scales])) };
    for (const nm of [`layers.${li}.attn.o_proj`, `layers.${li}.mlp.down_proj`]) { const r = rawBin(nm); W[nm] = { N: r.N, K: r.K, nb: r.nb, sign: upload(r.sign), scales: upload(r.scales) }; }
  }

  const embWq = readRef(T.embed_tokens.weight), embScales = readRef(T.embed_tokens.scales), embZp = readRef(T.embed_tokens.zp);
  const cosCache = readRef(T.cos_cache), sinCache = readRef(T.sin_cache);

  function embedDequant(ids) {
    const H = A.hidden, out = new Float32Array(ids.length * H);
    for (let r = 0; r < ids.length; r++) {
      const id = ids[r];
      for (let i = 0; i < 256; i++) for (let qd = 0; qd < 4; qd++) {
        const byte = tgt4[4 * embWq[id * 256 + i] + qd], baseK = (i * 4 + qd) * 2;
        for (let c = 0; c < 2; c++) {
          const k = baseK + c, code = (byte >> (4 * c)) & 15, blk = (k / 128) | 0;
          const zp = (embZp[id * 8 + ((blk / 2) | 0)] >> (4 * (blk & 1))) & 15;
          out[r * H + k] = (code - zp) * embScales[id * 16 + blk];
        }
      }
    }
    return out;
  }
  function ropeBufs(posBase, S) {
    const D = A.head_dim, cos = new Float32Array(S * D), sin = new Float32Array(S * D);
    for (let s = 0; s < S; s++) for (let d = 0; d < D; d++) { cos[s * D + d] = cosCache[(posBase + s) * 64 + (d % 64)]; sin[s * D + d] = sinCache[(posBase + s) * 64 + (d % 64)]; }
    return { cos: upload(cos), sin: upload(sin) };
  }

  const KV = A.kv_heads, Dh = A.head_dim, Hd = A.hidden, H = A.heads, F = A.intermediate;
  const Kc = [], Vc = [];
  for (let li = 0; li < A.layers; li++) { Kc.push(actBuf(MAXSEQ * KV * Dh)); Vc.push(actBuf(MAXSEQ * KV * Dh)); }

  async function readback(buf, n) {
    const rb = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.MAP_READ | CD });
    const enc = device.createCommandEncoder(); enc.copyBufferToBuffer(buf, 0, rb, 0, n * 4); device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ); const out = new Float32Array(rb.getMappedRange().slice(0)); rb.unmap(); return out;
  }

  // diagnostic: FULL = null -> every kernel at real size; FULL = Set(names) -> only those at real size,
  // all others dispatched as 1 workgroup. Lets us measure each kernel type's true in-context cost.
  let FULL = null;
  const isFull = (name) => FULL === null || FULL.has(name);
  // differential debug: FORCE_SLOW routes S=1 through the prefill (known-good) path; DBG0 collects
  // layer-0 checkpoint buffers so a fused step and a slow step can be compared kernel by kernel.
  let FORCE_SLOW = false, DBG0 = null;
  const cap = (li, name, buf) => { if (li === 0 && DBG0) DBG0[name] = buf; };
  function runIO(pass, name, fields, ins, outs, threads) {
    const entries = [{ binding: 0, resource: { buffer: upload(new Uint8Array(makeParams(fields)), U | CD) } }];
    ins.forEach((b, i) => entries.push({ binding: i + 1, resource: { buffer: b } }));
    outs.forEach((b, i) => entries.push({ binding: 1 + ins.length + i, resource: { buffer: b } }));
    pass.setPipeline(pipelines[name]);
    pass.setBindGroup(0, device.createBindGroup({ layout: pipelines[name].getBindGroupLayout(0), entries }));
    pass.dispatchWorkgroups(isFull(name) ? Math.ceil(threads / 64) : 1);
  }
  const run = (pass, name, fields, ins, out, threads) => runIO(pass, name, fields, ins, [out], threads);
  // dispatch exactly nWG workgroups (for subgroup kernels: one workgroup per row / per (query,head))
  function runN(pass, name, fields, ins, out, nWG) {
    const entries = [{ binding: 0, resource: { buffer: upload(new Uint8Array(makeParams(fields)), U | CD) } }];
    ins.forEach((b, i) => entries.push({ binding: i + 1, resource: { buffer: b } }));
    entries.push({ binding: ins.length + 1, resource: { buffer: out } });
    pass.setPipeline(pipelines[name]);
    pass.setBindGroup(0, device.createBindGroup({ layout: pipelines[name].getBindGroupLayout(0), entries }));
    pass.dispatchWorkgroups(isFull(name) ? nWG : 1);
  }
  // 2D workgroup dispatch (subgroup GEMV: one workgroup per output column)
  function runWG(pass, name, fields, ins, outs, wgX, wgY) {
    const entries = [{ binding: 0, resource: { buffer: upload(new Uint8Array(makeParams(fields)), U | CD) } }];
    ins.forEach((b, i) => entries.push({ binding: i + 1, resource: { buffer: b } }));
    outs.forEach((b, i) => entries.push({ binding: 1 + ins.length + i, resource: { buffer: b } }));
    pass.setPipeline(pipelines[name]);
    pass.setBindGroup(0, device.createBindGroup({ layout: pipelines[name].getBindGroupLayout(0), entries }));
    const f = isFull(name);
    pass.dispatchWorkgroups(f ? wgX : 1, f ? wgY : 1, 1);
  }
  const rms = (pass, x, g, R, Dn, out) => useSG
    ? runN(pass, 'rmsnorm_sg', [['u', R], ['u', Dn], ['f', A.rms_eps], ['u', 0]], [x, W[g].buf], out, R)
    : run(pass, 'rmsnorm', [['u', R], ['u', Dn], ['f', A.rms_eps], ['u', 0]], [x, W[g].buf], out, R);
  // fused q/k/v or gate/up matmul (split-K GEMV lost to reduction overhead at K=2048, reverted)
  function fusedMM(pass, w, inBuf, S, outs) {
    const Ntot = w.N0 + w.N1 + w.N2;
    if (useSG && S === 1) {
      const gx = Math.min(Ntot, 65535);
      runWG(pass, 'matmul_split_sg', [['u', w.K], ['u', w.nb], ['u', w.N0], ['u', w.N1], ['u', w.N2], ['u', gx]], [inBuf, w.sign, w.scales], outs, gx, Math.ceil(Ntot / gx));
    } else if (S === 1) {
      const gx = Math.min(Ntot, 65535);                     // no-subgroup decode: workgroup-reduction GEMV
      runWG(pass, 'matmul_split_wg', [['u', w.K], ['u', w.nb], ['u', w.N0], ['u', w.N1], ['u', w.N2], ['u', gx]], [inBuf, w.sign, w.scales], outs, gx, Math.ceil(Ntot / gx));
    } else {
      runIO(pass, 'matmul_split', [['u', S], ['u', w.K], ['u', w.nb], ['u', w.N0], ['u', w.N1], ['u', w.N2]], [inBuf, w.sign, w.scales], outs, S * Ntot);
    }
  }
  // o_proj / down_proj matmul with fused residual add
  function residMM(pass, w, inBuf, resid, S, out) {
    if (useSG && S === 1) {
      const nwg = Math.ceil(w.N / ROWS_MR);                 // multi-row GEMV: ROWS_MR output cols per workgroup
      const gx = Math.min(nwg, 65535);
      runWG(pass, 'matmul_resid_mr_sg', [['u', w.N], ['u', w.K], ['u', w.nb], ['u', gx], ['u', 0], ['u', 0]], [inBuf, w.sign, w.scales, resid], [out], gx, Math.ceil(nwg / gx));
    } else if (S === 1) {
      const gx = Math.min(w.N, 65535);                      // no-subgroup decode: workgroup-reduction GEMV + residual
      runWG(pass, 'matmul_resid_wg', [['u', w.N], ['u', w.K], ['u', w.nb], ['u', gx], ['u', 0], ['u', 0]], [inBuf, w.sign, w.scales, resid], [out], gx, Math.ceil(w.N / gx));
    } else {
      runIO(pass, 'matmul_resid', [['u', S], ['u', w.N], ['u', w.K], ['u', w.nb], ['u', 128], ['u', 0]], [inBuf, w.sign, w.scales, resid], [out], S * w.N);
    }
  }

  function layer(pass, li, h, S, posBase, cos, sin) {
    const Ltot = posBase + S;
    const n1 = actBuf(S * Hd); rms(pass, h, `layers.${li}.input_layernorm`, S, Hd, n1);
    const qkv = W[`layers.${li}.attn.qkv`];

    if (useSG && S === 1 && !FORCE_SLOW) {
      // fused decode path: the dispatch/barrier floor is the dependent-chain length, so we fold
      // copies and elementwise ops into the matmul/norm kernels (14 -> 9 dispatches per layer).
      const q = actBuf(H * Dh), k = actBuf(KV * Dh), v = actBuf(KV * Dh);
      const Ntot = qkv.N0 + qkv.N1 + qkv.N2, gx = Math.min(Ntot, 65535);
      // qkv GEMV into temps; v copied into the V cache (writing v straight into the persistent
      // cache from the matmul's 3rd output got dropped by the driver, so keep the v copy).
      runWG(pass, 'matmul_split_sg',
        [['u', qkv.K], ['u', qkv.nb], ['u', qkv.N0], ['u', qkv.N1], ['u', qkv.N2], ['u', gx]],
        [n1, qkv.sign, qkv.scales], [q, k, v], gx, Math.ceil(Ntot / gx));
      run(pass, 'copy', [['u', KV * Dh], ['u', posBase * KV * Dh], ['u', 0], ['u', 0]], [v], Vc[li], KV * Dh);
      // per-head RMSNorm + RoPE fused; q -> qr, k -> straight into the K cache at this position
      const qr = actBuf(H * Dh);
      runN(pass, 'rmsnorm_rope_sg', [['u', H], ['u', Dh], ['f', A.rms_eps], ['u', 0], ['u', Dh], ['u', 0]],
        [q, W[`layers.${li}.attn.q_norm`].buf, cos, sin], qr, H);
      runN(pass, 'rmsnorm_rope_sg', [['u', KV], ['u', Dh], ['f', A.rms_eps], ['u', posBase * KV * Dh], ['u', Dh], ['u', 0]],
        [k, W[`layers.${li}.attn.k_norm`].buf, cos, sin], Kc[li], KV);
      cap(li, 'qr', qr);
      const att = actBuf(H * Dh);
      runN(pass, 'attention_sg', [['u', 1], ['u', H], ['u', KV], ['u', Dh], ['u', posBase], ['u', Ltot]], [qr, Kc[li], Vc[li]], att, H);
      cap(li, 'att', att);
      const o = W[`layers.${li}.attn.o_proj`], h2 = actBuf(Hd);
      residMM(pass, o, att, h, 1, h2);
      const n2 = actBuf(Hd); rms(pass, h2, `layers.${li}.post_attention_layernorm`, 1, Hd, n2);
      // gate/up GEMV + SwiGLU fused, multi-row (ROWS_MR intermediate cols per workgroup)
      const gu = W[`layers.${li}.mlp.gateup`], sw = actBuf(F), nwgF = Math.ceil(F / ROWS_MR), gxF = Math.min(nwgF, 65535);
      runWG(pass, 'matmul_swiglu_mr_sg', [['u', gu.K], ['u', gu.nb], ['u', F], ['u', gxF], ['u', 0], ['u', 0]],
        [n2, gu.sign, gu.scales], [sw], gxF, Math.ceil(nwgF / gxF));
      cap(li, 'sw', sw);
      const d = W[`layers.${li}.mlp.down_proj`], hn = actBuf(Hd);
      residMM(pass, d, sw, h2, 1, hn);
      return hn;
    }

    // prefill / no-subgroup path: separate kernels (kept verbatim; validates correctness end to end)
    const q = actBuf(S * H * Dh), k = actBuf(S * KV * Dh), v = actBuf(S * KV * Dh);
    fusedMM(pass, qkv, n1, S, [q, k, v]);
    const qn = actBuf(S * H * Dh), kn = actBuf(S * KV * Dh);
    rms(pass, q, `layers.${li}.attn.q_norm`, S * H, Dh, qn);
    rms(pass, k, `layers.${li}.attn.k_norm`, S * KV, Dh, kn);
    const qr = actBuf(S * H * Dh), kr = actBuf(S * KV * Dh);
    run(pass, 'rope', [['u', S], ['u', H], ['u', Dh], ['u', 0]], [qn, cos, sin], qr, S * H * Dh);
    run(pass, 'rope', [['u', S], ['u', KV], ['u', Dh], ['u', 0]], [kn, cos, sin], kr, S * KV * Dh);
    run(pass, 'copy', [['u', S * KV * Dh], ['u', posBase * KV * Dh], ['u', 0], ['u', 0]], [kr], Kc[li], S * KV * Dh);
    run(pass, 'copy', [['u', S * KV * Dh], ['u', posBase * KV * Dh], ['u', 0], ['u', 0]], [v], Vc[li], S * KV * Dh);
    cap(li, 'qr', qr);
    const att = actBuf(S * H * Dh);
    const attF = [['u', S], ['u', H], ['u', KV], ['u', Dh], ['u', posBase], ['u', Ltot]];
    if (useSG) runN(pass, 'attention_sg', attF, [qr, Kc[li], Vc[li]], att, S * H);
    else run(pass, 'attention_cache', attF, [qr, Kc[li], Vc[li]], att, S * H);
    cap(li, 'att', att);
    const o = W[`layers.${li}.attn.o_proj`], h2 = actBuf(S * Hd);
    residMM(pass, o, att, h, S, h2);
    const n2 = actBuf(S * Hd); rms(pass, h2, `layers.${li}.post_attention_layernorm`, S, Hd, n2);
    const gu = W[`layers.${li}.mlp.gateup`], g = actBuf(S * F), u = actBuf(S * F);
    fusedMM(pass, gu, n2, S, [g, u, dummy]);
    const sw = actBuf(S * F); run(pass, 'swiglu', [['u', S * F], ['u', 0], ['u', 0], ['u', 0]], [g, u], sw, S * F);
    cap(li, 'sw', sw);
    const d = W[`layers.${li}.mlp.down_proj`], hn = actBuf(S * Hd);
    residMM(pass, d, sw, h2, S, hn);
    return hn;
  }
  function lmHead(pass, fn, M, out) {
    const lm = W.lm_head;
    if (useSG && M === 1) {
      const gx = Math.min(lm.N, 65535);
      runWG(pass, 'matmul_q2_sg', [['u', lm.N], ['u', lm.K], ['u', lm.nb], ['u', lm.zp], ['u', gx], ['u', 0]], [fn, lm.codes, lm.scales], [out], gx, Math.ceil(lm.N / gx));
    } else if (M === 1) {
      const gx = Math.min(lm.N, 65535);                     // no-subgroup decode: workgroup-reduction 2-bit GEMV
      runWG(pass, 'matmul_q2_wg', [['u', lm.N], ['u', lm.K], ['u', lm.nb], ['u', lm.zp], ['u', gx], ['u', 0]], [fn, lm.codes, lm.scales], [out], gx, Math.ceil(lm.N / gx));
    } else {
      run(pass, 'matmul_q2', [['u', M], ['u', lm.N], ['u', lm.K], ['u', lm.nb], ['u', 128], ['u', lm.zp]], [fn, lm.codes, lm.scales], out, M * lm.N);
    }
  }

  function stack(enc, h, S, posBase) {
    const { cos, sin } = ropeBufs(posBase, S);
    const pass = enc.beginComputePass();
    let cur = h, layer0 = null;
    for (let li = 0; li < A.layers; li++) { cur = layer(pass, li, cur, S, posBase, cos, sin); if (li === 0) layer0 = cur; }
    const fn = actBuf(S * Hd); rms(pass, cur, 'layers.28.final_norm_layernorm', S, Hd, fn);
    pass.end();
    return { fn, layer0 };
  }

  async function forward(ids) {
    const S = ids.length, embedOut = upload(embedDequant(ids), S_ | CD | CS);
    const enc = device.createCommandEncoder();
    const { fn, layer0 } = stack(enc, embedOut, S, 0);
    const logits = device.createBuffer({ size: S * W.lm_head.N * 4, usage: S_ | CS });
    const pass = enc.beginComputePass(); lmHead(pass, fn, S, logits); pass.end();
    device.queue.submit([enc.finish()]); await device.queue.onSubmittedWorkDone();
    return { embed: await readback(embedOut, S * Hd), layer0: await readback(layer0, S * Hd),
             finalnorm: await readback(fn, S * Hd), logits: await readback(logits, S * W.lm_head.N), vocab: W.lm_head.N, S };
  }
  const argmax = (a) => { let bi = 0, bv = -1e30; for (let i = 0; i < a.length; i++) if (a[i] > bv) { bv = a[i]; bi = i; } return bi; };

  async function generate(ids, nTokens, full = null) {
    FULL = full;
    const t0 = performance.now();
    const encP = device.createCommandEncoder();
    const { fn } = stack(encP, upload(embedDequant(ids), S_ | CD), ids.length, 0);
    const lg0 = device.createBuffer({ size: W.lm_head.N * 4, usage: S_ | CS });
    const lastP = actBuf(Hd); encP.copyBufferToBuffer(fn, (ids.length - 1) * Hd * 4, lastP, 0, Hd * 4);
    const passP = encP.beginComputePass(); lmHead(passP, lastP, 1, lg0); passP.end();
    device.queue.submit([encP.finish()]);
    let tok = argmax(await readback(lg0, W.lm_head.N));
    const prefillMs = performance.now() - t0;

    const gen = [tok];
    let recMs = 0, gpuMs = 0, rbMs = 0;
    const t1 = performance.now();
    for (let i = 1; i < nTokens; i++) {
      const pos = ids.length + i - 1;
      let t = performance.now();
      const enc = device.createCommandEncoder();
      const r = stack(enc, upload(embedDequant([tok]), S_ | CD), 1, pos);
      const last = actBuf(Hd); enc.copyBufferToBuffer(r.fn, 0, last, 0, Hd * 4);
      const lg = device.createBuffer({ size: W.lm_head.N * 4, usage: S_ | CS });
      const pass = enc.beginComputePass(); lmHead(pass, last, 1, lg); pass.end();
      device.queue.submit([enc.finish()]);
      recMs += performance.now() - t;
      t = performance.now(); await device.queue.onSubmittedWorkDone(); gpuMs += performance.now() - t;
      t = performance.now(); const logits = await readback(lg, W.lm_head.N); rbMs += performance.now() - t;
      tok = argmax(logits); gen.push(tok);
    }
    const decodeMs = performance.now() - t1, nd = nTokens - 1;
    FULL = null;
    return { prefillMs, decodeMs, tokPerSec: nd / (decodeMs / 1000), tokens: gen, firstArgmax: gen[0], recMs: recMs / nd, gpuMs: gpuMs / nd, rbMs: rbMs / nd };
  }

  // Run ONE decode step at the same position through the fused path and the slow (known-good) path
  // and return layer-0 checkpoints + final norm + logits for each, so a divergence pinpoints the
  // first fused kernel that differs. Both steps write their own cache[pos] then read 0..pos, and
  // neither touches 0..pos-1 (prefill), so they don't interfere -> no cache restore needed.
  async function debugDecode(prefillIds) {
    const encP = device.createCommandEncoder();
    stack(encP, upload(embedDequant(prefillIds), S_ | CD), prefillIds.length, 0);
    device.queue.submit([encP.finish()]); await device.queue.onSubmittedWorkDone();
    const pos = prefillIds.length, tok = prefillIds[prefillIds.length - 1];
    const runStep = async (forceSlow) => {
      FORCE_SLOW = forceSlow; DBG0 = {};
      const enc = device.createCommandEncoder();
      const r = stack(enc, upload(embedDequant([tok]), S_ | CD), 1, pos);
      const lg = device.createBuffer({ size: W.lm_head.N * 4, usage: S_ | CS });
      const pass = enc.beginComputePass(); lmHead(pass, r.fn, 1, lg); pass.end();
      device.queue.submit([enc.finish()]); await device.queue.onSubmittedWorkDone();
      const ck = {};
      for (const [name, b] of Object.entries(DBG0)) ck[name] = await readback(b, b.size / 4);
      // K/V cache slice at the just-written position (layer 0), to isolate k-write vs v-write bugs
      const off = pos * KV * Dh;
      ck.kc = (await readback(Kc[0], MAXSEQ * KV * Dh)).slice(off, off + KV * Dh);
      ck.vc = (await readback(Vc[0], MAXSEQ * KV * Dh)).slice(off, off + KV * Dh);
      ck.fn = await readback(r.fn, Hd);
      ck.logits = await readback(lg, W.lm_head.N);
      FORCE_SLOW = false; DBG0 = null;
      return ck;
    };
    const fast = await runStep(false), slow = await runStep(true);
    return { fast, slow };
  }

  return { device, adapter, forward, generate, debugDecode, useSG, sgSize: sgMax };
}
