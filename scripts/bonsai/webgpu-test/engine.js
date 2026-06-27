// Bonsai-1.7B WebGPU runtime (P2, correctness-first + generation). Loads weights via the
// manifest, runs the Qwen3 forward with the validated kernels, keeps a persistent KV cache,
// and generates autoregressively. Kernels are still the dumb-but-correct versions (P2 makes
// them fast); this exists to measure real decode tok/s.

const VIEW = { FLOAT: Float32Array, UINT8: Uint8Array, FLOAT16: Uint16Array };
const WGSLS = ['matmul_binary', 'matmul_binary_vec4', 'matmul_q2', 'rmsnorm', 'rope', 'swiglu', 'attention_cache', 'add', 'copy'];
const MAXSEQ = 256;

function makeParams(fields) {
  const ab = new ArrayBuffer(Math.ceil(fields.length / 4) * 16);
  const dv = new DataView(ab);
  fields.forEach(([t, v], i) => t === 'f' ? dv.setFloat32(i * 4, v, true) : dv.setUint32(i * 4, v >>> 0, true));
  return ab;
}

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
  const device = await adapter.requestDevice();
  const pipelines = {};
  for (const name of WGSLS) {
    const code = await (await fetch(`./${name}.wgsl`)).text();
    pipelines[name] = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'main' } });
  }

  const S_ = GPUBufferUsage.STORAGE, CD = GPUBufferUsage.COPY_DST, CS = GPUBufferUsage.COPY_SRC, U = GPUBufferUsage.UNIFORM;
  const upload = (typed, usage = S_ | CD) => { const b = device.createBuffer({ size: typed.byteLength, usage }); device.queue.writeBuffer(b, 0, typed); return b; };
  const actBuf = (n) => device.createBuffer({ size: n * 4, usage: S_ | CS | CD });

  // sign table: weight_quant byte -> 8 packed sign bits (bit j: 1=+1, 0=-1)
  const tgt2 = readRef(manifest.luts.tgt2), tgt4 = readRef(manifest.luts.tgt4);
  const signTable = new Uint8Array(256);
  for (let b = 0; b < 256; b++) {
    let bits = 0;
    for (let j = 0; j < 8; j++) bits |= (((tgt2[2 * b + (j >> 2)] >> (2 * (j & 3))) & 3) >> 1 & 1) << j;
    signTable[b] = bits;
  }

  const W = {};
  for (const [name, t] of Object.entries(T)) {
    if (t.kind === 'binary') {
      const wq = readRef(t.weight), sign = new Uint8Array(wq.length);
      for (let i = 0; i < wq.length; i++) sign[i] = signTable[wq[i]];
      W[name] = { N: t.N, K: t.K, nb: t.K / 128, sign: upload(sign), scales: upload(readRef(t.scales)) };
    } else if (t.kind === 'q2') {
      const wq = readRef(t.weight), codes = new Uint8Array(wq.length * 2);
      for (let i = 0; i < wq.length; i++) { codes[2 * i] = tgt2[2 * wq[i]]; codes[2 * i + 1] = tgt2[2 * wq[i] + 1]; }
      W[name] = { N: t.N, K: t.K, nb: t.K / 128, zp: 2, codes: upload(codes), scales: upload(readRef(t.scales)) };
    } else if (t.kind === 'f32' && t.weight) {
      W[name] = { buf: upload(readRef(t.weight)) };
    }
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
    for (let s = 0; s < S; s++) for (let d = 0; d < D; d++) {
      cos[s * D + d] = cosCache[(posBase + s) * 64 + (d % 64)];
      sin[s * D + d] = sinCache[(posBase + s) * 64 + (d % 64)];
    }
    return { cos: upload(cos), sin: upload(sin) };
  }

  // per-layer persistent KV cache: [MAXSEQ, KV, D]
  const KV = A.kv_heads, Dh = A.head_dim, Hd = A.hidden, H = A.heads, F = A.intermediate;
  const Kc = [], Vc = [];
  for (let li = 0; li < A.layers; li++) { Kc.push(actBuf(MAXSEQ * KV * Dh)); Vc.push(actBuf(MAXSEQ * KV * Dh)); }

  async function readback(buf, n) {
    const rb = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.MAP_READ | CD });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(buf, 0, rb, 0, n * 4);
    device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(rb.getMappedRange().slice(0));
    rb.unmap();
    return out;
  }

  function run(pass, name, fields, ins, out, threads) {
    const entries = [{ binding: 0, resource: { buffer: upload(new Uint8Array(makeParams(fields)), U | CD) } }];
    ins.forEach((b, i) => entries.push({ binding: i + 1, resource: { buffer: b } }));
    entries.push({ binding: ins.length + 1, resource: { buffer: out } });
    pass.setPipeline(pipelines[name]);
    pass.setBindGroup(0, device.createBindGroup({ layout: pipelines[name].getBindGroupLayout(0), entries }));
    pass.dispatchWorkgroups(Math.ceil(threads / 64));
  }
  const mm = (pass, wname, x, M, out) => { const w = W[wname]; run(pass, 'matmul_binary_vec4', [['u', M], ['u', w.N], ['u', w.K], ['u', w.nb], ['u', 128], ['u', 0]], [x, w.sign, w.scales], out, M * w.N); };
  const rms = (pass, x, g, R, Dn, out) => run(pass, 'rmsnorm', [['u', R], ['u', Dn], ['f', A.rms_eps], ['u', 0]], [x, W[g].buf], out, R);

  // one decoder layer; reads/writes the KV cache. posBase = absolute position of row 0.
  function layer(pass, li, h, S, posBase, cos, sin) {
    const Ltot = posBase + S;
    const n1 = actBuf(S * Hd); rms(pass, h, `layers.${li}.input_layernorm`, S, Hd, n1);
    const q = actBuf(S * H * Dh), k = actBuf(S * KV * Dh), v = actBuf(S * KV * Dh);
    mm(pass, `layers.${li}.attn.q_proj`, n1, S, q);
    mm(pass, `layers.${li}.attn.k_proj`, n1, S, k);
    mm(pass, `layers.${li}.attn.v_proj`, n1, S, v);
    const qn = actBuf(S * H * Dh), kn = actBuf(S * KV * Dh);
    rms(pass, q, `layers.${li}.attn.q_norm`, S * H, Dh, qn);
    rms(pass, k, `layers.${li}.attn.k_norm`, S * KV, Dh, kn);
    const qr = actBuf(S * H * Dh), kr = actBuf(S * KV * Dh);
    run(pass, 'rope', [['u', S], ['u', H], ['u', Dh], ['u', 0]], [qn, cos, sin], qr, S * H * Dh);
    run(pass, 'rope', [['u', S], ['u', KV], ['u', Dh], ['u', 0]], [kn, cos, sin], kr, S * KV * Dh);
    // append k,v to cache at posBase
    run(pass, 'copy', [['u', S * KV * Dh], ['u', posBase * KV * Dh], ['u', 0], ['u', 0]], [kr], Kc[li], S * KV * Dh);
    run(pass, 'copy', [['u', S * KV * Dh], ['u', posBase * KV * Dh], ['u', 0], ['u', 0]], [v], Vc[li], S * KV * Dh);
    const att = actBuf(S * H * Dh);
    run(pass, 'attention_cache', [['u', S], ['u', H], ['u', KV], ['u', Dh], ['u', posBase], ['u', Ltot]], [qr, Kc[li], Vc[li]], att, S * H);
    const ao = actBuf(S * Hd); mm(pass, `layers.${li}.attn.o_proj`, att, S, ao);
    const h2 = actBuf(S * Hd); run(pass, 'add', [['u', S * Hd], ['u', 0], ['u', 0], ['u', 0]], [h, ao], h2, S * Hd);
    const n2 = actBuf(S * Hd); rms(pass, h2, `layers.${li}.post_attention_layernorm`, S, Hd, n2);
    const g = actBuf(S * F), u = actBuf(S * F);
    mm(pass, `layers.${li}.mlp.gate_proj`, n2, S, g);
    mm(pass, `layers.${li}.mlp.up_proj`, n2, S, u);
    const sw = actBuf(S * F); run(pass, 'swiglu', [['u', S * F], ['u', 0], ['u', 0], ['u', 0]], [g, u], sw, S * F);
    const mo = actBuf(S * Hd); mm(pass, `layers.${li}.mlp.down_proj`, sw, S, mo);
    const hn = actBuf(S * Hd); run(pass, 'add', [['u', S * Hd], ['u', 0], ['u', 0], ['u', 0]], [h2, mo], hn, S * Hd);
    return hn;
  }

  function lmHead(pass, fn, M, out) { const lm = W.lm_head; run(pass, 'matmul_q2', [['u', M], ['u', lm.N], ['u', lm.K], ['u', lm.nb], ['u', 128], ['u', lm.zp]], [fn, lm.codes, lm.scales], out, M * lm.N); }

  // run the layer stack over S tokens starting at posBase; returns the final-norm buffer.
  function stack(enc, h, S, posBase) {
    const { cos, sin } = ropeBufs(posBase, S);
    const pass = enc.beginComputePass();
    let cur = h, layer0 = null;
    for (let li = 0; li < A.layers; li++) { cur = layer(pass, li, cur, S, posBase, cos, sin); if (li === 0) layer0 = cur; }
    const fn = actBuf(S * Hd); rms(pass, cur, 'layers.28.final_norm_layernorm', S, Hd, fn);
    pass.end();
    return { fn, layer0 };
  }

  // prefill with checkpoints (for the correctness test)
  async function forward(ids) {
    const S = ids.length, embedOut = upload(embedDequant(ids), S_ | CD | CS);
    const enc = device.createCommandEncoder();
    const { fn, layer0 } = stack(enc, embedOut, S, 0);
    const logits = device.createBuffer({ size: S * W.lm_head.N * 4, usage: S_ | CS });
    const pass = enc.beginComputePass(); lmHead(pass, fn, S, logits); pass.end();
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    return { embed: await readback(embedOut, S * Hd), layer0: await readback(layer0, S * Hd),
             finalnorm: await readback(fn, S * Hd), logits: await readback(logits, S * W.lm_head.N), vocab: W.lm_head.N, S };
  }

  async function lastLogits(fn, M) {
    // logits for the last row only (M=1 over the last position of fn)
    const last = actBuf(Hd);
    const enc0 = device.createCommandEncoder(); enc0.copyBufferToBuffer(fn, (M - 1) * Hd * 4, last, 0, Hd * 4); device.queue.submit([enc0.finish()]);
    const logits = device.createBuffer({ size: W.lm_head.N * 4, usage: S_ | CS });
    const enc = device.createCommandEncoder(); const pass = enc.beginComputePass(); lmHead(pass, last, 1, logits); pass.end(); device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    return readback(logits, W.lm_head.N);
  }
  const argmax = (a) => { let bi = 0, bv = -1e30; for (let i = 0; i < a.length; i++) if (a[i] > bv) { bv = a[i]; bi = i; } return bi; };

  // prefill the prompt then generate nTokens greedily; returns timing + tokens.
  async function generate(ids, nTokens) {
    const t0 = performance.now();
    const encP = device.createCommandEncoder();
    const { fn } = stack(encP, upload(embedDequant(ids), S_ | CD), ids.length, 0);
    device.queue.submit([encP.finish()]);
    let logits = await lastLogits(fn, ids.length);
    let tok = argmax(logits);
    const prefillMs = performance.now() - t0;

    const gen = [tok];
    const t1 = performance.now();
    for (let i = 1; i < nTokens; i++) {
      const pos = ids.length + i - 1;
      const enc = device.createCommandEncoder();
      const r = stack(enc, upload(embedDequant([tok]), S_ | CD), 1, pos);
      device.queue.submit([enc.finish()]);
      logits = await lastLogits(r.fn, 1);
      tok = argmax(logits);
      gen.push(tok);
    }
    const decodeMs = performance.now() - t1;
    return { prefillMs, decodeMs, tokPerSec: (nTokens - 1) / (decodeMs / 1000), tokens: gen, firstArgmax: gen[0] };
  }

  return { device, adapter, forward, generate };
}
