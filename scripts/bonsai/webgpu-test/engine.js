// Bonsai-1.7B WebGPU runtime (P1, correctness-first). Loads weights via the manifest,
// runs the full Qwen3 forward with the validated kernels, returns logits + checkpoints.
// Not optimized: kernels are the dumb-but-correct versions; P2 makes it fast.

const VIEW = { FLOAT: Float32Array, UINT8: Uint8Array, FLOAT16: Uint16Array };
const WGSLS = ['matmul_binary', 'matmul_q2', 'rmsnorm', 'rope', 'swiglu', 'attention', 'add', 'expand_kv'];

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

  // sign table: weight_quant byte -> 8 packed sign bits (bit j = sign of weight j; 1=+1, 0=-1)
  const tgt2 = readRef(manifest.luts.tgt2);                 // [256,2]
  const tgt4 = readRef(manifest.luts.tgt4);                 // [256,4]
  const signTable = new Uint8Array(256);
  for (let b = 0; b < 256; b++) {
    let bits = 0;
    for (let j = 0; j < 8; j++) {
      const byte = tgt2[2 * b + (j >> 2)];
      const code = (byte >> (2 * (j & 3))) & 3;
      bits |= ((code >> 1) & 1) << j;                       // code 3 -> +1 (bit 1), code 1 -> -1 (bit 0)
    }
    signTable[b] = bits;
  }

  // ---- load weights ----
  const W = {};   // logical name -> {sign|codes, scales, N, K, nb, kind, zp?}
  for (const [name, t] of Object.entries(T)) {
    if (t.kind === 'binary') {
      const wq = readRef(t.weight);
      const sign = new Uint8Array(wq.length);
      for (let i = 0; i < wq.length; i++) sign[i] = signTable[wq[i]];
      W[name] = { kind: 'binary', N: t.N, K: t.K, nb: t.K / 128, sign: upload(sign), scales: upload(readRef(t.scales)) };
    } else if (t.kind === 'q2') {              // lm_head: store 2-bit codes (tgt2-expanded)
      const wq = readRef(t.weight);
      const codes = new Uint8Array(wq.length * 2);
      for (let i = 0; i < wq.length; i++) { codes[2 * i] = tgt2[2 * wq[i]]; codes[2 * i + 1] = tgt2[2 * wq[i] + 1]; }
      W[name] = { kind: 'q2', N: t.N, K: t.K, nb: t.K / 128, zp: 2, codes: upload(codes), scales: upload(readRef(t.scales)) };
    } else if (t.kind === 'f32' && t.weight) {   // norm gammas; cos/sin caches are loaded separately below
      W[name] = { kind: 'f32', buf: upload(readRef(t.weight)) };
    }
  }

  // embedding (4-bit) dequant for given ids, on CPU (only S rows)
  const emb = T.embed_tokens;
  const embWq = readRef(emb.weight);             // [vocab, 256] uint8
  const embScales = readRef(emb.scales);         // [vocab, 16] f32
  const embZp = readRef(emb.zp);                 // [vocab, 8] uint8 (4-bit, 2 per byte)
  const cosCache = readRef(T.cos_cache);         // [32768, 64] f32
  const sinCache = readRef(T.sin_cache);

  function embedDequant(ids) {
    const H = A.hidden, out = new Float32Array(ids.length * H);
    for (let r = 0; r < ids.length; r++) {
      const id = ids[r];
      for (let i = 0; i < 256; i++) {                     // 256 src bytes -> 1024 expanded bytes
        const b = embWq[id * 256 + i];
        for (let q = 0; q < 4; q++) {
          const byte = tgt4[4 * b + q];                    // 2 four-bit codes per byte
          const baseK = (i * 4 + q) * 2;
          for (let c = 0; c < 2; c++) {
            const k = baseK + c;
            const code = (byte >> (4 * c)) & 15;
            const blk = (k / 128) | 0;
            const zpByte = embZp[id * 8 + ((blk / 2) | 0)];
            const zp = (zpByte >> (4 * (blk & 1))) & 15;
            out[r * H + k] = (code - zp) * embScales[id * 16 + blk];
          }
        }
      }
    }
    return out;
  }

  function ropeFull(S) {
    const D = A.head_dim, cos = new Float32Array(S * D), sin = new Float32Array(S * D);
    for (let s = 0; s < S; s++) for (let d = 0; d < D; d++) {
      cos[s * D + d] = cosCache[s * 64 + (d % 64)];
      sin[s * D + d] = sinCache[s * 64 + (d % 64)];
    }
    return { cos: upload(cos), sin: upload(sin) };
  }

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

  // ---- forward ----
  async function forward(ids) {
    const S = ids.length, Hd = A.hidden, H = A.heads, KV = A.kv_heads, D = A.head_dim, F = A.intermediate;
    const { cos, sin } = ropeFull(S);
    let h = upload(embedDequant(ids), S_ | CD | CS);       // [S, Hd] (CS so the checkpoint can read it back)
    const embedOut = h;                                    // checkpoint (CPU-built, uploaded)

    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    const run = (name, fields, ins, out, threads) => {
      const pBuf = upload(new Uint8Array(makeParams(fields)), U | CD);
      const entries = [{ binding: 0, resource: { buffer: pBuf } }];
      ins.forEach((b, i) => entries.push({ binding: i + 1, resource: { buffer: b } }));
      entries.push({ binding: ins.length + 1, resource: { buffer: out } });
      pass.setPipeline(pipelines[name]);
      pass.setBindGroup(0, device.createBindGroup({ layout: pipelines[name].getBindGroupLayout(0), entries }));
      pass.dispatchWorkgroups(Math.ceil(threads / 64));
    };
    const mm = (wname, x, M, out) => {                     // binary matmul x[M,K]@W.T -> out[M,N]
      const w = W[wname];
      run('matmul_binary', [['u', M], ['u', w.N], ['u', w.K], ['u', w.nb], ['u', 128], ['u', 0]], [x, w.sign, w.scales], out, M * w.N);
    };
    const rms = (x, gammaName, R, Dn, out) =>
      run('rmsnorm', [['u', R], ['u', Dn], ['f', A.rms_eps], ['u', 0]], [x, W[gammaName].buf], out, R);
    const addv = (a, b, n, out) => run('add', [['u', n], ['u', 0], ['u', 0], ['u', 0]], [a, b], out, n);

    let layer0 = null;
    for (let li = 0; li < A.layers; li++) {
      const n1 = actBuf(S * Hd);
      rms(h, `layers.${li}.input_layernorm`, S, Hd, n1);
      const q = actBuf(S * H * D), k = actBuf(S * KV * D), v = actBuf(S * KV * D);
      mm(`layers.${li}.attn.q_proj`, n1, S, q);
      mm(`layers.${li}.attn.k_proj`, n1, S, k);
      mm(`layers.${li}.attn.v_proj`, n1, S, v);
      const qn = actBuf(S * H * D), kn = actBuf(S * KV * D);
      rms(q, `layers.${li}.attn.q_norm`, S * H, D, qn);
      rms(k, `layers.${li}.attn.k_norm`, S * KV, D, kn);
      const qr = actBuf(S * H * D), kr = actBuf(S * KV * D);
      run('rope', [['u', S], ['u', H], ['u', D], ['u', 0]], [qn, cos, sin], qr, S * H * D);
      run('rope', [['u', S], ['u', KV], ['u', D], ['u', 0]], [kn, cos, sin], kr, S * KV * D);
      const kx = actBuf(S * H * D), vx = actBuf(S * H * D);
      run('expand_kv', [['u', S], ['u', H], ['u', KV], ['u', D]], [kr], kx, S * H * D);
      run('expand_kv', [['u', S], ['u', H], ['u', KV], ['u', D]], [v], vx, S * H * D);
      const att = actBuf(S * H * D);
      run('attention', [['u', S], ['u', H], ['u', D], ['u', 0]], [qr, kx, vx], att, S * H);
      const ao = actBuf(S * Hd);
      mm(`layers.${li}.attn.o_proj`, att, S, ao);
      const h2 = actBuf(S * Hd);
      addv(h, ao, S * Hd, h2);
      const n2 = actBuf(S * Hd);
      rms(h2, `layers.${li}.post_attention_layernorm`, S, Hd, n2);
      const g = actBuf(S * F), u = actBuf(S * F);
      mm(`layers.${li}.mlp.gate_proj`, n2, S, g);
      mm(`layers.${li}.mlp.up_proj`, n2, S, u);
      const sw = actBuf(S * F);
      run('swiglu', [['u', S * F], ['u', 0], ['u', 0], ['u', 0]], [g, u], sw, S * F);
      const mo = actBuf(S * Hd);
      mm(`layers.${li}.mlp.down_proj`, sw, S, mo);
      const hn = actBuf(S * Hd);
      addv(h2, mo, S * Hd, hn);
      h = hn;
      if (li === 0) layer0 = h;
    }
    const fn = actBuf(S * Hd);
    rms(h, 'layers.28.final_norm_layernorm', S, Hd, fn);
    const lm = W.lm_head, logits = device.createBuffer({ size: S * lm.N * 4, usage: S_ | CS });
    run('matmul_q2', [['u', S], ['u', lm.N], ['u', lm.K], ['u', lm.nb], ['u', 128], ['u', lm.zp]], [fn, lm.codes, lm.scales], logits, S * lm.N);
    pass.end();
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    return {
      embed: await readback(embedOut, S * Hd),
      layer0: await readback(layer0, S * Hd),
      finalnorm: await readback(fn, S * Hd),
      logits: await readback(logits, S * lm.N),
      vocab: lm.N, S,
    };
  }

  return { device, adapter, forward };
}
