// Tiled register-blocked binary GEMM writing to 3 output buffers (qkv or gate/up), for PREFILL (M>1).
// Weights concatenated along N (rows N0|N1|N2); each 64-wide output tile lies entirely in one range
// (N0,N1 are multiples of 64), so a workgroup routes its whole tile to out0/out1/out2 cleanly.
// Same 64x64 tile / 16x16 threads / 4x4 register tile / BK=16 design as matmul_resid_tiled.
const BM: u32 = 64u;
const BN: u32 = 64u;
const BK: u32 = 16u;
struct Params { M: u32, K: u32, nb: u32, N0: u32, N1: u32, N2: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;        // [M, K]
@group(0) @binding(2) var<storage, read> signbits: array<u32>; // [N0+N1+N2, K/32]
@group(0) @binding(3) var<storage, read> scales: array<f32>;   // [N0+N1+N2, nb]
@group(0) @binding(4) var<storage, read_write> out0: array<f32>;
@group(0) @binding(5) var<storage, read_write> out1: array<f32>;
@group(0) @binding(6) var<storage, read_write> out2: array<f32>;

var<workgroup> xs: array<f32, 1024>;
var<workgroup> ws: array<f32, 1024>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let Ntot = p.N0 + p.N1 + p.N2;
  let tid = lid.x;
  let tileM = wg.y * BM;
  let tileN = wg.x * BN;
  let tr = (tid / 16u) * 4u;
  let tc = (tid % 16u) * 4u;
  var acc: array<f32, 16>;
  for (var i = 0u; i < 16u; i = i + 1u) { acc[i] = 0.0; }

  let Ksteps = p.K / BK;
  for (var ks = 0u; ks < Ksteps; ks = ks + 1u) {
    let k0 = ks * BK;
    for (var e = tid; e < BM * BK; e = e + 256u) {
      let r = e / BK; let c = e % BK; let gm = tileM + r;
      xs[e] = select(0.0, x[gm * p.K + (k0 + c)], gm < p.M);
    }
    for (var e = tid; e < BK * BN; e = e + 256u) {
      let c = e / BN; let nloc = e % BN; let gn = tileN + nloc; let gk = k0 + c;
      var wv = 0.0;
      if (gn < Ntot) {
        let bit = (signbits[gn * (p.K / 32u) + (gk >> 5u)] >> (gk & 31u)) & 1u;
        let s = scales[gn * p.nb + (gk / 128u)];
        wv = select(-s, s, bit != 0u);
      }
      ws[c * BN + nloc] = wv;
    }
    workgroupBarrier();
    for (var kk = 0u; kk < BK; kk = kk + 1u) {
      var xr: array<f32, 4>;
      for (var tm = 0u; tm < 4u; tm = tm + 1u) { xr[tm] = xs[(tr + tm) * BK + kk]; }
      for (var tn = 0u; tn < 4u; tn = tn + 1u) {
        let wv = ws[kk * BN + (tc + tn)];
        for (var tm = 0u; tm < 4u; tm = tm + 1u) { acc[tm * 4u + tn] = acc[tm * 4u + tn] + xr[tm] * wv; }
      }
    }
    workgroupBarrier();
  }

  for (var tm = 0u; tm < 4u; tm = tm + 1u) {
    let gm = tileM + tr + tm;
    if (gm >= p.M) { continue; }
    for (var tn = 0u; tn < 4u; tn = tn + 1u) {
      let gn = tileN + tc + tn;
      if (gn >= Ntot) { continue; }
      let v = acc[tm * 4u + tn];
      if (gn < p.N0) { out0[gm * p.N0 + gn] = v; }
      else if (gn < p.N0 + p.N1) { out1[gm * p.N1 + (gn - p.N0)] = v; }
      else { out2[gm * p.N2 + (gn - p.N0 - p.N1)] = v; }
    }
  }
}
