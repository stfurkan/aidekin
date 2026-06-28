// Tiled register-blocked binary GEMM with fused residual, for PREFILL (M>1):
//   y[M,N] = x[M,K] @ W[N,K]^T + resid[M,N],  W binary {-1,+1} sign-packed, per-128-block fp32 scale.
// 64x64 output tile per workgroup, 16x16 threads each computing a 4x4 register tile, BK=16 K-step.
// Shared memory stages the activation tile and the decoded+scaled weight tile, killing the scalar
// path's redundant re-reads (each weight row was re-read M times). No subgroup ops -> all devices.
// Scale folds into the weight tile (BK=16 divides 128, so each K-tile lies in one scale block).
// Near-bit-exact: f32 accumulation, but tiled K-order differs from the scalar path in the last ULPs.
const BM: u32 = 64u;
const BN: u32 = 64u;
const BK: u32 = 16u;
struct Params { M: u32, N: u32, K: u32, nb: u32, _0: u32, _1: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;        // [M, K]
@group(0) @binding(2) var<storage, read> signbits: array<u32>; // [N, K/32]
@group(0) @binding(3) var<storage, read> scales: array<f32>;   // [N, nb]
@group(0) @binding(4) var<storage, read> resid: array<f32>;    // [M, N]
@group(0) @binding(5) var<storage, read_write> y: array<f32>;  // [M, N]

var<workgroup> xs: array<f32, 1024>;   // BM*BK
var<workgroup> ws: array<f32, 1024>;   // BK*BN

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  let tileM = wg.y * BM;
  let tileN = wg.x * BN;
  let tr = (tid / 16u) * 4u;             // this thread's 4x4 tile row base within the workgroup tile
  let tc = (tid % 16u) * 4u;
  var acc: array<f32, 16>;
  for (var i = 0u; i < 16u; i = i + 1u) { acc[i] = 0.0; }

  let Ksteps = p.K / BK;
  for (var ks = 0u; ks < Ksteps; ks = ks + 1u) {
    let k0 = ks * BK;
    for (var e = tid; e < BM * BK; e = e + 256u) {            // stage activation tile
      let r = e / BK; let c = e % BK; let gm = tileM + r;
      xs[e] = select(0.0, x[gm * p.K + (k0 + c)], gm < p.M);
    }
    for (var e = tid; e < BK * BN; e = e + 256u) {            // stage decoded+scaled weight tile
      let c = e / BN; let nloc = e % BN; let gn = tileN + nloc; let gk = k0 + c;
      var wv = 0.0;
      if (gn < p.N) {
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
    if (gm < p.M) {
      for (var tn = 0u; tn < 4u; tn = tn + 1u) {
        let gn = tileN + tc + tn;
        if (gn < p.N) { let idx = gm * p.N + gn; y[idx] = acc[tm * 4u + tn] + resid[idx]; }
      }
    }
  }
}
