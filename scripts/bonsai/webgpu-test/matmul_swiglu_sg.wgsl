// Fused gate/up GEMV + SwiGLU for decode (M=1). The gate/up weights are concatenated
// (gate rows [0,F), up rows [F,2F)); one subgroup (= one workgroup) computes BOTH g[n] and
// u[n] for the same intermediate index n, then writes silu(g)*u directly. Removes the separate
// swiglu dispatch and halves the mid-MLP writes (F floats out instead of 2F).
enable subgroups;
override SG: u32 = 32u;
struct Params { K: u32, nb: u32, F: u32, gridX: u32, _p0: u32, _p1: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;   // [K/4]
@group(0) @binding(2) var<storage, read> signbits: array<u32>;  // [2F, K/32]
@group(0) @binding(3) var<storage, read> scales: array<f32>;    // [2F, nb]
@group(0) @binding(4) var<storage, read_write> y: array<f32>;   // [F]

@compute @workgroup_size(SG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
  let n = wg.y * p.gridX + wg.x;
  if (n >= p.F) { return; }
  let Kvec = p.K / 4u;
  let gRow = n * (p.K / 32u);
  let uRow = (p.F + n) * (p.K / 32u);
  let gS = n * p.nb;
  let uS = (p.F + n) * p.nb;

  var gAcc = 0.0;
  var uAcc = 0.0;
  for (var gi = lane; gi < Kvec; gi = gi + SG) {
    let k = gi * 4u;
    let blk = k / 128u;
    let xv = x[gi];
    let gw = (signbits[gRow + (k >> 5u)] >> (k & 31u)) & 0xfu;
    let gv = vec4<f32>(select(-1.0, 1.0, (gw & 1u) != 0u), select(-1.0, 1.0, (gw & 2u) != 0u),
                       select(-1.0, 1.0, (gw & 4u) != 0u), select(-1.0, 1.0, (gw & 8u) != 0u));
    gAcc = gAcc + dot(xv, gv) * scales[gS + blk];
    let uw = (signbits[uRow + (k >> 5u)] >> (k & 31u)) & 0xfu;
    let uv = vec4<f32>(select(-1.0, 1.0, (uw & 1u) != 0u), select(-1.0, 1.0, (uw & 2u) != 0u),
                       select(-1.0, 1.0, (uw & 4u) != 0u), select(-1.0, 1.0, (uw & 8u) != 0u));
    uAcc = uAcc + dot(xv, uv) * scales[uS + blk];
  }
  let g = subgroupAdd(gAcc);
  let u = subgroupAdd(uAcc);
  if (lane == 0u) { y[n] = (g / (1.0 + exp(-g))) * u; }
}
