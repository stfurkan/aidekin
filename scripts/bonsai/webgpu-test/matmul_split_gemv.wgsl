// Split-K GEMV for decode (M=1): one workgroup (128 threads) per output column, threads
// split the K dimension and reduce in workgroup memory. Fused: routes to out0/out1/out2 by
// column range (qkv or gate/up). Massively more threads than one-thread-per-output, which is
// what decode (matrix-vector) needs to fill the GPU. 2D dispatch (gridX) since N can exceed 65535.
struct Params { K: u32, nb: u32, N0: u32, N1: u32, N2: u32, gridX: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;   // [K/4]
@group(0) @binding(2) var<storage, read> signbits: array<u32>;  // [N0+N1+N2, K/32]
@group(0) @binding(3) var<storage, read> scales: array<f32>;    // [N0+N1+N2, nb]
@group(0) @binding(4) var<storage, read_write> out0: array<f32>;
@group(0) @binding(5) var<storage, read_write> out1: array<f32>;
@group(0) @binding(6) var<storage, read_write> out2: array<f32>;

const T = 128u;
var<workgroup> red: array<f32, 128>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let Ntot = p.N0 + p.N1 + p.N2;
  let n = wg.y * p.gridX + wg.x;
  if (n >= Ntot) { return; }
  let tid = lid.x;
  let Kvec = p.K / 4u;
  let wRow = n * (p.K / 32u);
  let sbase = n * p.nb;

  var acc = 0.0;
  for (var gi = tid; gi < Kvec; gi = gi + T) {
    let k = gi * 4u;
    let word = signbits[wRow + (k >> 5u)];
    let bits4 = (word >> (k & 31u)) & 0xfu;
    let sv = vec4<f32>(select(-1.0, 1.0, (bits4 & 1u) != 0u), select(-1.0, 1.0, (bits4 & 2u) != 0u),
                       select(-1.0, 1.0, (bits4 & 4u) != 0u), select(-1.0, 1.0, (bits4 & 8u) != 0u));
    acc = acc + dot(x[gi], sv) * scales[sbase + (k / 128u)];
  }
  red[tid] = acc;
  workgroupBarrier();
  for (var s = 64u; s > 0u; s = s >> 1u) { if (tid < s) { red[tid] = red[tid] + red[tid + s]; } workgroupBarrier(); }

  if (tid == 0u) {
    if (n < p.N0) { out0[n] = red[0]; }
    else if (n < p.N0 + p.N1) { out1[n - p.N0] = red[0]; }
    else { out2[n - p.N0 - p.N1] = red[0]; }
  }
}
