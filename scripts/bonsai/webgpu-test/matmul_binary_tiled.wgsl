// Tiled binary matmul: y[M,N] = x[M,K] @ W[N,K]^T, W = (+/-1) * per-block scale.
// 16x16 output tile per workgroup; K processed one 128-block at a time so the per-block
// scale applies cleanly. x and sign bits for the tile are staged in workgroup memory and
// reused across the tile (cuts global traffic ~16x vs the one-thread-per-output kernel).
struct Params { M: u32, N: u32, K: u32, nb: u32, block: u32, _pad: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;         // [M, K]
@group(0) @binding(2) var<storage, read> signbits: array<u32>;  // [N, K/32]
@group(0) @binding(3) var<storage, read> scales: array<f32>;    // [N, nb]
@group(0) @binding(4) var<storage, read_write> y: array<f32>;   // [M, N]

const TM = 16u;
const TN = 16u;
const BK = 128u;   // = block size

var<workgroup> As: array<f32, 2048>;   // [TM][BK] = 16*128
var<workgroup> Bs: array<u32, 64>;     // [TN][BK/32] = 16*4
var<workgroup> Ss: array<f32, 16>;     // [TN]

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let tx = lid.x;
  let ty = lid.y;
  let tid = ty * TN + tx;            // 0..255
  let mBase = wg.y * TM;
  let nBase = wg.x * TN;
  let m = mBase + ty;
  let n = nBase + tx;
  let kWords = p.K / 32u;

  var acc = 0.0;
  let nblocks = p.K / BK;
  for (var b = 0u; b < nblocks; b = b + 1u) {
    // stage x tile [TM, BK]
    for (var i = tid; i < TM * BK; i = i + 256u) {
      let rm = i / BK;
      let ck = i % BK;
      let mm = mBase + rm;
      As[i] = select(0.0, x[mm * p.K + b * BK + ck], mm < p.M);
    }
    // stage sign bits [TN, 4] and scales [TN]
    if (tid < TN * 4u) {
      let rn = tid / 4u;
      let w = tid % 4u;
      let nn = nBase + rn;
      Bs[tid] = select(0u, signbits[nn * kWords + b * 4u + w], nn < p.N);
    }
    if (tid < TN) {
      let nn = nBase + tid;
      Ss[tid] = select(0.0, scales[nn * p.nb + b], nn < p.N);
    }
    workgroupBarrier();

    var bsum = 0.0;
    for (var k = 0u; k < BK; k = k + 1u) {
      let bit = (Bs[tx * 4u + (k >> 5u)] >> (k & 31u)) & 1u;
      let xv = As[ty * BK + k];
      bsum = bsum + select(-xv, xv, bit == 1u);
    }
    acc = acc + bsum * Ss[tx];
    workgroupBarrier();
  }

  if (m < p.M && n < p.N) { y[m * p.N + n] = acc; }
}
