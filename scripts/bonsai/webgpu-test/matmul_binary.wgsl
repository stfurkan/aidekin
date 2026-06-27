// Binary (1-bit) matmul: y[M,N] = x[M,K] @ W[N,K]^T, where W[n,k] = (+/-1) * scale[n, k/block].
// Correctness-first reference kernel (one invocation per output element, fp32). Slow on
// purpose; P2 optimizes it. Signs are packed 32-per-u32 (bit 1 = +1, bit 0 = -1).

struct Params { M: u32, N: u32, K: u32, nb: u32, block: u32, _pad: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;         // [M, K]
@group(0) @binding(2) var<storage, read> signbits: array<u32>;  // [N, K/32]
@group(0) @binding(3) var<storage, read> scales: array<f32>;    // [N, nb]
@group(0) @binding(4) var<storage, read_write> y: array<f32>;   // [M, N]

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= p.M * p.N) { return; }
  let m = idx / p.N;
  let n = idx % p.N;
  let xbase = m * p.K;
  let wbase = n * (p.K / 32u);
  let sbase = n * p.nb;

  var acc = 0.0;
  for (var b = 0u; b < p.nb; b = b + 1u) {
    var bsum = 0.0;
    let k0 = b * p.block;
    for (var j = 0u; j < p.block; j = j + 1u) {
      let k = k0 + j;
      let bit = (signbits[wbase + (k >> 5u)] >> (k & 31u)) & 1u;
      let xv = x[xbase + k];
      bsum = bsum + select(-xv, xv, bit == 1u);
    }
    acc = acc + bsum * scales[sbase + b];
  }
  y[idx] = acc;
}
