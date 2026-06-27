// Binary matmul, n-major / small-M batched: one thread per output column n, reads each
// weight bit ONCE and applies it across all M rows (the dumb kernel re-read weights M times).
// No barriers / no shared memory. Assumes M <= 16 (true for short prefills; M-tiling is a
// later step for long prompts). y[M,N] = x[M,K] @ W[N,K]^T, W = (+/-1) * per-block scale.
struct Params { M: u32, N: u32, K: u32, nb: u32, block: u32, _pad: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;         // [M, K]
@group(0) @binding(2) var<storage, read> signbits: array<u32>;  // [N, K/32]
@group(0) @binding(3) var<storage, read> scales: array<f32>;    // [N, nb]
@group(0) @binding(4) var<storage, read_write> y: array<f32>;   // [M, N]

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  if (n >= p.N) { return; }
  let wbase = n * (p.K / 32u);
  let sbase = n * p.nb;

  var acc: array<f32, 16>;
  for (var m = 0u; m < p.M; m = m + 1u) { acc[m] = 0.0; }

  for (var b = 0u; b < p.nb; b = b + 1u) {
    var bsum: array<f32, 16>;
    for (var m = 0u; m < p.M; m = m + 1u) { bsum[m] = 0.0; }
    let k0 = b * p.block;
    for (var j = 0u; j < p.block; j = j + 1u) {
      let k = k0 + j;
      let bit = (signbits[wbase + (k >> 5u)] >> (k & 31u)) & 1u;   // read weight bit once
      let s = select(-1.0, 1.0, bit == 1u);
      for (var m = 0u; m < p.M; m = m + 1u) { bsum[m] = bsum[m] + s * x[m * p.K + k]; }
    }
    let sc = scales[sbase + b];
    for (var m = 0u; m < p.M; m = m + 1u) { acc[m] = acc[m] + bsum[m] * sc; }
  }

  for (var m = 0u; m < p.M; m = m + 1u) { y[m * p.N + n] = acc[m]; }
}
