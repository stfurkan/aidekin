// Causal GQA attention, per (query position, head). q/k/v are [S, H, D] with k/v already
// expanded to H heads. scale = 1/sqrt(D). One invocation per (s, h); head_dim capped at 128.
struct Params { S: u32, H: u32, D: u32, _pad: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;       // [S, H, D]
@group(0) @binding(2) var<storage, read> k: array<f32>;       // [S, H, D]
@group(0) @binding(3) var<storage, read> v: array<f32>;       // [S, H, D]
@group(0) @binding(4) var<storage, read_write> y: array<f32>; // [S, H, D]

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= p.S * p.H) { return; }
  let h = idx % p.H;
  let s = idx / p.H;
  let invscale = 1.0 / sqrt(f32(p.D));
  let qbase = (s * p.H + h) * p.D;

  // pass 1: max score over j <= s
  var m = -1e30;
  for (var j = 0u; j <= s; j = j + 1u) {
    let kbase = (j * p.H + h) * p.D;
    var dot = 0.0;
    for (var d = 0u; d < p.D; d = d + 1u) { dot = dot + q[qbase + d] * k[kbase + d]; }
    m = max(m, dot * invscale);
  }

  // pass 2: softmax-weighted sum of v
  var acc: array<f32, 128>;
  for (var d = 0u; d < p.D; d = d + 1u) { acc[d] = 0.0; }
  var denom = 0.0;
  for (var j = 0u; j <= s; j = j + 1u) {
    let kbase = (j * p.H + h) * p.D;
    var dot = 0.0;
    for (var d = 0u; d < p.D; d = d + 1u) { dot = dot + q[qbase + d] * k[kbase + d]; }
    let w = exp(dot * invscale - m);
    denom = denom + w;
    let vbase = (j * p.H + h) * p.D;
    for (var d = 0u; d < p.D; d = d + 1u) { acc[d] = acc[d] + w * v[vbase + d]; }
  }

  let obase = (s * p.H + h) * p.D;
  for (var d = 0u; d < p.D; d = d + 1u) { y[obase + d] = acc[d] / denom; }
}
