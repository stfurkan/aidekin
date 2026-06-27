// GQA expand: replicate each KV head to H/KV query heads. in [S, KV, D] -> out [S, H, D],
// out[s, h, d] = in[s, h / (H/KV), d]. One invocation per output element.
struct Params { S: u32, H: u32, KV: u32, D: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> inp: array<f32>;     // [S, KV, D]
@group(0) @binding(2) var<storage, read_write> out: array<f32>; // [S, H, D]

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= p.S * p.H * p.D) { return; }
  let d = idx % p.D;
  let h = (idx / p.D) % p.H;
  let s = idx / (p.H * p.D);
  let kvh = h / (p.H / p.KV);
  out[idx] = inp[(s * p.KV + kvh) * p.D + d];
}
