#!/usr/bin/env python3
"""Generate GPU kernel test fixtures from the verified weights (P1).

Emits little-endian binary blobs + fixture.json that the WebGPU test harness loads,
so each WGSL kernel can be checked against a known-correct result without a GPU here.

Currently emits the binary-matmul fixture for layers.0.attn.q_proj:
  x [M,K] fp32 (seeded), signbits [N, K/32] u32 (1=+1, 0=-1), scales [N, nb] fp32,
  expected [M,N] fp32 = x @ W.T  where W = (+/-1) * per-block scale.

Usage: python make_fixtures.py --work <dir> [--out <dir>] [--m 8]
"""
import argparse
import json
import os

import numpy as np

NP_DT = {"FLOAT": np.float32, "FLOAT16": np.float16, "UINT8": np.uint8, "INT64": np.int64}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", default="scripts/bonsai/webgpu-test/fixtures")
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--tensor", default="layers.0.attn.q_proj")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    M = json.load(open(f"{args.work}/manifest.json"))
    data = open(f"{args.work}/{M['data_file']}", "rb")
    aux = open(f"{args.work}/{M['aux_file']}", "rb").read()

    def raw(ref: dict) -> np.ndarray:
        if ref["src"] == "aux":
            b = aux[ref["off"]:ref["off"] + ref["len"]]
        else:
            data.seek(ref["off"])
            b = data.read(ref["len"])
        return np.frombuffer(b, NP_DT[ref["dtype"]])

    tgt2 = raw(M["luts"]["tgt2"]).reshape(256, 2)
    t = M["tensors"][args.tensor]
    N, K = t["N"], t["K"]
    nb = K // 128

    wq = raw(t["weight"]).reshape(N, K // 8)
    exp = tgt2[wq].reshape(N, K // 4)
    codes = np.empty((N, K), np.uint8)
    for k in range(4):
        codes[:, k::4] = (exp >> (2 * k)) & 3
    bits = ((codes >> 1) & 1).astype(np.uint32)              # 1 -> +1, 0 -> -1  (code 3 vs 1)
    scales = raw(t["scales"]).astype(np.float32).reshape(N, nb)

    # pack bits to u32: bit k of row n at word k//32, position k%32
    shifts = (np.uint32(1) << np.arange(32, dtype=np.uint32))
    signbits = (bits.reshape(N, K // 32, 32) * shifts).sum(2).astype(np.uint32)

    W = (2.0 * bits.astype(np.float32) - 1.0) * scales[:, np.arange(K) // 128]   # [N,K]
    rng = np.random.default_rng(0)
    x = rng.standard_normal((args.m, K)).astype(np.float32)
    expected = (x @ W.T).astype(np.float32)

    x.tofile(f"{args.out}/x.f32.bin")
    signbits.tofile(f"{args.out}/signbits.u32.bin")
    scales.tofile(f"{args.out}/scales.f32.bin")
    expected.tofile(f"{args.out}/expected.f32.bin")
    json.dump({"tensor": args.tensor, "M": args.m, "N": N, "K": K, "nb": nb, "block": 128},
              open(f"{args.out}/fixture.json", "w"), indent=1)
    print(f"wrote fixtures to {args.out}/  (M={args.m}, N={N}, K={K}, nb={nb})")
    print(f"  expected[0,:4] = {expected[0,:4].tolist()}")


if __name__ == "__main__":
    main()
