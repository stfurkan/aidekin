#!/usr/bin/env python3
"""Generate WebGPU kernel test fixtures from the verified weights/reference (P1).

Emits one folder per kernel under <out>/<name>/ with little-endian input blobs,
expected.f32.bin, and params.json. The browser harness checks each kernel against
these known-correct results. Inputs are seeded; weights/gammas/rope caches are real.

Usage: python make_fixtures.py --work <dir> [--out scripts/bonsai/webgpu-test/fixtures]
"""
import argparse
import json
import os

import numpy as np

NP_DT = {"FLOAT": np.float32, "FLOAT16": np.float16, "UINT8": np.uint8, "INT64": np.int64}
RNG = np.random.default_rng(0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", default="scripts/bonsai/webgpu-test/fixtures")
    args = ap.parse_args()

    M = json.load(open(f"{args.work}/manifest.json"))
    A = M["arch"]
    data = open(f"{args.work}/{M['data_file']}", "rb")
    aux = open(f"{args.work}/{M['aux_file']}", "rb").read()
    T = M["tensors"]

    def raw(ref: dict) -> np.ndarray:
        if ref["src"] == "aux":
            b = aux[ref["off"]:ref["off"] + ref["len"]]
        else:
            data.seek(ref["off"])
            b = data.read(ref["len"])
        return np.frombuffer(b, NP_DT[ref["dtype"]])

    def emit(name: str, params: dict, **arrays: np.ndarray) -> None:
        d = f"{args.out}/{name}"
        os.makedirs(d, exist_ok=True)
        for k, v in arrays.items():
            v.tofile(f"{d}/{k}.bin")
        json.dump(params, open(f"{d}/params.json", "w"), indent=1)
        print(f"  {name}: {params}")

    H, KV, D = A["heads"], A["kv_heads"], A["head_dim"]

    # 1) binary matmul (layers.0.attn.q_proj)
    tgt2 = raw(M["luts"]["tgt2"]).reshape(256, 2)
    t = T["layers.0.attn.q_proj"]
    N, K, nb = t["N"], t["K"], t["K"] // 128
    wq = raw(t["weight"]).reshape(N, K // 8)
    exp = tgt2[wq].reshape(N, K // 4)
    codes = np.empty((N, K), np.uint8)
    for k in range(4):
        codes[:, k::4] = (exp >> (2 * k)) & 3
    bits = ((codes >> 1) & 1).astype(np.uint32)
    scales = raw(t["scales"]).astype(np.float32).reshape(N, nb)
    shifts = np.uint32(1) << np.arange(32, dtype=np.uint32)
    signbits = (bits.reshape(N, K // 32, 32) * shifts).sum(2).astype(np.uint32)
    W = (2.0 * bits.astype(np.float32) - 1.0) * scales[:, np.arange(K) // 128]
    Mrows = 8
    x = RNG.standard_normal((Mrows, K)).astype(np.float32)
    emit("binary_matmul", {"M": Mrows, "N": N, "K": K, "nb": nb, "block": 128},
         x=x, signbits=signbits, scales=scales.reshape(-1), expected=(x @ W.T).astype(np.float32))

    # 2) rmsnorm (real gamma)
    gamma = raw(T["layers.0.input_layernorm"]["weight"]).astype(np.float32)
    Dn = gamma.shape[0]
    xr = RNG.standard_normal((8, Dn)).astype(np.float32)
    rms = xr / np.sqrt((xr ** 2).mean(-1, keepdims=True) + A["rms_eps"]) * gamma
    emit("rmsnorm", {"R": 8, "D": Dn, "eps": A["rms_eps"]}, x=xr, gamma=gamma, expected=rms.astype(np.float32))

    # 3) rope (real YaRN cos/sin caches, full = concat(half, half))
    S = 12
    cos_c = raw(T["cos_cache"]).reshape(T["cos_cache"]["shape"]).astype(np.float32)
    sin_c = raw(T["sin_cache"]).reshape(T["sin_cache"]["shape"]).astype(np.float32)
    cos = np.concatenate([cos_c[:S], cos_c[:S]], -1)
    sin = np.concatenate([sin_c[:S], sin_c[:S]], -1)
    xq = RNG.standard_normal((S, H, D)).astype(np.float32)
    half = D // 2
    rot = np.concatenate([-xq[..., half:], xq[..., :half]], -1)
    rope = xq * cos[:, None, :] + rot * sin[:, None, :]
    emit("rope", {"S": S, "H": H, "D": D}, x=xq.reshape(-1), cos=cos.reshape(-1), sin=sin.reshape(-1),
         expected=rope.reshape(-1).astype(np.float32))

    # 4) swiglu
    F = A["intermediate"]
    gate = RNG.standard_normal((8, F)).astype(np.float32)
    up = RNG.standard_normal((8, F)).astype(np.float32)
    sg = (gate / (1.0 + np.exp(-gate))) * up
    emit("swiglu", {"n": 8 * F}, gate=gate.reshape(-1), up=up.reshape(-1), expected=sg.reshape(-1).astype(np.float32))

    # 5) attention (causal, per-head; k/v already expanded to H heads)
    q = RNG.standard_normal((S, H, D)).astype(np.float32)
    k = RNG.standard_normal((S, H, D)).astype(np.float32)
    v = RNG.standard_normal((S, H, D)).astype(np.float32)
    scale = 1.0 / np.sqrt(D)
    outa = np.empty((S, H, D), np.float32)
    for h in range(H):
        for s in range(S):
            sc = (q[s, h] @ k[:s + 1, h].T) * scale
            sc -= sc.max()
            p = np.exp(sc)
            p /= p.sum()
            outa[s, h] = p @ v[:s + 1, h]
    emit("attention", {"S": S, "H": H, "D": D}, q=q.reshape(-1), k=k.reshape(-1), v=v.reshape(-1),
         expected=outa.reshape(-1).astype(np.float32))

    print(f"wrote fixtures to {args.out}/")


if __name__ == "__main__":
    main()
