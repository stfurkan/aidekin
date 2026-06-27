#!/usr/bin/env python3
"""Verify the Bonsai-1.7B q1 (1-bit) weight-decode recipe and scan the quantization.

Decodes the lookup-compressed q1 weights and checks them against the full-precision
reference (prism-ml/Bonsai-1.7B-unpacked) via HTTP range requests, then scans every
linear weight to confirm the model is pure binary. See README.md.

Usage: python verify_weights.py --work <dir containing model_q1.onnx + .onnx_data>
"""
import argparse
import json
import struct
import urllib.request

import numpy as np
import onnx
from onnx import numpy_helper

REF_URL = "https://huggingface.co/prism-ml/Bonsai-1.7B-unpacked/resolve/main/model.safetensors"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, help="dir with model_q1.onnx and model_q1.onnx_data")
    work = ap.parse_args().work

    graph = onnx.load(f"{work}/model_q1.onnx", load_external_data=False).graph
    inits = {i.name: i for i in graph.initializer}
    data = open(f"{work}/model_q1.onnx_data", "rb")

    def ext(name: str) -> bytes:
        t = inits[name]
        loc = {d.key: d.value for d in t.external_data}
        data.seek(int(loc["offset"]))
        return data.read(int(loc["length"]))

    tgt2 = numpy_helper.to_array(inits["unpack_lut_src1_tgt2"])              # [256,2], 1->2 bytes (2-bit)
    tgt4 = np.frombuffer(ext("unpack_lut_src1_tgt4"), np.uint8).reshape(256, 4)  # 1->4 bytes (4-bit)

    def codes(wq: np.ndarray, lut: np.ndarray, bits: int) -> np.ndarray:
        """Expand lookup-compressed bytes [N, K/8] to per-weight codes [N, K]."""
        per = 8 // bits
        expanded = lut[wq].reshape(wq.shape[0], -1)        # [N, K/per]
        out = np.empty((wq.shape[0], expanded.shape[1] * per), np.uint8)
        for k in range(per):
            out[:, k::per] = (expanded >> (bits * k)) & ((1 << bits) - 1)
        return out

    def dequant(c: np.ndarray, scales: np.ndarray, zp_row, bits: int, block: int = 128) -> np.ndarray:
        K = c.shape[1]
        out = np.empty(K, np.float32)
        per = 8 // bits
        for b in range(K // block):
            z = (zp_row[b // per] >> (bits * (b % per))) & ((1 << bits) - 1)
            out[b * block:(b + 1) * block] = (c[0, b * block:(b + 1) * block].astype(np.int32) - int(z)) * scales[b]
        return out

    # f16 answer-key via range requests
    def net(o: int, l: int) -> bytes:
        req = urllib.request.Request(REF_URL, headers={"Range": f"bytes={o}-{o + l - 1}"})
        return urllib.request.urlopen(req, timeout=120).read()

    hlen = struct.unpack("<Q", net(0, 8))[0]
    hdr = json.loads(net(8, hlen).decode())
    base = 8 + hlen

    def f16_row(name: str, cols: int) -> np.ndarray:
        o0, _ = hdr[name]["data_offsets"]
        return np.frombuffer(net(base + o0, cols * 2), np.float16).astype(np.float32)

    def cos(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    print("=== decode recipe verification (row 0, cosine vs f16) ===")
    # linear q_proj layer 0 (2-bit binary)
    wq = np.frombuffer(ext("model_layers_0_attn_q_proj_MatMul_weight_quant"), np.uint8).reshape(2048, 256)
    sc = np.frombuffer(ext("model_layers_0_attn_q_proj_MatMul_weight_scales"), np.float32).reshape(2048, 16)
    zp = np.frombuffer(ext("model_layers_0_attn_q_proj_MatMul_weight_zp_2b"), np.uint8).reshape(2048, 4)
    dec = dequant(codes(wq[:1], tgt2, 2), sc[0], zp[0], 2)
    print("  layers.0.attn.q_proj (2-bit):", round(cos(dec, f16_row("model.layers.0.self_attn.q_proj.weight", 2048)), 5))
    # tied vocab table: lm_head at 2-bit, embedding at 4-bit
    wqe = np.frombuffer(ext("model_embed_tokens_weight_quant"), np.uint8).reshape(151669, 256)
    sce = np.frombuffer(ext("model_embed_tokens_weight_scales"), np.float32).reshape(151669, 16)
    zp2 = np.frombuffer(ext("model_embed_tokens_weight_quant_tied_zp_2b"), np.uint8).reshape(151669, 4)
    zp4 = np.frombuffer(ext("model_embed_tokens_weight_zp_4b"), np.uint8).reshape(151669, 8)
    ref_emb = f16_row("model.embed_tokens.weight", 2048)
    print("  lm_head tied table (2-bit):", round(cos(dequant(codes(wqe[:1], tgt2, 2), sce[0], zp2[0], 2), ref_emb), 5))
    print("  embedding tied table (4-bit):", round(cos(dequant(codes(wqe[:1], tgt4, 4), sce[0], zp4[0], 4), ref_emb), 5))

    print("\n=== global quant scan (all linear MatMulNBits) ===")
    nodes = [n for n in graph.node if n.op_type == "MatMulNBits" and n.name != "/lm_head/MatMul_Quant"]
    total = np.zeros(4, np.int64)
    zpset: set[int] = set()
    binary = ternary = other = 0
    shapes: dict[tuple[int, int], int] = {}
    for n in nodes:
        scales_name = n.input[2]
        zp_name = n.input[3]
        wq_name = scales_name.replace("_weight_scales", "_weight_quant")
        K = next(a.i for a in n.attribute if a.name == "K")
        N = next(a.i for a in n.attribute if a.name == "N")
        nb = K // 128
        c = codes(np.frombuffer(ext(wq_name), np.uint8).reshape(N, K // 8), tgt2, 2)
        h = np.bincount(c.reshape(-1), minlength=4)
        total += h
        zpb = np.frombuffer(ext(zp_name), np.uint8).reshape(N, nb // 4)
        for b in range(nb):
            zpset |= set(np.unique((zpb[:, b // 4] >> (2 * (b % 4))) & 3).tolist())
        if h[0] == 0 and h[2] == 0:
            binary += 1
        elif h[0] == 0:
            ternary += 1
        else:
            other += 1
        shapes[(N, K)] = shapes.get((N, K), 0) + 1
    print("  weights scanned:", len(nodes))
    print("  code histogram [-2x, -1x, 0, +1x]:", dict(enumerate(total.tolist())))
    print("  zero-points seen:", sorted(zpset))
    print(f"  pure-binary: {binary} | ternary: {ternary} | has -2 level: {other}")
    print("  shapes (N,K):count:", {f"{k[0]}x{k[1]}": v for k, v in shapes.items()})


if __name__ == "__main__":
    main()
