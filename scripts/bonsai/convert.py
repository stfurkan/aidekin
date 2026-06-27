#!/usr/bin/env python3
"""Emit a load manifest for the Bonsai-1.7B q1 weights (P1 step 1).

The browser runtime must load weights without parsing ONNX. This walks the q1 graph
and writes:
  - manifest.json: arch config + every tensor mapped to byte ranges (logical names).
  - bonsai.aux.bin: the small tensors that are inline in the graph or scattered
    (norm gammas, the tgt2 lookup table). Big tensors (weight_quant / scales / zero
    points / rope caches) are referenced directly in the shipped model_q1.onnx_data,
    so the ~290 MB download is unchanged.

Usage: python convert.py --work <dir> [--out <dir>]
"""
import argparse
import json
import re

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

NORM_OPS = {"SimplifiedLayerNormalization", "SkipSimplifiedLayerNormalization"}


def logical_linear(quant_name: str) -> str:
    """model_layers_0_attn_q_proj_MatMul_weight_quant -> layers.0.attn.q_proj"""
    m = re.match(r"model_layers_(\d+)_(attn|mlp)_(.+?)_MatMul_weight_quant", quant_name)
    if not m:
        raise ValueError(f"unrecognized linear weight name: {quant_name}")
    return f"layers.{m.group(1)}.{m.group(2)}.{m.group(3)}"


def logical_norm(gamma_name: str) -> str:
    """model.layers.0.attn.q_norm.layernorm.weight -> layers.0.attn.q_norm ; model.norm.weight -> norm"""
    return gamma_name.removeprefix("model.").removesuffix(".weight").replace(".layernorm", "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or args.work

    cfg = json.load(open(f"{args.work}/config.json"))
    graph = onnx.load(f"{args.work}/model_q1.onnx", load_external_data=False).graph
    inits = {i.name: i for i in graph.initializer}

    aux = bytearray()

    def place(name: str) -> dict:
        """Reference an initializer: external -> point into model_q1.onnx_data; inline -> copy to aux."""
        t = inits[name]
        if t.data_location == TensorProto.EXTERNAL:
            d = {x.key: x.value for x in t.external_data}
            return {"src": "data", "off": int(d["offset"]), "len": int(d["length"])}
        arr = numpy_helper.to_array(t)
        b = arr.tobytes()
        off = len(aux)
        aux.extend(b)
        return {"src": "aux", "off": off, "len": len(b), "dtype": str(arr.dtype), "shape": list(arr.shape)}

    tensors: dict[str, dict] = {}

    # linear weights + lm_head (MatMulNBits)
    for n in graph.node:
        if n.op_type != "MatMulNBits":
            continue
        scales, zp = n.input[2], n.input[3]
        K = next(a.i for a in n.attribute if a.name == "K")
        N = next(a.i for a in n.attribute if a.name == "N")
        if n.name == "/lm_head/MatMul_Quant":
            name, kind, wq, lut = "lm_head", "q2", "model_embed_tokens_weight_quant", "tgt2"
        else:
            wq = scales.replace("_weight_scales", "_weight_quant")
            name, kind, lut = logical_linear(wq), "binary", "tgt2"
        tensors[name] = {"kind": kind, "N": N, "K": K, "block": 128, "bits": 2, "lut": lut,
                         "weight": place(wq), "scales": place(scales), "zp": place(zp)}

    # input embedding (GatherBlockQuantized, 4-bit view of the tied table)
    gbq = next(n for n in graph.node if n.op_type == "GatherBlockQuantized")
    bits = next(a.i for a in gbq.attribute if a.name == "bits")
    tensors["embed_tokens"] = {
        "kind": "q4", "rows": cfg["vocab_size"], "cols": cfg["hidden_size"], "block": 128, "bits": bits, "lut": "tgt4",
        "weight": place("model_embed_tokens_weight_quant"),
        "scales": place("model_embed_tokens_weight_scales"),
        "zp": place("model_embed_tokens_weight_zp_4b"),
    }

    # RMSNorm gammas
    for n in graph.node:
        if n.op_type not in NORM_OPS:
            continue
        gamma = next((i for i in n.input if i in inits), None)
        if gamma is None:
            continue
        tensors[logical_norm(gamma)] = {"kind": "f32", "weight": place(gamma)}

    # RoPE caches (YaRN-scaled; reference as-is for exact parity)
    for c in ("cos_cache", "sin_cache"):
        if c in inits:
            tensors[c] = {"kind": "f32", "shape": list(inits[c].dims), **place(c)}

    # the two lookup tables
    luts = {"tgt2": place("unpack_lut_src1_tgt2"), "tgt4": place("unpack_lut_src1_tgt4")}

    manifest = {
        "version": 1,
        "data_file": "model_q1.onnx_data",
        "aux_file": "bonsai.aux.bin",
        "arch": {
            "model_type": cfg["model_type"], "layers": cfg["num_hidden_layers"],
            "hidden": cfg["hidden_size"], "intermediate": cfg["intermediate_size"],
            "heads": cfg["num_attention_heads"], "kv_heads": cfg["num_key_value_heads"],
            "head_dim": cfg["head_dim"], "rms_eps": cfg["rms_norm_eps"],
            "rope": cfg.get("rope_parameters", {"rope_theta": 1e6}),
            "vocab": cfg["vocab_size"], "eos": cfg["eos_token_id"],
            "tie_word_embeddings": cfg["tie_word_embeddings"], "act": cfg["hidden_act"],
        },
        "luts": luts,
        "tensors": tensors,
    }

    open(f"{out}/bonsai.aux.bin", "wb").write(aux)
    json.dump(manifest, open(f"{out}/manifest.json", "w"), indent=1)
    print(f"manifest: {len(tensors)} tensors | aux.bin: {len(aux)/1e6:.3f} MB")
    kinds: dict[str, int] = {}
    for t in tensors.values():
        kinds[t["kind"]] = kinds.get(t["kind"], 0) + 1
    print("tensor kinds:", kinds)

    # round-trip self-check: decode layers.0.attn.q_proj row 0 from the manifest, cosine vs f16
    _selfcheck(args.work, manifest, aux)


def _selfcheck(work: str, manifest: dict, aux: bytes) -> None:
    import struct
    import urllib.request

    data = open(f"{work}/model_q1.onnx_data", "rb")

    def read(ref: dict) -> bytes:
        if ref["src"] == "aux":
            return bytes(aux[ref["off"]:ref["off"] + ref["len"]])
        data.seek(ref["off"])
        return data.read(ref["len"])

    tgt2 = np.frombuffer(read(manifest["luts"]["tgt2"]), np.uint8).reshape(256, 2)
    t = manifest["tensors"]["layers.0.attn.q_proj"]
    wq = np.frombuffer(read(t["weight"]), np.uint8).reshape(t["N"], t["K"] // 8)[:1]
    sc = np.frombuffer(read(t["scales"]), np.float32).reshape(t["N"], t["K"] // 128)[0]
    zp = np.frombuffer(read(t["zp"]), np.uint8).reshape(t["N"], t["K"] // 128 // 4)[0]
    exp = tgt2[wq].reshape(1, -1)
    c = np.empty((1, t["K"]), np.uint8)
    for k in range(4):
        c[:, k::4] = (exp >> (2 * k)) & 3
    dec = np.empty(t["K"], np.float32)
    for b in range(t["K"] // 128):
        z = (zp[b // 4] >> (2 * (b % 4))) & 3
        dec[b * 128:(b + 1) * 128] = (c[0, b * 128:(b + 1) * 128].astype(np.int32) - int(z)) * sc[b]

    url = "https://huggingface.co/prism-ml/Bonsai-1.7B-unpacked/resolve/main/model.safetensors"
    def net(o, l):
        return urllib.request.urlopen(urllib.request.Request(url, headers={"Range": f"bytes={o}-{o + l - 1}"}), timeout=120).read()
    hlen = struct.unpack("<Q", net(0, 8))[0]
    hdr = json.loads(net(8, hlen).decode())
    o0, _ = hdr["model.layers.0.self_attn.q_proj.weight"]["data_offsets"]
    ref = np.frombuffer(net(8 + hlen + o0, 2048 * 2), np.float16).astype(np.float32)
    cos = float(np.dot(dec, ref) / (np.linalg.norm(dec) * np.linalg.norm(ref) + 1e-9))
    print("round-trip self-check (q_proj L0 via manifest) cosine vs f16:", round(cos, 5))


if __name__ == "__main__":
    main()
