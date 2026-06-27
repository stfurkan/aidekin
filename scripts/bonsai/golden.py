#!/usr/bin/env python3
"""Capture golden reference logits from the q1 model for P1 correctness checks.

Runs model_q1.onnx on CPU (onnxruntime) for a fixed prompt and saves the exact
logits. The from-scratch WebGPU runtime must reproduce these number-for-number.

Usage: python golden.py --work <dir> [--prompt "..."] [--out <dir>]
"""
import argparse
import json

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

LAYERS = 28
KV_HEADS = 8
HEAD_DIM = 128


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--prompt", default="The capital of France is Paris. The capital of Japan is")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or f"{args.work}/golden"

    tok = Tokenizer.from_file(f"{args.work}/tokenizer.json")
    ids = tok.encode(args.prompt).ids
    S = len(ids)

    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(f"{args.work}/model_q1.onnx", so, providers=["CPUExecutionProvider"])

    feeds = {
        "input_ids": np.array([ids], np.int64),
        "attention_mask": np.ones((1, S), np.int64),
        "num_logits_to_keep": np.array(S, np.int64),
    }
    for i in range(LAYERS):
        feeds[f"past_key_values.{i}.key"] = np.zeros((1, KV_HEADS, 0, HEAD_DIM), np.float32)
        feeds[f"past_key_values.{i}.value"] = np.zeros((1, KV_HEADS, 0, HEAD_DIM), np.float32)

    logits = sess.run(["logits"], feeds)[0][0]  # [S, vocab]
    last = logits[-1].astype(np.float32)
    top = np.argsort(last)[::-1][:5]

    import os
    os.makedirs(out, exist_ok=True)
    np.save(f"{out}/input_ids.npy", np.array(ids, np.int64))
    np.save(f"{out}/logits_all.npy", logits.astype(np.float32))
    json.dump(
        {
            "prompt": args.prompt,
            "ids": ids,
            "S": S,
            "argmax": int(top[0]),
            "logits_last_first16": [round(float(x), 5) for x in last[:16]],
        },
        open(f"{out}/meta.json", "w"),
        indent=2,
    )
    print("prompt:", repr(args.prompt), "| S =", S)
    print("argmax next token:", int(top[0]), "->", repr(tok.decode([int(top[0])])))
    print("top5:", [(int(t), repr(tok.decode([int(t)])), round(float(last[t]), 2)) for t in top])
    print(f"saved -> {out}/  (logits_all.npy {logits.nbytes / 1e6:.1f} MB, input_ids.npy, meta.json)")


if __name__ == "__main__":
    main()
