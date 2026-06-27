# Bonsai-1.7B weight tooling (WebGPU engine, P0)

Offline tooling that establishes and verifies how the Bonsai-1.7B "q1" ONNX weights
are packed, so the custom WebGPU runtime can load them directly. Python is used here
only because reading the ONNX graph (initializer byte offsets, op attributes) is far
simpler with the `onnx` package; nothing here ships to the browser.

## Findings (all verified, cosine 1.0 vs the full-precision reference)

Source of truth: `onnx-community/Bonsai-1.7B-ONNX` -> `onnx/model_q1.onnx` (graph,
~0.5 MB) + `onnx/model_q1.onnx_data` (~290 MB weights). Answer-key for verification:
`prism-ml/Bonsai-1.7B-unpacked/model.safetensors` (~3.4 GB, f16).

Architecture: Qwen3, 28 layers, hidden 2048, intermediate 6144 (SwiGLU/silu), 16
attention heads / 8 KV heads (GQA), head_dim 128, RMSNorm eps 1e-6, RoPE theta 1e6
with YaRN (factor 4, original 8192), tied embeddings, vocab 151669.

Packing. Each weight is stored as `*_weight_quant` uint8 `[N, K/8]`, lookup-compressed
to ~1 bit/weight on disk (this is why it is 290 MB). Two shared lookup tables expand it:

- `unpack_lut_src1_tgt2` `[256,2]` (inline): 1 source byte -> 2 bytes = standard 2-bit
  codes. Used by the 196 linear weights and the lm_head.
- `unpack_lut_src1_tgt4` `[256,4]` (external): 1 -> 4 bytes = 4-bit codes. Used only by
  the input-embedding lookup (higher precision where it matters).

After expansion, codes are LSB-first (`code_k = (byte >> (2*k)) & 3` for 2-bit).
Dequant: `value = (code - zero_point) * scale`, with fp32 `scales [N, K/128]`,
`block_size = 128`, and packed per-block `zero_points`.

Quantization nature (global scan of all 196 attention + MLP matmuls, ~1.41B weights):

- Pure binary `{-1, +1}`. Only codes 1 (`-1`) and 3 (`+1`) ever occur, never 0 or 2.
  Zero-points are uniformly 2. Zero anomalies. So the hot matmul is sign-only
  add/subtract, and weights can live at 1 bit each in VRAM (~176 MB).
- The tied vocab table is the one higher-precision part: lm_head reads it at 2-bit,
  the embedding lookup at 4-bit, both sharing one fp32 scale table.

End-to-end: the q1 model runs on CPU (onnxruntime) and is coherent ("...the capital of
Japan is" -> " Tokyo"), confirming the decode chain is correct.

## Setup

```sh
cd scripts/bonsai
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt        # latest onnx / numpy / onnxruntime / tokenizers

# fetch the q1 graph + weights + tokenizer into a work dir (kept out of git)
mkdir -p .work
base=https://huggingface.co/onnx-community/Bonsai-1.7B-ONNX/resolve/main
curl -L -o .work/model_q1.onnx       $base/onnx/model_q1.onnx
curl -L -C - -o .work/model_q1.onnx_data $base/onnx/model_q1.onnx_data   # ~290 MB, -C - resumes
curl -L -o .work/tokenizer.json      $base/tokenizer.json
```

## Usage

```sh
.venv/bin/python verify_weights.py --work .work   # verify decode recipe + global quant scan
.venv/bin/python golden.py --work .work --out .work/golden   # capture reference logits for P1
```
