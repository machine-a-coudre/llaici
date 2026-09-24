#!/usr/bin/env python3
"""FINETUNING.md §6, Apple Silicon alternative to scripts/08_merge_and_quantize.py.

scripts/08_merge_and_quantize.py merges the LoRA adapter and quantizes to GGUF in
one Unsloth call (`save_pretrained_gguf`) — CUDA-only, same reason as
scripts/06_finetune.py. For an adapter trained with scripts/06_finetune_mlx.py
(mlx-lm), this script does the equivalent in two separate steps instead, since no
single mlx-lm call does both:

1. **Merge**: `mlx_lm.fuse` reattaches the LoRA adapter to the base model and
   writes a Hugging Face-compatible directory (`config.json` + `.safetensors` +
   tokenizer files) — confirmed from `mlx_lm/fuse.py`'s own source (`save()` call),
   not guessed. `--dequantize` is required here: scripts/06_finetune_mlx.py trains
   QLoRA against a 4-bit base (`mlx-community/Qwen3-0.6B-4bit`), and fusing a LoRA
   delta into still-quantized weights would compound the base model's own
   quantization error into the merge (same reasoning scripts/08_merge_and_quantize.py's
   docstring gives for Unsloth's own dequantize-before-merge behavior).
   `mlx_lm.fuse --export-gguf` is **not** used for this: confirmed directly in
   `mlx_lm/fuse.py` — it raises `ValueError` unless `config["model_type"]` is
   `llama`/`mixtral`/`mistral`; Qwen3's model_type isn't in that list.
2. **Convert + quantize**: llama.cpp's own `convert_hf_to_gguf.py` reads the fused
   HF directory `mlx_lm.fuse` just produced and writes a GGUF directly — llama.cpp
   is what actually serves the model (`llama-server`, see FINETUNING.md §7), and it
   only ever loads GGUF, never a raw HF/safetensors directory. Its `--outtype`
   choices (confirmed from the script's own argparse) include `q8_0` directly, so
   this single call produces the DESIGN.md target ("Quantize to GGUF Q8 (~800 MB)")
   without a separate `llama-quantize` step — that binary is only needed for
   quantization levels `convert_hf_to_gguf.py` doesn't support directly
   (`q4_k_m` etc.), not for `q8_0`.

Requires a local clone of https://github.com/ggml-org/llama.cpp (for
`convert_hf_to_gguf.py` and its `requirements.txt`) — that script isn't published
as an installable package on its own, unlike `mlx-lm`. Pass its path with
`--llama-cpp-dir`.

⚠️ Not run/tested — same reason as scripts/06_finetune_mlx.py (no Apple Silicon
Mac available in this environment) and, on top of that, no adapter exists yet
from either fine-tuning path to actually feed this script.

Run (on a Mac, after scripts/06_finetune_mlx.py has produced an adapter):
    python3 scripts/08_merge_and_quantize_mlx.py --llama-cpp-dir ~/llama.cpp
    python3 scripts/08_merge_and_quantize_mlx.py --llama-cpp-dir ~/llama.cpp --outtype q8_0
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-model", default="mlx-community/Qwen3-0.6B-4bit", help="must match scripts/06_finetune_mlx.py's --model")
    parser.add_argument("--adapter-path", default="models/llaici-qwen3-0.6b-mlx-lora", help="LoRA adapter dir from scripts/06_finetune_mlx.py")
    parser.add_argument("--fused-dir", default="models/llaici-qwen3-0.6b-mlx-fused", help="HF-format merged model, written by mlx_lm.fuse")
    parser.add_argument("--llama-cpp-dir", required=True, help="path to a local ggml-org/llama.cpp checkout (needs convert_hf_to_gguf.py)")
    parser.add_argument("--output-dir", default="models/llaici-qwen3-0.6b-gguf")
    parser.add_argument(
        "--outtype", default="q8_0", choices=["f32", "f16", "bf16", "q8_0", "auto"],
        help="convert_hf_to_gguf.py's --outtype — DESIGN.md target is q8_0 (~800MB)",
    )
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print(f"# running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()

    if shutil.which("mlx_lm.fuse") is None:
        sys.exit('mlx_lm.fuse not found on PATH — install it with: pip install "mlx-lm[train]"')

    convert_script = Path(args.llama_cpp_dir) / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        sys.exit(
            f"{convert_script} not found — pass --llama-cpp-dir pointing to a clone of "
            "https://github.com/ggml-org/llama.cpp (with its requirements.txt installed)"
        )

    # Step 1: merge the LoRA adapter into the (dequantized) base model, HF format out.
    run([
        "mlx_lm.fuse",
        "--model", args.base_model,
        "--adapter-path", args.adapter_path,
        "--save-path", args.fused_dir,
        "--dequantize",
    ])

    # Step 2: convert the fused HF directory straight to a quantized GGUF.
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    outfile = Path(args.output_dir) / f"llaici-qwen3-0.6b.{args.outtype}.gguf"
    run([
        sys.executable, str(convert_script),
        args.fused_dir,
        "--outfile", str(outfile),
        "--outtype", args.outtype,
    ])

    print(f"# GGUF ({args.outtype}) model written to {outfile}")


if __name__ == "__main__":
    main()
