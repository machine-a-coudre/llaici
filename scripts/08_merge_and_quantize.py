#!/usr/bin/env python3
"""FINETUNING.md §6 — merge the LoRA adapter and export to GGUF Q8.

Loads the LoRA adapter produced by scripts/06_finetune.py back into Unsloth and
exports it as a single quantized GGUF file, ready for scripts/09 (serve via
llama-server) and DESIGN.md STEP 6 (deployment).

API verified against Unsloth's own documentation rather than guessed:
- `FastLanguageModel.from_pretrained(model_name=<adapter_dir>, ...)` is the
  documented way to reload a saved LoRA adapter directory — it resolves the base
  model from the adapter's own config and reattaches the trained adapter weights,
  no separate PeftModel.from_pretrained() call needed for this case.
- `model.save_pretrained_gguf(output_dir, tokenizer, quantization_method="q8_0")`
  merges the adapter into the base weights (in fp16 internally, to avoid
  compounding the base model's own 4-bit quantization error — Unsloth
  dequantizes before merging, per its docs) and quantizes to GGUF Q8 in one call.
  No separate "merge" step is needed before this — merging is handled inside
  `save_pretrained_gguf` itself. "q8_0" (not "q4_k_m"/"f16"/etc.) is the specific
  quantization level DESIGN.md names ("Quantize to GGUF Q8 (~800 MB)").

⚠️ Requires the same CUDA + Unsloth environment as scripts/06_finetune.py and
scripts/07_evaluate.py, plus a completed adapter from the former. Not run/tested
in this development environment — see 06_finetune.py's docstring for why.

Run (on a CUDA machine, after scripts/06_finetune.py has produced an adapter):
    python3 scripts/08_merge_and_quantize.py
    python3 scripts/08_merge_and_quantize.py --model-name unsloth/Qwen3-1.7B-unsloth-bnb-4bit --quantization-method q4_k_m
"""

import argparse
import os

from model_paths import model_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Only used to name the default directories: the base model itself is resolved
    # from the adapter's own config (see docstring).
    parser.add_argument("--model-name", default="unsloth/Qwen3-0.6B-unsloth-bnb-4bit", help="base model given to scripts/06_finetune.py")
    parser.add_argument("--adapter-dir", default=None, help="default: models/llaici-<model>-lora")
    parser.add_argument("--output-dir", default=None, help="default: models/llaici-<model>-gguf")
    parser.add_argument("--quantization-method", default="q8_0", help="DESIGN.md STEP 5 target: GGUF Q8 (~800MB)")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    args = parser.parse_args()
    args.adapter_dir = args.adapter_dir or model_dir(args.model_name, "lora")
    args.output_dir = args.output_dir or model_dir(args.model_name, "gguf")

    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter_dir,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained_gguf(args.output_dir, tokenizer, quantization_method=args.quantization_method)

    print(f"# GGUF ({args.quantization_method}) model written to {args.output_dir}")


if __name__ == "__main__":
    main()
