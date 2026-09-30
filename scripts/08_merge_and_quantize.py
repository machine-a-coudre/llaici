#!/usr/bin/env python3
"""FINETUNING.md §6 — merge the LoRA adapter into its base model and export to GGUF Q8.

Two explicit steps, the same shape as scripts/08_merge_and_quantize_mlx.py:

1. **Merge, with PEFT** (Hugging Face's own LoRA library): loads the *16-bit*
   version of the base model, attaches the adapter from scripts/06_finetune.py,
   `merge_and_unload()`, and saves the result as a plain Hugging Face model in a
   new directory (`models/llaici-<model>-merged/`). The adapter was trained on a
   4-bit base (QLoRA), but merging into 16-bit weights is the standard practice:
   merging into still-quantized weights would compound quantization error.
   Everything is written to new files — unlike Unsloth's `save_pretrained_gguf`,
   used here before, which copied the base weights out of the Hugging Face cache
   and merged into that copy in place, failing with "Permission denied" when the
   cache files are read-only.
2. **Convert + quantize, with llama.cpp**: its `convert_hf_to_gguf.py` reads the
   merged directory and writes a single GGUF file — what `llama-server` serves
   (FINETUNING.md §7). Its `--outtype` accepts `q8_0` directly (the DESIGN.md
   target, "Quantize to GGUF Q8"); levels like `q4_k_m` would need llama.cpp's
   compiled `llama-quantize` binary on top, not handled here.

Runs on CPU (a 0.6B model merges in seconds, no GPU needed), in the same venv as
steps 06-07 (`transformers`, `peft`, `torch`), with no Unsloth import. Requires a
local clone of https://github.com/ggml-org/llama.cpp, passed with --llama-cpp-dir:
`convert_hf_to_gguf.py` isn't published as an installable package. It's run with
this venv's Python; if it reports a missing module, install llama.cpp's
`requirements/requirements-convert_hf_to_gguf.txt` into the venv.

The 16-bit base model is derived from the adapter's own `adapter_config.json`
(its `base_model_name_or_path`, minus the 4-bit suffix): `unsloth/Qwen3-0.6B-unsloth-bnb-4bit`
-> `unsloth/qwen3-0.6b`. Downloaded from Hugging Face on first use (~1.2 GB for
Qwen3 0.6B); override with --base-model-16bit if the derived name doesn't exist.

Run (after scripts/06_finetune.py has produced an adapter):
    python3 scripts/08_merge_and_quantize.py --llama-cpp-dir ~/llama.cpp
    python3 scripts/08_merge_and_quantize.py --llama-cpp-dir ~/llama.cpp --model-name unsloth/Qwen3-1.7B-unsloth-bnb-4bit
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from console import human_size, print_box
from model_paths import QUANT_SUFFIXES, model_dir, model_slug


def base_16bit(adapter_dir: str) -> str:
    """16-bit counterpart of the 4-bit base the adapter was trained on."""
    config = json.loads((Path(adapter_dir) / "adapter_config.json").read_text())
    base = config["base_model_name_or_path"]
    for suffix in QUANT_SUFFIXES:
        if base.lower().endswith(suffix):
            # Lowercased like Unsloth's own 4-bit -> 16-bit mapping, so an already
            # cached copy (e.g. from an earlier Unsloth export) is reused.
            return base[: -len(suffix)].lower()
    return base


def merge(base_model: str, adapter_dir: str, merged_dir: str) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"# merging {adapter_dir} into {base_model} (16-bit) -> {merged_dir}")
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, adapter_dir).merge_and_unload()
    model.save_pretrained(merged_dir)
    # From the adapter dir, not the base model: it carries the chat template the
    # model was trained with (qwen3-instruct), which ends up embedded in the GGUF.
    AutoTokenizer.from_pretrained(adapter_dir).save_pretrained(merged_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-name", default="unsloth/Qwen3-0.6B-unsloth-bnb-4bit", help="base model given to scripts/06_finetune.py (names the directories)")
    parser.add_argument("--llama-cpp-dir", required=True, help="path to a local ggml-org/llama.cpp clone (needs convert_hf_to_gguf.py)")
    parser.add_argument("--adapter-dir", default=None, help="default: models/llaici-<model>-lora")
    parser.add_argument("--base-model-16bit", default=None, help="default: derived from the adapter's adapter_config.json (see docstring)")
    parser.add_argument("--merged-dir", default=None, help="merged 16-bit model, HF format (default: models/llaici-<model>-merged)")
    parser.add_argument("--output-dir", default=None, help="default: models/llaici-<model>-gguf")
    parser.add_argument(
        "--outtype", default="q8_0", choices=["f32", "f16", "bf16", "q8_0", "auto"],
        help="convert_hf_to_gguf.py's --outtype — DESIGN.md target is q8_0",
    )
    args = parser.parse_args()
    args.adapter_dir = args.adapter_dir or model_dir(args.model_name, "lora")
    args.merged_dir = args.merged_dir or model_dir(args.model_name, "merged")
    args.output_dir = args.output_dir or model_dir(args.model_name, "gguf")

    if not (Path(args.adapter_dir) / "adapter_config.json").exists():
        sys.exit(f"{args.adapter_dir}/adapter_config.json not found — run `make finetune` first "
                 "(and pass the same MODEL= if you trained another model)")
    convert_script = Path(args.llama_cpp_dir).expanduser() / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        sys.exit(
            f"{convert_script} not found: the GGUF export needs a local llama.cpp clone.\n"
            f"  git clone https://github.com/ggml-org/llama.cpp {args.llama_cpp_dir}\n"
            "or point to an existing one: make merge-and-quantize LLAMA_CPP_DIR=/path/to/llama.cpp"
        )

    # Step 1: merge the adapter into the 16-bit base model.
    merge(args.base_model_16bit or base_16bit(args.adapter_dir), args.adapter_dir, args.merged_dir)

    # Step 2: convert the merged HF directory straight to a quantized GGUF.
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    outfile = Path(args.output_dir) / f"llaici-{model_slug(args.model_name)}.{args.outtype}.gguf"
    cmd = [sys.executable, str(convert_script), args.merged_dir, "--outfile", str(outfile), "--outtype", args.outtype]
    print(f"# running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    print_box(
        f"Model ready — GGUF ({args.outtype})",
        [f"{outfile}  ({human_size(outfile)})"],
        [
            f"serve it:  llama-server -m {outfile} --port 8080  (see app/README.md)",
            f"the merged 16-bit model ({args.merged_dir}/, {human_size(args.merged_dir)}) is no longer needed: delete it to free space",
        ],
    )


if __name__ == "__main__":
    main()
