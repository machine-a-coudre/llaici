#!/usr/bin/env python3
"""FINETUNING.md §3/§4, Apple Silicon alternative to scripts/06_finetune.py.

scripts/06_finetune.py uses Unsloth, which requires an NVIDIA CUDA GPU — it will
not run on a Mac (Unsloth's own kernels are CUDA/Triton-based; MPS support is
still in progress upstream, see https://github.com/unslothai/unsloth/issues/2608).
This script does the same job (QLoRA fine-tune Qwen3-0.6B on {question, sql}
pairs) using `mlx-lm` instead, which runs natively on Apple Silicon via MLX.

API/behavior verified against mlx-lm's own docs/source rather than guessed
(`mlx_lm/LORA.md`, `mlx_lm/lora.py` on the `ml-explore/mlx-lm` GitHub repo):
- No documented Python training API — `LORA.md` explicitly only documents CLI
  usage (`mlx_lm.lora`, `mlx_lm.fuse`, `mlx_lm.generate`). This script is a thin
  wrapper around the `mlx_lm.lora` CLI (invoked via subprocess), not a
  reimplementation against an undocumented internal API.
- `mlx_lm.lora`'s data loader requires an exact filename convention — a `train.jsonl`
  (required) and `valid.jsonl` (optional, note: *not* `val.jsonl`) inside the
  directory passed to `--data`. scripts/05_format_for_training.py's output
  (`train_formatted.jsonl` / `val_formatted.jsonl`) is copied into a `mlx/`
  subdirectory under those exact names before training — not renamed in place,
  to avoid touching the STEP 4 pipeline's own output files.
- LoRA hyperparameters (`rank`, `dropout`, `scale` — mlx-lm's name for what
  Unsloth/PEFT call `lora_alpha`) have **no CLI flag**: `mlx_lm/lora.py`'s
  `build_parser()` only exposes training-loop flags (`--iters`, `--batch-size`,
  `--learning-rate`, ...); `lora_parameters` is only settable via the `-c/--config`
  YAML file. This script always writes a temporary YAML (all options in one place,
  not a CLI-flags/YAML-config split) and runs `mlx_lm.lora --config <file>`.
- `--mask-prompt` (loss computed only on the assistant's SQL completion, not the
  question) defaults to **on** here, unlike scripts/06_finetune.py's SFTTrainer
  call which doesn't mask — a deliberate difference, not a port bug: this task
  only cares about the SQL half of each example, so wasting training signal on
  reconstructing the *question* (which the model doesn't need to generate) is
  avoidable here since mlx-lm exposes the option directly.
- Qwen3 support: `mlx_lm/lora.py`'s own `CONFIG_DEFAULTS["model"]` is literally
  `"Qwen/Qwen3-0.6b"` (mlx-lm's own default), and real Qwen3 LoRA fine-tunes are
  documented in the wild (e.g. github.com/sciences44/mlx-lora-finetune, a
  Text-to-SQL MLX LoRA fine-tune of Qwen3.5) — despite `LORA.md`'s prose model
  list (Mistral/Llama/Phi2/Mixtral/Qwen2/Gemma/OLMo/MiniCPM/InternLM2) not
  mentioning Qwen3 by name; that list is stale documentation, not a hard block.
  ⚠️ Known rough edge: ml-explore/mlx#2616 reports Qwen3 LoRA training covering
  fewer trainable parameters than expected on some mlx-lm versions — check
  printed trainable-parameter count/percentage against the `--rank` requested if
  results look off.

⚠️ Not run/tested in this development environment (no Apple Silicon Mac available
here either) — written against documented mlx-lm behavior, same caveat
scripts/06_finetune.py has for its own (CUDA) requirement.

Output is a LoRA adapter directory (`--adapter-path`), same role as
scripts/06_finetune.py's output. Merging + GGUF export is a separate step, not
implemented here: `mlx_lm.fuse` produces a Hugging Face-compatible safetensors
directory (documented), but `mlx_lm.fuse --export-gguf` is documented as
"limited to Mistral, Mixtral, and Llama style models" — Qwen3 isn't in that
list, so scripts/08_merge_and_quantize.py's GGUF-export role would need
`mlx_lm.fuse` (merge only, HF format out) followed by llama.cpp's own
`convert_hf_to_gguf.py` + `llama-quantize`, not `mlx_lm.fuse --export-gguf`.

Run (on a Mac, after `pip install "mlx-lm[train]"`):
    python3 scripts/06_finetune_mlx.py
    python3 scripts/06_finetune_mlx.py --rank 16 --iters 500 --learning-rate 1e-4
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from model_paths import model_dir

# Deferred: PyYAML ships with mlx-lm's own dependencies (mlx_lm/lora.py imports it
# directly) but isn't otherwise a project dependency — kept at call time, not
# module level, so this file can still be inspected/linted without mlx-lm
# installed, same reasoning as scripts/06_finetune.py's deferred unsloth import.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="mlx-community/Qwen3-0.6B-4bit", help="QLoRA base model (quantized MLX checkpoint)")
    parser.add_argument("--data-dir", default="data/training", help="dir holding train_formatted.jsonl/val_formatted.jsonl (scripts/05)")
    parser.add_argument("--adapter-path", default=None, help="default: models/llaici-<model>-mlx-lora (see model_paths.py)")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    # LoRA (DESIGN.md defaults: rank 16-32, lr 2e-4 — same target as scripts/06_finetune.py's
    # --rank/--lr, mlx-lm's own defaults (rank 8, lr 1e-5) are more conservative).
    parser.add_argument("--rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-scale", type=float, default=20.0, help="mlx-lm's name for LoRA alpha/scale (default per mlx_lm.lora)")
    parser.add_argument("--iters", type=int, default=600, help="training iterations (mlx-lm counts steps, not epochs)")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=16, help="layers to fine-tune, -1 for all")
    parser.add_argument("--val-batches", type=int, default=25)
    parser.add_argument("--steps-per-report", type=int, default=10)
    parser.add_argument("--steps-per-eval", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--grad-checkpoint", action="store_true", help="trade compute for memory (see mlx-lm LORA.md 'Memory Issues')")
    parser.add_argument("--no-mask-prompt", dest="mask_prompt", action="store_false", help="compute loss on the whole example, not just the SQL completion")
    parser.set_defaults(mask_prompt=True)
    args = parser.parse_args()
    args.adapter_path = args.adapter_path or model_dir(args.model, "mlx-lora")
    return args


def prepare_mlx_data_dir(data_dir: Path) -> Path:
    """mlx_lm.lora requires exact filenames (train.jsonl/valid.jsonl) in its --data
    directory — copies scripts/05's output there under those names rather than
    renaming in place, so the STEP 4 pipeline's own files are left untouched."""
    train_src = data_dir / "train_formatted.jsonl"
    val_src = data_dir / "val_formatted.jsonl"
    for src in (train_src, val_src):
        if not src.exists():
            raise FileNotFoundError(f"{src} not found — run `make format-for-training` (scripts/05) first")

    mlx_data_dir = data_dir / "mlx"
    mlx_data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(train_src, mlx_data_dir / "train.jsonl")
    shutil.copyfile(val_src, mlx_data_dir / "valid.jsonl")
    return mlx_data_dir


def main() -> None:
    import yaml

    args = parse_args()

    if shutil.which("mlx_lm.lora") is None:
        sys.exit("mlx_lm.lora not found on PATH — install it with: pip install \"mlx-lm[train]\"")

    mlx_data_dir = prepare_mlx_data_dir(Path(args.data_dir))

    config = {
        "model": args.model,
        "train": True,
        "fine_tune_type": "lora",
        "data": str(mlx_data_dir),
        "seed": args.seed,
        "num_layers": args.num_layers,
        "batch_size": args.batch_size,
        "iters": args.iters,
        "val_batches": args.val_batches,
        "learning_rate": args.learning_rate,
        "steps_per_report": args.steps_per_report,
        "steps_per_eval": args.steps_per_eval,
        "adapter_path": args.adapter_path,
        "save_every": args.save_every,
        "max_seq_length": args.max_seq_length,
        "grad_checkpoint": args.grad_checkpoint,
        "mask_prompt": args.mask_prompt,
        "lora_parameters": {
            "rank": args.rank,
            "dropout": args.lora_dropout,
            "scale": args.lora_scale,
        },
    }

    Path(args.adapter_path).mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(config, f)
        config_path = f.name

    print(f"# mlx_lm.lora config written to {config_path}")
    try:
        subprocess.run(["mlx_lm.lora", "--config", config_path], check=True)
    finally:
        Path(config_path).unlink(missing_ok=True)

    print(f"# LoRA adapter saved to {args.adapter_path}")


if __name__ == "__main__":
    main()
