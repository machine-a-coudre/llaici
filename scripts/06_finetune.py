#!/usr/bin/env python3
"""FINETUNING.md §3/§4 — configure and run QLoRA fine-tuning on Qwen3-0.6B.

⚠️ Requires an NVIDIA GPU with CUDA. Unsloth does not support CPU-only training —
this script targets DESIGN.md's stated hardware (a consumer-grade CUDA GPU).
Tested end to end on an NVIDIA RTX 50xx (Blackwell) GPU with a small dataset (~100
training examples): trains, evaluates each epoch and saves the adapter. Not yet run
at full dataset scale.

Model/template choices, verified against Unsloth's own docs/source rather than
guessed (DESIGN.md's original "Qwen 3.5 0.8B" doesn't correspond to any real
published Qwen release — no such model or size exists):
- Base model: `unsloth/Qwen3-0.6B-unsloth-bnb-4bit` — the Instruct-tuned 4-bit
  checkpoint (not the "-Base-" variant, which lacks chat-formatting ability the
  `messages`-format dataset here relies on).
- Chat template key: `"qwen3-instruct"`, confirmed present in
  `unsloth/chat_templates.py` (CHAT_TEMPLATES dict) — deliberately not
  `"qwen3-thinking"`, since this task wants direct SQL output, not a <think>
  reasoning trace.
- LoRA defaults (`target_modules`, dropout, bias, use_gradient_checkpointing,
  use_rslora) are Unsloth's own general recommendation, not benchmarked against
  this project's dataset.

Run (on a CUDA machine, after `pip install unsloth`):
    python3 scripts/06_finetune.py
    python3 scripts/06_finetune.py --rank 16 --epochs 3 --lr 1e-4

Input: data/training/train_formatted.jsonl / val_formatted.jsonl (scripts/05_format_for_training.py),
each row {"messages": [{"role": ..., "content": ...}, ...]}.
Output: the LoRA adapter only, in --output-dir (not merged — merge/quantize is
FINETUNING.md §6, a separate step: scripts/08_merge_and_quantize.py).
"""

import argparse

from model_paths import model_dir, model_slug
from model_picker import choose_model
from console import human_size, print_box

DEFAULT_MODEL = "unsloth/Qwen3-0.6B-unsloth-bnb-4bit"

# Deferred: only importable on a CUDA machine with unsloth/torch/trl/peft/
# bitsandbytes installed — kept at call time, not module level, so this file can
# still be inspected/linted on this dev machine without those dependencies.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-name", default=None, help=f"Hugging Face id; omitted: pick among the models already on this machine, else {DEFAULT_MODEL}")
    parser.add_argument("--chat-template", default="qwen3-instruct")
    parser.add_argument("--train-file", default="data/training/train_formatted.jsonl")
    parser.add_argument("--val-file", default="data/training/val_formatted.jsonl")
    parser.add_argument("--output-dir", default=None, help="default: models/llaici-<model>-lora (see model_paths.py)")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    # LoRA (DESIGN.md defaults: rank 16-32, lr 2e-4, 2 epochs — see FINETUNING.md §3)
    parser.add_argument("--rank", type=int, default=32, help="LoRA rank (Unsloth recommends 16 or 32)")
    parser.add_argument("--lora-alpha", type=int, default=None, help="default: 2x --rank, per Unsloth's guidance")
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    if args.model_name is None:
        args.model_name = choose_model(DEFAULT_MODEL)
        print(f"# base model: {args.model_name} — pass the same MODEL= to `make evaluate` / `make merge-and-quantize`")
        if not model_slug(args.model_name).startswith("qwen3") and args.chat_template == "qwen3-instruct":
            print("# ⚠️  not a Qwen3 model: pass a matching --chat-template (make ... CHAT_TEMPLATE=)")
    args.output_dir = args.output_dir or model_dir(args.model_name, "lora")
    return args


def main() -> None:
    args = parse_args()
    lora_alpha = args.lora_alpha or args.rank * 2

    # unsloth first: it patches trl/transformers/peft on import, and warns (possibly
    # slower training / more memory) if they were already imported.
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
    )

    tokenizer = get_chat_template(tokenizer, chat_template=args.chat_template)

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )

    def formatting_prompts_func(examples):
        texts = [
            tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
            for convo in examples["messages"]
        ]
        return {"text": texts}

    train_dataset = load_dataset("json", data_files=args.train_file, split="train")
    train_dataset = train_dataset.map(formatting_prompts_func, batched=True)

    val_dataset = load_dataset("json", data_files=args.val_file, split="train")
    val_dataset = val_dataset.map(formatting_prompts_func, batched=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=args.max_seq_length,
            packing=False,
            per_device_train_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            eval_strategy="epoch",
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=args.seed,
            output_dir=args.output_dir,
        ),
    )

    trainer.train()

    # Saves the LoRA adapter only (small — a few tens of MB). Merging into the
    # base model and GGUF quantization is FINETUNING.md §6, a separate step.
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    model_arg = f" MODEL={args.model_name}"
    print_box(
        "Fine-tuning done — LoRA adapter saved",
        [
            f"adapter     {args.output_dir}/  ({human_size(args.output_dir)})",
            f"base model  {args.model_name}",
        ],
        [
            f"check its SQL:      make evaluate{model_arg}",
            f"export to GGUF:     make merge-and-quantize{model_arg}",
        ],
    )


if __name__ == "__main__":
    main()
