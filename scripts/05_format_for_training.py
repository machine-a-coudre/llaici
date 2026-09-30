#!/usr/bin/env python3
"""FINETUNING.md steps 1a/1b — format train/val pairs for Qwen fine-tuning.

Converts {question, sql} pairs (scripts/04_split_train_val.py output) into the
"messages" conversation format HuggingFace's `tokenizer.apply_chat_template()` /
Unsloth expect: one JSONL row per example, each a list of {role, content} turns.

Deliberately does NOT hand-write Qwen's ChatML special tokens (<|im_start|>, etc.)
here. At training time, Qwen's own tokenizer applies its chat template to these
messages, guaranteeing the exact format the checkpoint was pretrained to expect —
hand-rolling the ChatML string ourselves would risk subtly diverging from that
(wrong newline, missing EOS token...). See FINETUNING.md 1a: "check that model's
official recommended chat template ... rather than guessing a generic one" — using
`messages` + the model's own tokenizer *is* that check, deferred to training time
instead of guessed here.

A short system message frames the task (SQL only, no explanation) so behavior is
anchored consistently across every example rather than left implicit — pass
--system "" to omit it.

Reads from data/samples/ (intermediate pipeline files) but writes to data/training/,
so the only files the fine-tuning step (06) actually consumes live apart from the rest.

Run:
    python3 scripts/05_format_for_training.py
    python3 scripts/05_format_for_training.py --in data/samples/train.jsonl --out data/training/train_formatted.jsonl
"""

import argparse
import json
from pathlib import Path

DEFAULT_SYSTEM = (
    "You translate a natural-language geographic question into a single DuckDB "
    "SQL query, using DuckDB's spatial extension functions (ST_*, list_contains, "
    "etc.) where needed. Respond with SQL only, no explanation."
)


def to_messages(question: str, sql: str, system: str | None) -> list[dict]:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": question})
    messages.append({"role": "assistant", "content": sql})
    return messages


def convert_file(in_path: str, out_path: str, system: str | None) -> int:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(in_path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            row = json.loads(line)
            messages = to_messages(row["question"], row["sql"], system)
            fout.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", default=None, help="single input file (default: format both train.jsonl and val.jsonl)")
    parser.add_argument("--out", dest="out_path", default=None, help="output file, required if --in is given")
    parser.add_argument("--dir", default="data/samples", help="directory holding train.jsonl/val.jsonl when --in is omitted")
    parser.add_argument("--out-dir", default="data/training", help="where train_formatted.jsonl/val_formatted.jsonl go when --in is omitted")
    parser.add_argument("--system", default=DEFAULT_SYSTEM, help="system message prepended to every example (pass \"\" to omit)")
    args = parser.parse_args()

    system = args.system if args.system else None

    if args.in_path:
        if not args.out_path:
            parser.error("--out is required when --in is given")
        n = convert_file(args.in_path, args.out_path, system)
        print(f"# {n} examples written to {args.out_path}")
        return

    written = []
    for split in ("train", "val"):
        in_path = f"{args.dir}/{split}.jsonl"
        out_path = f"{args.out_dir}/{split}_formatted.jsonl"
        written.append((split, out_path, convert_file(in_path, out_path, system)))
    print_summary(written)


def print_summary(written: list[tuple[str, str, int]]) -> None:
    """Hard-to-miss box pointing at the files step 06 will train on (same ANSI
    style as the Makefile's PRINT_THREADS reminder)."""
    bold_green, cyan, dim, reset = "\033[1;32m", "\033[36m", "\033[2m", "\033[0m"
    rows = [f"{split:<5}  {path}  ({n} examples)" for split, path, n in written]
    title = "✔  Training files ready"
    width = max(len(title), *(len(r) for r in rows)) + 2
    print()
    print(f"{bold_green}╭{'─' * (width + 2)}╮{reset}")
    print(f"{bold_green}│ {title:<{width}} │{reset}")
    print(f"{bold_green}├{'─' * (width + 2)}┤{reset}")
    for r in rows:
        print(f"{bold_green}│{reset} {cyan}{r:<{width}}{reset} {bold_green}│{reset}")
    print(f"{bold_green}╰{'─' * (width + 2)}╯{reset}")
    print(f"{dim}   → next: make finetune (CUDA) or make finetune-mlx (Apple Silicon){reset}")
    print()


if __name__ == "__main__":
    main()
