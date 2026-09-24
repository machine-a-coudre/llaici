#!/usr/bin/env python3
"""FINETUNING.md step 1c — split dataset.jsonl into train/val sets.

`dataset.jsonl` only contains `{question, sql}` (DESIGN.md: "the dataset contains only
{question, sql}"), with no template info to stratify by. This script recovers it by
looking each row's `sql` up in `validated.jsonl` (every dataset.jsonl row was
generated from a validated.jsonl row, and `sql` is unique per (template, params)
pair) — no need to re-run question generation.

Stratified by template, not a plain random split: a random split could by chance
leave a low-volume template (`bordering`, `places_containment`, `places_proximity`
— see TEMPLATES.md) entirely out of validation, making that template's quality
invisible to any evaluation done afterward (FINETUNING.md step 5).

Not implemented here (FINETUNING.md steps 1a/1b, still open questions): applying a
chat/prompt template to each {question, sql} pair — that depends on which base
model is chosen (Qwen vs Gemma), not yet decided. This script only produces the
train/val split of the raw pairs; prompt formatting is a separate step once a model
is picked.

Run:
    python3 scripts/04_split_train_val.py
    python3 scripts/04_split_train_val.py --val-ratio 0.1 --seed 42
"""

import argparse
import json
import random
from collections import defaultdict


def load_sql_to_template(validated_path: str) -> dict[str, str]:
    mapping = {}
    with open(validated_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            mapping[row["sql"]] = row["template"]
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="data/samples/dataset.jsonl")
    parser.add_argument("--validated", default="data/samples/validated.jsonl")
    parser.add_argument("--train-out", default="data/samples/train.jsonl")
    parser.add_argument("--val-out", default="data/samples/val.jsonl")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="fraction held out per template for validation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sql_to_template = load_sql_to_template(args.validated)

    by_template: dict[str, list[dict]] = defaultdict(list)
    unknown = 0
    with open(args.dataset, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            template = sql_to_template.get(row["sql"])
            if template is None:
                # Shouldn't happen in normal use (every dataset.jsonl row comes from
                # validated.jsonl) — falls back to an "_unknown" bucket rather than
                # crashing, e.g. if dataset.jsonl and validated.jsonl are out of sync.
                unknown += 1
                template = "_unknown"
            by_template[template].append(row)

    rng = random.Random(args.seed)
    train_rows, val_rows = [], []
    per_template_counts = {}
    for template, rows in by_template.items():
        rng.shuffle(rows)
        # At least 1 val example per template (when there's more than 1 row to begin
        # with) — the whole point of stratifying, otherwise a small template could
        # round down to 0 val examples and disappear from evaluation entirely.
        n_val = max(1, round(len(rows) * args.val_ratio)) if len(rows) > 1 else 0
        per_template_counts[template] = (len(rows), n_val)
        val_rows.extend(rows[:n_val])
        train_rows.extend(rows[n_val:])

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)

    with open(args.train_out, "w", encoding="utf-8") as f:
        for row in train_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(args.val_out, "w", encoding="utf-8") as f:
        for row in val_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"# {len(train_rows)} train, {len(val_rows)} val, {len(by_template)} templates"
          + (f", {unknown} unmatched rows (dataset.jsonl/validated.jsonl out of sync?)" if unknown else ""))
    for template, (total, n_val) in sorted(per_template_counts.items()):
        print(f"#   {template}: {total} total -> {total - n_val} train / {n_val} val")


if __name__ == "__main__":
    main()
