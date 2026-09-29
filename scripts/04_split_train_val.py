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

Split by SQL, not by (question, sql) pair: every SQL query comes with several
questions (one FR + one EN phrasing at least, see 03_generate_questions.py). The
held-out unit is therefore the SQL query: all of its questions go to the same side.
A per-pair split used to send e.g. the FR question to train and the EN one to val,
so the model had already seen the exact expected SQL during training and val scores
came out optimistic (leakage). `--val-ratio` is thus a fraction of distinct SQL
queries per template; the fraction of pairs follows approximately.

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
    parser.add_argument("--val-ratio", type=float, default=0.1, help="fraction of distinct SQL queries held out per template for validation (all their questions go to val)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sql_to_template = load_sql_to_template(args.validated)

    # template -> sql -> every (question, sql) row sharing that sql
    by_template: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
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
            by_template[template][row["sql"]].append(row)

    rng = random.Random(args.seed)
    train_rows, val_rows = [], []
    per_template_counts = {}
    for template, groups in by_template.items():
        # Sorted before shuffling so the split only depends on --seed, not on the
        # dataset.jsonl line order.
        sqls = sorted(groups)
        rng.shuffle(sqls)
        # At least 1 val SQL per template (when there's more than 1 SQL to begin
        # with) — the whole point of stratifying, otherwise a small template could
        # round down to 0 val examples and disappear from evaluation entirely.
        n_val = max(1, round(len(sqls) * args.val_ratio)) if len(sqls) > 1 else 0
        val_part = [row for sql in sqls[:n_val] for row in groups[sql]]
        train_part = [row for sql in sqls[n_val:] for row in groups[sql]]
        per_template_counts[template] = (len(sqls), n_val, len(train_part), len(val_part))
        val_rows.extend(val_part)
        train_rows.extend(train_part)

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
    for template, (n_sql, n_val_sql, n_train, n_val) in sorted(per_template_counts.items()):
        print(f"#   {template}: {n_sql} SQL -> {n_sql - n_val_sql} train / {n_val_sql} val "
              f"({n_train} / {n_val} pairs)")


if __name__ == "__main__":
    main()
