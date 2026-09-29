#!/usr/bin/env python3
"""FINETUNING.md step 1c — split dataset.jsonl into train/val sets.

`dataset.jsonl` only contains `{question, sql}` (DESIGN.md: "the dataset contains only
{question, sql}"), with no template info to stratify by. This script recovers it by
looking each row's `sql` up in `validated_samples.jsonl` (every dataset.jsonl row was
generated from a validated_samples.jsonl row, and `sql` is unique per (template, params)
pair) — no need to re-run question generation.

Stratified by template, not a plain random split: a random split could by chance
leave a low-volume template (`bordering`, `places_containment`, `places_proximity`
— see TEMPLATES.md) entirely out of validation, making that template's quality
invisible to any evaluation done afterward (FINETUNING.md step 5).

Split by sampled entity, not by (question, sql) pair: every entity comes with
several questions (FR + EN phrasings, see 03_generate_questions.py), and its FR and
EN questions usually have *different* SQL ("Espagne" vs "Spain" in the ILIKE, see
02_fill_and_validate.py) — but the same query shape, reference and parameters. The
held-out unit is therefore the entity (`entity` in validated_samples.jsonl): all of
its questions, in both languages, go to the same side. A per-pair (or even per-SQL)
split would send e.g. the FR version to train and the EN one to val, so the model
had already seen the expected answer during training and val scores came out
optimistic (leakage). Entities sharing an identical SQL (the same place sampled
twice) are merged into one group too. `--val-ratio` is thus a fraction of groups
per template; the fraction of pairs follows approximately.

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
import sys
from collections import defaultdict


def load_sql_groups(validated_path: str) -> dict[str, tuple[str, int]]:
    """sql -> (template, group id). A group is an entity (its FR and EN rows), merged
    with any other entity producing an identical SQL (union-find over entity ids)."""
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        while parent.setdefault(x, x) != x:
            x = parent[x]
        return x

    sql_rows: dict[str, tuple[str, int]] = {}
    with open(validated_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            entity = row["entity"]
            if row["sql"] in sql_rows:
                parent[find(entity)] = find(sql_rows[row["sql"]][1])
            else:
                sql_rows[row["sql"]] = (row["template"], entity)
    return {sql: (template, find(entity)) for sql, (template, entity) in sql_rows.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="data/samples/dataset.jsonl")
    parser.add_argument("--validated", default="data/samples/validated_samples.jsonl")
    parser.add_argument("--train-out", default="data/samples/train.jsonl")
    parser.add_argument("--val-out", default="data/samples/val.jsonl")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="fraction of sampled entities held out per template for validation (all their questions, FR and EN, go to val)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sql_groups = load_sql_groups(args.validated)

    # template -> group (entity) -> every (question, sql) row of that group
    by_template: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    unknown = 0
    with open(args.dataset, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            template, group = sql_groups.get(row["sql"], (None, -1))
            if template is None:
                # Shouldn't happen in normal use (every dataset.jsonl row comes from
                # validated_samples.jsonl) — falls back to an "_unknown" bucket rather than
                # crashing, e.g. if dataset.jsonl and validated_samples.jsonl are out of sync.
                unknown += 1
                template = "_unknown"
            by_template[template][group].append(row)

    rng = random.Random(args.seed)
    train_rows, val_rows = [], []
    per_template_counts = {}
    for template, groups in by_template.items():
        # Sorted before shuffling so the split only depends on --seed, not on the
        # dataset.jsonl line order.
        keys = sorted(groups)
        rng.shuffle(keys)
        # At least 1 val entity per template (when there's more than 1 to begin
        # with) — the whole point of stratifying, otherwise a small template could
        # round down to 0 val examples and disappear from evaluation entirely.
        n_val = max(1, round(len(keys) * args.val_ratio)) if len(keys) > 1 else 0
        val_part = [row for key in keys[:n_val] for row in groups[key]]
        train_part = [row for key in keys[n_val:] for row in groups[key]]
        per_template_counts[template] = (len(keys), n_val, len(train_part), len(val_part))
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
          + (f", {unknown} unmatched rows (dataset.jsonl/validated_samples.jsonl out of sync?)" if unknown else ""))
    for template, (n_groups, n_val_groups, n_train, n_val) in sorted(per_template_counts.items()):
        print(f"#   {template}: {n_groups} entities -> {n_groups - n_val_groups} train / {n_val_groups} val "
              f"({n_train} / {n_val} pairs)")

    # A template with a single entity can't be split (see n_val above): it goes
    # entirely to train and is invisible to evaluation. Typically a too-small --rows
    # at sampling (e.g. ROWS=10 over 15 templates -> 1 entity per template).
    single = sorted(t for t, (n_groups, _, _, _) in per_template_counts.items() if n_groups <= 1)
    if not val_rows:
        print(
            f"# WARNING: {args.val_out} is EMPTY — every template has at most 1 entity, "
            "so nothing could be held out. Resample with more rows (at least ~2 validated entities "
            "per template, e.g. make generate-dataset ROWS=150).",
            file=sys.stderr,
        )
    elif single:
        print(
            f"# WARNING: {len(single)} template(s) with only 1 entity, entirely in train "
            f"(no val coverage): {', '.join(single)}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
