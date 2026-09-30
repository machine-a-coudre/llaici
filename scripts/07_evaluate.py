#!/usr/bin/env python3
"""FINETUNING.md §5 — evaluate the fine-tuned model's *generated* SQL.

Loss curves aren't the real test for this project (FINETUNING.md §5): what matters
is whether the SQL the model actually generates parses, executes against DuckDB,
and returns the right answer — mirroring why TEMPLATES.md insists every hand-written
template be validated by execution, not just written and trusted.

For each {question, sql} pair in the held-out validation set (scripts/04_split_train_val.py
output — the *raw* val.jsonl, not the "messages"-formatted training copy), this script:
1. Feeds `question` to the fine-tuned model and gets back generated SQL.
2. Executes the generated SQL against the real DuckDB views (same database every
   other script in this project validates against).
3. Executes the *gold* SQL too, and compares the two result sets (by `id`, the
   first column every template SELECTs) — the strongest signal available: same
   ids back means the model didn't just produce *some* valid SQL, it produced
   the *right* one.

Reuses fill_and_validate.py's "execute and check" approach, applied to generated
SQL instead of hand-built template SQL (FINETUNING.md §5's own framing of what
this script should do).

⚠️ Requires the same CUDA + Unsloth environment as scripts/06_finetune.py, plus a
LoRA adapter already produced by it (--adapter-dir). Not runnable/tested in this
development environment — see that script's docstring for why. Loads the base
model + adapter directly via Unsloth for inference (FastLanguageModel.for_inference),
not through llama-server: that's FINETUNING.md §7, downstream of the merge/quantize
step (§6) this evaluation happens before.

Run (on a CUDA machine, after scripts/06_finetune.py has produced an adapter):
    python3 scripts/07_evaluate.py
    python3 scripts/07_evaluate.py --model-name unsloth/Qwen3-1.7B-unsloth-bnb-4bit --val-file data/samples/val.jsonl
"""

import argparse
import json
import re

from model_paths import model_dir

DB_PATH = "data/db/llaici.duckdb"

DEFAULT_SYSTEM = (
    "You translate a natural-language geographic question into a single DuckDB "
    "SQL query, using DuckDB's spatial extension functions (ST_*, list_contains, "
    "etc.) where needed. Respond with SQL only, no explanation."
)


def connect(threads: int = 2):
    import duckdb

    con = duckdb.connect(DB_PATH, read_only=True)
    con.execute(f"SET threads={threads}")  # see scripts/01_sample_entities.py's connect() note
    con.execute("INSTALL spatial")  # no-op once installed; host runs lack the image's pre-install
    con.execute("LOAD spatial")
    return con


def extract_sql(generated_text: str) -> str:
    """Best-effort cleanup of raw model output into a bare SQL statement.

    Strips markdown code fences (```sql ... ```) a chat model commonly wraps
    output in despite the system prompt asking for "SQL only" — not guaranteed
    to handle every way a 0.6B model might misbehave, just the common case.
    """
    text = generated_text.strip()
    fence_match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()
    return text


def result_ids(con, sql: str) -> tuple[str, set[str] | None]:
    """Runs `sql`, returns (status, set-of-ids-in-first-column or None).

    status is one of: "syntax_error", "empty", "ok".
    """
    try:
        rows = con.execute(sql).fetchall()
    except Exception as e:
        return f"syntax_error: {e}", None
    if not rows:
        return "empty", set()
    return "ok", {row[0] for row in rows}


def generate_sql(model, tokenizer, question: str, max_new_tokens: int) -> str:
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM},
        {"role": "user", "content": question},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    outputs = model.generate(input_ids=inputs, max_new_tokens=max_new_tokens, use_cache=True)
    generated = tokenizer.decode(outputs[0][inputs.shape[1] :], skip_special_tokens=True)
    return extract_sql(generated)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-name", default="unsloth/Qwen3-0.6B-unsloth-bnb-4bit")
    parser.add_argument("--chat-template", default="qwen3-instruct")
    parser.add_argument("--adapter-dir", default=None, help="default: models/llaici-<model>-lora, as written by scripts/06_finetune.py")
    parser.add_argument("--val-file", default="data/samples/val.jsonl")
    parser.add_argument("--out", default="data/samples/eval_results.jsonl")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--threads",
        type=int,
        default=2,
        help="DuckDB SET threads=N (default 2, kept low on purpose); tune to roughly "
        "the machine's CPU core count for a faster large run (e.g. --threads 10)",
    )
    args = parser.parse_args()
    args.adapter_dir = args.adapter_dir or model_dir(args.model_name, "lora")

    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template=args.chat_template)
    model.load_adapter(args.adapter_dir)
    FastLanguageModel.for_inference(model)

    con = connect(args.threads)

    counts = {"syntax_error": 0, "empty": 0, "ok_mismatch": 0, "ok_match": 0}
    with open(args.val_file, encoding="utf-8") as fin, open(args.out, "w", encoding="utf-8") as fout:
        for line in fin:
            row = json.loads(line)
            question, gold_sql = row["question"], row["sql"]

            generated_sql = generate_sql(model, tokenizer, question, args.max_new_tokens)
            gen_status, gen_ids = result_ids(con, generated_sql)
            gold_status, gold_ids = result_ids(con, gold_sql)

            if gen_status.startswith("syntax_error"):
                bucket = "syntax_error"
            elif gen_status == "empty":
                bucket = "empty"
            elif gold_status == "ok" and gen_ids == gold_ids:
                bucket = "ok_match"
            else:
                bucket = "ok_mismatch"
            counts[bucket] += 1

            fout.write(json.dumps({
                "question": question,
                "gold_sql": gold_sql,
                "generated_sql": generated_sql,
                "gen_status": gen_status,
                "bucket": bucket,
            }, ensure_ascii=False) + "\n")

    total = sum(counts.values())
    print(f"# {total} validation examples evaluated")
    for bucket, n in counts.items():
        pct = 100 * n / total if total else 0
        print(f"#   {bucket}: {n} ({pct:.1f}%)")
    print(f"# per-example detail written to {args.out}")


if __name__ == "__main__":
    main()
