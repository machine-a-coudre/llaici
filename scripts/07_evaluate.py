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
import multiprocessing as mp
import queue
import random
import re
import time
from pathlib import Path

from console import print_box
from model_paths import model_dir

# Verdict thresholds on the share of exact matches (ok_match). A judgment call, not
# a standard: for a "type a question, get a map" tool, a wrong map 1 time in 5 is
# already noticeable.
GOOD_MATCH_PCT = 80
FAIR_MATCH_PCT = 50
# Below this many training examples, "train on more data" is the first advice.
SMALL_TRAIN_SET = 5000
TRAIN_FILE = "data/training/train_formatted.jsonl"
# Per-query time limits. The gold query runs first and is timed; the generated one
# gets TIMEOUT_FACTOR x that time (at least --query-timeout): a correct query takes
# about as long as the gold one, one taking 5x longer is badly filtered (e.g. a
# spatial join over a whole table). A fixed limit would either be too short for the
# heavy templates (left/right bank, along: computed over `water`) or too long for
# the light ones.
TIMEOUT_FACTOR = 5

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


def _query_worker(threads: int, requests, results) -> None:
    con = connect(threads)
    results.put("ready")
    while (sql := requests.get()) is not None:
        results.put(result_ids(con, sql))


class QueryRunner:
    """Runs SQL in a child process, killed and restarted on timeout.

    con.interrupt() isn't enough: DuckDB only checks for it at some internal
    checkpoints, which some spatial queries don't reach for minutes — same finding
    as scripts/01_sample_entities.py, which kills its attempts the same way."""

    def __init__(self, threads: int):
        self.threads = threads
        # spawn, not fork: the parent has CUDA initialized, which isn't fork-safe.
        self.ctx = mp.get_context("spawn")
        self._start()

    def _start(self) -> None:
        self.requests, self.results = self.ctx.Queue(), self.ctx.Queue()
        self.proc = self.ctx.Process(target=_query_worker, args=(self.threads, self.requests, self.results), daemon=True)
        self.proc.start()
        self.results.get()  # "ready": connection + spatial loaded, not counted in any timeout

    def run(self, sql: str, timeout: float) -> tuple[str, set[str] | None, float]:
        """(status, ids, seconds) — status "timeout" if the query didn't finish in time."""
        start = time.monotonic()
        self.requests.put(sql)
        try:
            status, ids = self.results.get(timeout=timeout)
        except queue.Empty:
            self.proc.kill()
            self.proc.join()
            self._start()
            return "timeout", None, time.monotonic() - start
        return status, ids, time.monotonic() - start

    def close(self) -> None:
        self.requests.put(None)
        self.proc.join(timeout=5)


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


def generate_sql(model, tokenizer, question: str, max_new_tokens: int) -> tuple[str, bool]:
    """Returns (sql, truncated): truncated when generation hit --max-new-tokens."""
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM},
        {"role": "user", "content": question},
    ]
    # return_dict: also returns the attention_mask, which generate() can't infer
    # itself since Qwen's pad token is its eos token (it warns otherwise).
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True)
    new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    generated = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return extract_sql(generated), len(new_tokens) >= max_new_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-name", default="unsloth/Qwen3-0.6B-unsloth-bnb-4bit")
    parser.add_argument("--chat-template", default="qwen3-instruct")
    parser.add_argument("--adapter-dir", default=None, help="default: models/llaici-<model>-lora, as written by scripts/06_finetune.py")
    parser.add_argument("--val-file", default="data/samples/val.jsonl")
    parser.add_argument("--out", default=None, help="per-example detail (default: eval_results.jsonl inside the adapter dir, next to the model it scores)")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    # The templates' SQL is long (7-language ILIKE chains): up to ~530 tokens, so 256
    # cut ~1/4 of the queries mid-statement, scored as syntax errors.
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--threads",
        type=int,
        default=2,
        help="DuckDB SET threads=N (default 2, kept low on purpose); tune to roughly "
        "the machine's CPU core count for a faster large run (e.g. --threads 10)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="evaluate a random sample of N validation questions instead of all of them: each "
        "one is a full generation, so ~2,000 questions (ROWS=5000) can take hours; ~300 gives "
        "a representative score in a fraction of the time (default: all)",
    )
    parser.add_argument("--seed", type=int, default=42, help="for --max-examples: same seed, same sample")
    parser.add_argument(
        "--query-timeout", type=float, default=30,
        help=f"minimum time limit (s) for a generated query; the actual limit is {TIMEOUT_FACTOR}x the gold "
        "query's time when that's longer. Past it, the answer counts as 'too slow'",
    )
    parser.add_argument(
        "--gold-timeout", type=float, default=300,
        help="time limit (s) for the gold query; past it, the question is skipped (not the model's fault)",
    )
    args = parser.parse_args()
    args.adapter_dir = args.adapter_dir or model_dir(args.model_name, "lora")
    args.out = args.out or f"{args.adapter_dir}/eval_results.jsonl"

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
    # Qwen3's generation_config.json is tuned for chat: sampling (temperature 0.6,
    # top_p/top_k) and max_length=40960. Evaluation wants greedy decoding, so a run
    # is reproducible and scores the model rather than the dice; and max_length
    # would clash with --max-new-tokens (a warning on every example).
    gen_config = model.generation_config
    gen_config.do_sample = False
    gen_config.temperature = gen_config.top_p = gen_config.top_k = None
    gen_config.max_length = None

    runner = QueryRunner(args.threads)

    counts = {"syntax_error": 0, "empty": 0, "timeout": 0, "ok_mismatch": 0, "ok_match": 0}
    truncated = 0
    skipped = 0
    with open(args.val_file, encoding="utf-8") as fin:
        val_rows = [json.loads(line) for line in fin]
    args.val_total = len(val_rows)
    if args.max_examples is not None and args.max_examples < len(val_rows):
        # Fixed seed: two models evaluated with the same --max-examples see the same
        # questions, so their scores are comparable.
        val_rows = random.Random(args.seed).sample(val_rows, args.max_examples)
        print(f"# evaluating a random sample of {len(val_rows)} of {args.val_total} validation questions (--max-examples)")

    with open(args.out, "w", encoding="utf-8") as fout:
        try:
            for row in val_rows:
                question, gold_sql = row["question"], row["sql"]
                # Gold first: its time sets the generated query's limit.
                gold_status, gold_ids, gold_s = runner.run(gold_sql, args.gold_timeout)
                generated_sql, was_truncated = generate_sql(model, tokenizer, question, args.max_new_tokens)
                truncated += was_truncated

                gen_status, gen_s = "not run", None
                if gold_status != "ok":
                    # Too slow, or no longer valid against the current data: can't be
                    # scored, and it's not the model's fault — left out of the score.
                    bucket = "skipped"
                    skipped += 1
                else:
                    gen_timeout = max(args.query_timeout, TIMEOUT_FACTOR * gold_s)
                    gen_status, gen_ids, gen_s = runner.run(generated_sql, gen_timeout)
                    if gen_status == "timeout":
                        bucket = "timeout"
                    elif gen_status.startswith("syntax_error"):
                        bucket = "syntax_error"
                    elif gen_status == "empty":
                        bucket = "empty"
                    elif gen_ids == gold_ids:
                        bucket = "ok_match"
                    else:
                        bucket = "ok_mismatch"
                    counts[bucket] += 1

                fout.write(json.dumps({
                    "question": question,
                    "gold_sql": gold_sql,
                    "generated_sql": generated_sql,
                    "gen_status": gen_status,
                    "gold_status": gold_status,
                    "bucket": bucket,
                    "truncated": was_truncated,
                    "gold_seconds": round(gold_s, 2),
                    "gen_seconds": None if gen_s is None else round(gen_s, 2),
                }, ensure_ascii=False) + "\n")
                fout.flush()  # so progress can be followed with wc -l
        finally:
            runner.close()

    print_verdict(counts, truncated, skipped, args)


def print_verdict(counts: dict[str, int], truncated: int, skipped: int, args: argparse.Namespace) -> None:
    """Colored verdict box, plus advice targeted at the most frequent failure."""
    total = sum(counts.values())
    pct = {b: 100 * n / total if total else 0 for b, n in counts.items()}
    match = pct["ok_match"]
    if match >= GOOD_MATCH_PCT:
        status, title = "ok", f"Evaluation: good — {match:.0f}% correct SQL"
    elif match >= FAIR_MATCH_PCT:
        status, title = "warn", f"Evaluation: fair — {match:.0f}% correct SQL"
    else:
        status, title = "fail", f"Evaluation: not good enough — {match:.0f}% correct SQL"

    labels = {
        "ok_match": "correct (same rows as expected)",
        "ok_mismatch": "wrong rows",
        "empty": "empty result",
        "syntax_error": "SQL error",
        "timeout": "too slow (timed out)",
    }
    rows = [f"{labels[b]:<32} {counts[b]:>5}  ({pct[b]:5.1f}%)" for b in ("ok_match", "ok_mismatch", "empty", "syntax_error", "timeout")]
    evaluated = total + skipped
    sampled = f" (random sample of {args.val_total})" if evaluated < args.val_total else ""
    rows += ["", f"{total} validation questions scored{sampled} — detail: {args.out}"]
    if skipped:
        rows.append(f"{skipped} skipped: gold query too slow (> {args.gold_timeout:.0f}s) or failed")

    advice = []
    if status != "ok":
        train_size = sum(1 for _ in open(TRAIN_FILE, encoding="utf-8")) if Path(TRAIN_FILE).exists() else None
        if train_size is not None and train_size < SMALL_TRAIN_SET:
            advice.append(f"only {train_size} training examples: generate more first (make generate-dataset ROWS=5000 or more)")
        if truncated:
            advice.append(f"{truncated} answers hit --max-new-tokens ({args.max_new_tokens}) and were cut off: raise it (.venv-finetune/bin/python scripts/07_evaluate.py --max-new-tokens 2048)")
        worst = max(("syntax_error", "empty", "ok_mismatch", "timeout"), key=counts.get)
        if counts[worst]:
            advice.append({
                "syntax_error": "mostly SQL errors (invented columns/tables): more data, or more epochs (make finetune EPOCHS=3)",
                "empty": "mostly empty results (place name or category altered): more data, check examples in the detail file",
                "ok_mismatch": "mostly wrong rows (wrong template/category): more data or epochs, check examples in the detail file",
                "timeout": "mostly too-slow queries (badly filtered spatial joins?): check examples in the detail file; THREADS=10 also speeds DuckDB up",
            }[worst])
        advice.append("what to do, in detail: docs/TROUBLESHOOTING.md \"Evaluation (step 07)\"")
    else:
        advice.append(f"export it: make merge-and-quantize MODEL={args.model_name} LLAMA_CPP_DIR=<llama.cpp clone>")
    print_box(title, rows, advice, status=status)


if __name__ == "__main__":
    main()
