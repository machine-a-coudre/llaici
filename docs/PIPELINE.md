# PIPELINE.md — command reference

Exact commands, flags, and defaults for every step in `README.md`'s pipeline table. See `DESIGN.md` for the overall design, `SCHEMA.md`/`TEMPLATES.md` for the data model and SQL templates, and `FINETUNING.md` for the fine-tuning steps' rationale.

---

## Setup — download Overture data

Downloads the raw Parquet extracts (`division`, `division_area`, `infrastructure`, `water`, `place`) into `data/parquet/` via `overturemaps`.

```bash
make download-overture
```

Default bbox is Western Europe (`-9.72,35.91,3.59,43.82`). Override it with:

```bash
make download-overture BBOX=xmin,ymin,xmax,ymax
```

---

## Step 00 — Build the DuckDB logical views (`scripts/00_init.sql`)

Creates the flat, stable views (`divisions`, `division_areas`, `infrastructures`, `water`, `places`) described in `SCHEMA.md`, reading from `data/parquet/`.

Runs automatically the first time the DuckDB container starts (see `entrypoint.sh`) if `data/db/llaici.duckdb` doesn't exist yet:

```bash
make up
```

To force a refresh on an existing database (e.g. after changing `00_init.sql`):

```bash
docker compose run --rm --entrypoint sh duckdb -c "duckdb /app/data/db/llaici.duckdb -c '.read /app/scripts/00_init.sql'"
```

### Equivalent raw `docker run` commands (without compose)

```bash
docker build --platform linux/amd64 -t llaici-duckdb .

docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/scripts:/app/scripts \
  llaici-duckdb \
  duckdb /app/data/db/llaici.duckdb -c ".read /app/scripts/00_init.sql"

docker run -it --rm -p 4213:4213 \
  -v $(pwd)/data:/app/data \
  llaici-duckdb \
  duckdb -ui /app/data/db/llaici.duckdb
```

Only `data/` and `scripts/` are bind-mounted into the container (at `/app/data` and `/app/scripts`) — the container never gets access to the rest of the project (source code, `.git`, etc.). `00_init.sql` uses paths relative to the repo root (`data/parquet/...`, `data/db/...`), which resolve identically whether DuckDB runs inside the container (`WORKDIR /app` mirrors that same relative layout) or directly on the host from the repo root (e.g. connecting from DBeaver).

---

## Step 01 — Sample entities (`scripts/01_sample_entities.py`)

Draws real, join-valid parameter sets from the DuckDB views to fill the SQL templates in `TEMPLATES.md` (see `DESIGN.md` STEP 4). Each template samples into its **own** file, `data/samples/entities_<name>.jsonl` (e.g. `entities_along.jsonl`, `entities_center.jsonl`) — never straight into the final `entities.jsonl`. With the default `TEMPLATE=all`, every template's file is (re)generated and then merged into `data/samples/entities.jsonl`.

```bash
make sample-entities ROWS=100      # quick test batch, all templates
make sample-entities ROWS=70000    # full batch for training, all templates
make sample-entities ROWS=70000 THREADS=10   # same, on a machine with CPU cores to spare
```

- `ROWS` — total sample rows across all templates (split evenly); defaults to 100 if omitted.
- Each sampling attempt runs in its own subprocess with a hard timeout (see the script's "per-attempt timeout" note) — a rare attempt that hangs is killed automatically after a few seconds and counted as a miss, rather than blocking the run.
- Only countries with at least one locality in `divisions` are sampled from (see `fetch_countries()`). The `--bbox` download also pulls in the country polygons of countries crossing the antimeridian (Russia, the US): their bbox spans every longitude, so it intersects any requested bbox, but none of their localities/infrastructures come along. Without this filter they'd be picked like any other country, wasting attempts — and for `bordering`, their huge polygons blew memory up to an OOM kill.
- Each attempt worker is also capped at half the machine's RAM (`WORKER_MEM_FRACTION`, via `RLIMIT_AS` + DuckDB `memory_limit`): a runaway query fails that attempt (counted as a miss) instead of triggering the OOM killer on the whole session.
- **Performance note**: both this script and step 02 default to `SET threads=2` (see `connect()` in each), deliberately low to stay light — overridable with `--threads`/`THREADS=` (both scripts, plus step 07). More RAM does **not** speed this up — DuckDB won't use more CPU cores than this setting regardless of available memory. **The right value depends on how many CPU cores the machine actually has** — pick a `--threads` value at or below that count (e.g. `THREADS=10` on a 10+ core machine); setting it higher than the core count doesn't help and can add contention.

### Regenerating a single template

Useful after tweaking one template's SQL or sampler (`TEMPLATES.md`) without waiting on a full re-run of every other template:

```bash
make sample-entities TEMPLATE=along ROWS=500   # regenerates only entities_along.jsonl
make merge-samples                             # rebuilds entities.jsonl from every entities_<name>.jsonl as-is
```

`TEMPLATE=<name>` (one of the `SAMPLERS` keys in `scripts/01_sample_entities.py`, e.g. `along`, `center`, `bordering`, `show_division`...) only ever overwrites that one template's own file — `entities.jsonl` is **left untouched** until an explicit merge. `make merge-samples` (`--template merge`) does that merge on its own, with no resampling — it just concatenates whichever `entities_<name>.jsonl` files already exist, in a fixed order, warning (not failing) about any that are missing.

If `entities.jsonl` already exists, both merge paths (`--template merge` and the automatic one after `--template all`) ask what to do with it before touching anything:

```
⚠  data/samples/entities.jsonl already exists.
[m]erge into it, [o]verwrite it, or [c]ancel? [m/o/c]:
```

- **`m` (merge)** — appends the fresh merge onto the existing file's content, keeping what was already there. Can produce duplicate rows if some of that content was already merged in before (harmless: `02_fill_and_validate.py` re-validates every row regardless, and `03_generate_questions.py` dedupes the final `(question, sql)` pairs downstream) — this is meant for combining genuinely different batches (e.g. runs with different `ROWS`/countries), not for "regenerated the same template, merge it back in".
- **`o` (overwrite)** — the previous default: a fresh full replace, discarding whatever `entities.jsonl` held before.
- **`c` (cancel)**, or anything else typed — aborts, `entities.jsonl` left exactly as it was; for the `--template all` path this is checked *before* sampling starts, so a cancelled run doesn't waste time regenerating every template's file first.

Pass `--yes`/`-y` to skip the prompt and always overwrite (e.g. for scripted/CI use) — the `sample-entities`/`generate-dataset` `make` targets already do this, since they're meant to run unattended; `make merge-samples` does not, so it always prompts.

---

## Step 02 — Fill templates and validate by execution (`scripts/02_fill_and_validate.py`)

Fills each sampled row's matching SQL template (see `TEMPLATES.md`) with its real values and executes it, keeping only pairs that return a non-empty result. Output: `data/samples/validated_samples.jsonl`.

**One row per language (FR, EN) per sampled entity, each with its own SQL.** Samplers return Overture's `names.primary`, the *local* name — "España", or "Maroc ⵍⵎⵖⵔⵉⴱ المغرب" (Morocco's primary name carries all three official scripts). Used as-is, FR questions read "où se trouve España" and the model never sees the names users actually type. So each entity's name is resolved to its `name_fr` / `name_en` exonym (one batched lookup per reference table), and the template is filled and executed once per language: the FR row matches `ILIKE '%Espagne%'`, the EN row `ILIKE '%Spain%'` — the SQL always contains the term typed in the question (the model doesn't translate, see `DESIGN.md` STEP 3). Details:
- **10% local names** (`LOCAL_NAME_RATIO=`, default 0.1): that share of (entity, language) rows keeps the local name even when an exonym exists, since some users do type "España".
- **No exonym** (most small towns): the local name is used — it's what everyone types anyway.
- **Non-Latin names are never used** (no one types "المغرب" in a FR/EN question): if neither the exonym nor the local name is in Latin script, that (entity, language) gets no row — counted as "no usable name" in the summary.
- Each row carries `lang` and `entity` (the line number in `entities.jsonl`), used by steps 03 and 04.

Not to be confused with the **validation set** (`val.jsonl` / `val_formatted.jsonl`, step 04): "validated" here means *the filled SQL was checked by execution* — a data-quality filter before the dataset is built. The validation set is the held-out split used to measure the model (eval loss during `06_finetune.py`, generated-SQL checks in `07_evaluate.py`).

```bash
make validate-samples
make validate-samples THREADS=10   # see step 01's performance note on --threads/THREADS
make validate-samples LOCAL_NAME_RATIO=0.2   # keep 20% local names instead of 10%
```

---

## Step 03 — Generate NL questions, FR + EN (`scripts/03_generate_questions.py`)

Fills hand-written phrase templates per relation type with each validated pair's real values — only in that row's `lang`, since its name and SQL are already language-specific (step 02) — deduplicates exact `(question, sql)` pairs, and (optionally) caps pairs per template so no single relation type dominates. Output: the final `{question, sql}` dataset, `data/samples/dataset.jsonl`.

```bash
make generate-questions
make generate-questions MAX_PER_TEMPLATE=5000   # cap pairs per template (useful at ROWS=70000 scale)
```

⚠️ **TODO (next step):** FR phrasings have no article before country/region names ("où se trouve Espagne" instead of "l'Espagne", "frontalières de Maroc" instead of "du Maroc") — see the TODO in `03_generate_questions.py`.

---

## Step 04 — Split into train/val (`scripts/04_split_train_val.py`)

Splits `dataset.jsonl` into train/val sets, stratified by template so a low-volume template (e.g. `bordering`) doesn't end up entirely absent from validation. The split is done **per sampled entity**: all the questions of an entity (FR and EN, every phrasing) go to the same side, even though its FR and EN SQL differ ("Espagne" vs "Spain", see step 02) — otherwise the model would have seen the same query, just with the other language's name, in training. Entities that produced an identical SQL (same place sampled twice) are grouped together too. `VAL_RATIO` is therefore a fraction of entities per template, not of pairs (`FINETUNING.md` §1c). The script warns if `val.jsonl` comes out empty or if a template has a single entity (entirely in train) — typically a too-small `ROWS`.

```bash
make split-dataset
make split-dataset VAL_RATIO=0.05   # default 0.1 (10%)
```

---

## Step 05 — Format for training (`scripts/05_format_for_training.py`)

Converts `train.jsonl`/`val.jsonl` into the `messages` (system/user/assistant) format `tokenizer.apply_chat_template()` / Unsloth expect — the exact Qwen ChatML tokens are applied later, at training time, by Qwen's own tokenizer, not hand-written here (`FINETUNING.md` §1a/1b).

```bash
make format-for-training
```

---

## Step 06 — Fine-tune, QLoRA on Qwen3-0.6B (`scripts/06_finetune.py`)

`FINETUNING.md` §3/§4. ⚠️ Requires an NVIDIA GPU with CUDA; Unsloth is installed in a dedicated venv by `make finetune-venv` (run automatically by `make finetune`) — unavailable in this development environment, so unlike every step above, **this one has not been run/tested**.

```bash
make finetune
# or directly, to override defaults:
python3 scripts/06_finetune.py --rank 16 --epochs 3
```

Saves a LoRA adapter to `models/llaici-qwen3-0.6b-lora/` — merging into the base model and GGUF quantization (step 08) is a separate, later step.

---

## Step 07 — Evaluate the fine-tuned model (`scripts/07_evaluate.py`)

`FINETUNING.md` §5. For each held-out validation question, generates SQL with the fine-tuned model and executes it against the real DuckDB views, comparing the result to the gold SQL's own result (not just "did it run"). Same CUDA/Unsloth requirement as step 06, and needs an adapter from that step first — **not run/tested** here either.

```bash
make evaluate
make evaluate THREADS=10   # see step 01's performance note on --threads/THREADS
```

Writes per-example detail (question, gold SQL, generated SQL, outcome) to `data/samples/eval_results.jsonl`, plus a summary: % syntax errors, % empty results, % that ran but returned the wrong rows, % exact match with the gold query's result.

---

## Step 08 — Merge and quantize to GGUF Q8 (`scripts/08_merge_and_quantize.py`)

`FINETUNING.md` §6. Merges the LoRA adapter into the base model and exports a single quantized GGUF file (`quantization_method=q8_0`, `DESIGN.md`'s ~800MB target), ready for `llama-server` (`FINETUNING.md` §7). Same CUDA/Unsloth requirement as steps 06-07 — **not run/tested** here either.

```bash
make merge-and-quantize
```

Writes the GGUF model to `models/llaici-qwen3-0.6b-gguf/`.
