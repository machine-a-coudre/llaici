# FINETUNING.md — STEP 5: fine-tuning the model

This document details `DESIGN.md` STEP 5 (fine-tuning), which starts once STEP 4's `dataset.jsonl` exists (`{question, sql}` pairs — see `make sample-entities` / `validate-samples` / `generate-questions`). Nothing here is implemented yet; this describes the steps and the decisions each one requires.

⚠️ Concept to keep in mind throughout (`DESIGN.md`): fine-tuning anchors the model to `SCHEMA.md`'s view/column names. It does not teach the model geography — it teaches it to translate a question into the right SQL *shape*, filling in names/values from the question itself. A model that has never heard of "Lyon" can still generate correct SQL for "restaurants in Lyon", because it learned the pattern, not the place.

---

## 1. Prepare the dataset

`dataset.jsonl` is raw `{question, sql}` pairs — not yet in a form a training framework can consume directly.

### 1a. Pick a prompt format

**Decided: Qwen** (§2) — chat/ChatML format, not Alpaca-style instruction/input/output.

Rather than hand-writing Qwen's ChatML special tokens (`<|im_start|>`, `<|im_end|>`, ...) into the data file, the format used is the **`messages` list** (`[{"role": "system"/"user"/"assistant", "content": ...}]`) that `tokenizer.apply_chat_template()` / Unsloth consume directly. The actual ChatML string gets produced later, at training time, by Qwen's own tokenizer — guaranteed to match exactly what the checkpoint was pretrained on, rather than risking a subtly-wrong hand-rolled version (missing newline, wrong EOS token, etc.). This is what "check that model's official recommended chat template before picking" (previous revision of this doc) resolves to in practice: defer to the tokenizer instead of guessing the template ourselves.

### 1b. Convert `dataset.jsonl` into that format

**Implemented**: `scripts/05_format_for_training.py` (`make format-for-training`). Reads `train.jsonl`/`val.jsonl` (from §1c) and writes `train_formatted.jsonl`/`val_formatted.jsonl`, each row a `{"messages": [...]}` object: a system turn (short, fixed instruction — "respond with SQL only, no explanation"; override via `--system`, or `--system ""` to omit), a user turn (the question), and an assistant turn (the SQL). Verified: 576 train / 63 val examples converted, structure spot-checked.

### 1c. Split train / validation

**Implemented**: `scripts/04_split_train_val.py` (`make split-dataset`, `VAL_RATIO=` to override the default 10%). Reads `dataset.jsonl` (`{question, sql}` only, no template) and `validated.jsonl` (has `template`), joins them back together via the `sql` string (unique per template/params pair), then splits **per template** rather than on the pooled dataset — guarantees at least 1 validation example for every template, however small (verified: `bordering` at 18 total rows still got 2 held out, instead of a plain random split risking 0). Output: `data/samples/train.jsonl` and `data/samples/val.jsonl`, both still in the raw `{question, sql}` shape.

**Split unit = the SQL query, not the (question, sql) pair.** Each SQL query has several questions (at least one FR + one EN phrasing from `03_generate_questions.py`), and all of them go to the same side. The first version split per pair, so the FR question of a query could land in train and its EN question in val: the model had already seen the exact expected SQL during training, and val scores were optimistic (leakage). Consequence: `VAL_RATIO` is now a fraction of distinct SQL queries per template (still at least 1 per template when it has more than 1); the share of *pairs* in val follows it only approximately. The script prints both counts (SQL and pairs) per template.

What this still doesn't prevent: the same *place* (e.g. "Lyon") can appear in a train query and a different val query (another template, another category). That's intended — val measures generalization to new (template, params) combinations, not to never-seen place names.

Not addressed: explicit FR/EN balance in the split (only template-stratified) — `03_generate_questions.py` already produces a roughly fixed number of EN/FR phrasings per validated pair, so language balance should hold approximately without extra work, but this isn't verified/enforced by the split script itself.

**Deliverable**: `train.jsonl`/`val.jsonl` (raw pairs) plus, after §1a/1b, `train_formatted.jsonl`/`val_formatted.jsonl` — ready to hand to the training framework.

---

## 2. Choose the base model

`DESIGN.md` named two candidates, both small enough to fine-tune cheaply and run on CPU at inference (STEP 6): "Qwen 3.5 0.8B" and Gemma 3 270M. Qwen was picked over Gemma without a head-to-head comparison — a direct choice, not a benchmarked one.

⚠️ **Correction**: "Qwen 3.5 0.8B" doesn't correspond to any real published model — there is no "Qwen 3.5" release (the actual families are Qwen, Qwen1.5, Qwen2, Qwen2.5, Qwen3), and no official "0.8B" size either. Verified against Unsloth's own model listing before picking a real substitute:

| Model | Size | Notes |
|---|---|---|
| **`unsloth/Qwen3-0.6B-unsloth-bnb-4bit`** ✅ chosen | 0.6B params | Real, current Qwen3 release, closest official size to `DESIGN.md`'s intended ~0.8B. The **Instruct**-tuned variant (not `-Base-`) — needed since the `messages`-format dataset (§1b) relies on chat-formatting ability a base (non-instruct) checkpoint lacks. |
| Gemma 3 270M | 270M params | Not pursued |

§1a's chat template and §1b's formatting script are built specifically for Qwen's `messages`/ChatML convention off the back of this choice.

---

## 3. Configure LoRA / QLoRA fine-tuning

**Implemented**: `scripts/06_finetune.py` (`make finetune`).

- **Unsloth**: the fine-tuning library `DESIGN.md` names, chosen for VRAM efficiency and speed on consumer GPUs.
- **QLoRA 4-bit**: the base model stays quantized to 4-bit and frozen; only small LoRA adapter matrices are trained on top. Avoids loading/training the full model at full precision — this is what makes fine-tuning a <1B model feasible on a single consumer-grade CUDA GPU (`DESIGN.md`'s target hardware).
- Hyperparameters (`--rank`, `--epochs`, `--lr` flags, defaults 32 / 2 / 2e-4 per `DESIGN.md`) and LoRA settings (`target_modules`, `lora_alpha = 2×rank`, `dropout = 0`, `bias = "none"`, `use_gradient_checkpointing = "unsloth"`, `use_rslora = False`) — verified against Unsloth's own LoRA hyperparameters guide, not invented. Still general-purpose recommendations, not numbers calibrated against `llaici`'s specific dataset — expect to adjust after looking at validation loss (§5).
- Chat template applied via `unsloth.chat_templates.get_chat_template(tokenizer, chat_template="qwen3-instruct")` — confirmed present in Unsloth's `chat_templates.py` source (deliberately not `"qwen3-thinking"`, which is for chain-of-thought output this task doesn't want).

⚠️ **Not run/tested**: Unsloth requires CUDA, unavailable in this development environment. Every other script in `scripts/00-05` was validated by actually running it against real data; this one hasn't been, and can't be until it runs on a CUDA-capable machine.

### Apple Silicon alternative: `scripts/06_finetune_mlx.py`

Unsloth's kernels are CUDA/Triton-based — it doesn't run on a Mac (MPS support is still in progress upstream, [unslothai/unsloth#2608](https://github.com/unslothai/unsloth/issues/2608)). "Unsloth Studio" (the desktop app) *does* support Apple Silicon, but via a different backend (MLX), not by running the same `unsloth` Python library — a separate script was needed rather than a flag on `06_finetune.py`.

`scripts/06_finetune_mlx.py` (`make finetune-mlx`) does the same job via [`mlx-lm`](https://github.com/ml-explore/mlx-lm) instead, which runs natively on Apple Silicon:
- Wraps the `mlx_lm.lora` CLI (via subprocess) rather than an internal Python API — `mlx_lm/LORA.md` only documents CLI usage, no training API is documented to call directly.
- Copies `scripts/05`'s output (`train_formatted.jsonl`/`val_formatted.jsonl`) into a `data/samples/mlx/{train,valid}.jsonl` layout — `mlx_lm.lora`'s data loader requires those exact filenames in its `--data` directory.
- LoRA rank/dropout/scale have no CLI flag in `mlx_lm.lora` — only settable via a YAML `-c/--config` file (confirmed in `mlx_lm/lora.py`'s `build_parser()`), so the script always writes one, rather than mixing CLI flags and a config file.
- Base model: `mlx-community/Qwen3-0.6B-4bit` — mlx-lm's own `CONFIG_DEFAULTS["model"]` is literally `"Qwen/Qwen3-0.6b"`, and real Qwen3 LoRA fine-tunes exist in the wild (e.g. a Text-to-SQL MLX LoRA fine-tune of Qwen3.5 at `sciences44/mlx-lora-finetune`) — `LORA.md`'s prose model list not naming Qwen3 explicitly is stale documentation, not a real block. ⚠️ [ml-explore/mlx#2616](https://github.com/ml-explore/mlx/issues/2616): some mlx-lm versions cover fewer trainable parameters than expected for Qwen3 LoRA — worth checking the printed trainable-parameter count if results look off.
- `--mask-prompt` defaults **on** here (loss computed only on the SQL completion, not the question) — mlx-lm exposes this directly, unlike `06_finetune.py`'s `SFTTrainer` call.

⚠️ Same "not run/tested" caveat as `06_finetune.py`, for the opposite reason: no Apple Silicon Mac was available to test against either, only Unsloth/CUDA docs vs. mlx-lm/MLX docs.

**Merge + GGUF export**: see §6 below — `scripts/08_merge_and_quantize_mlx.py` covers this for an MLX-trained adapter.

---

## 4. Run training

**Implemented** as part of `scripts/06_finetune.py` (the same script covers §3 config and §4 training — `SFTTrainer.train()`). `scripts/06_finetune_mlx.py` covers both for the Apple Silicon path the same way (`mlx_lm.lora --train`).

- Expected cost (`DESIGN.md` "💻 Hardware and Costs"): 30min-1h for ~70k pairs on a consumer CUDA GPU, $0 (local GPU, no cloud). Not verified against this project's actual (currently much smaller — 576 train rows) dataset.
- Watch **validation loss**, not just training loss (`eval_strategy="epoch"` in the script, `--steps-per-eval` in the MLX script). This matters more than usual here: since the dataset is template-generated (not organically diverse), a model can drive training loss very low by memorizing the fixed SQL skeletons per template rather than learning to map arbitrary question phrasing to the right structure. Validation loss on held-out examples is the signal that catches this.

---

## 5. Evaluate

**Implemented**: `scripts/07_evaluate.py` (`make evaluate`). Loss curves alone aren't the real test for this project — what actually matters:

- **Syntactic validity**: does the generated SQL parse at all?
- **Executability**: does it run against DuckDB without error, against the actual logical views (`SCHEMA.md`)? This is the project's real correctness bar — SQL that "looks right" but fails at execution is worthless here (mirrors the whole reason `TEMPLATES.md` insists every template be tested against real data, not just written by hand).
- **Correctness, not just executability**: the script goes one step further than "does it run" — it also executes the *gold* SQL for the same question and compares result sets by `id` (the first column every template `SELECT`s). A model could produce syntactically fine SQL against the right view that still answers a subtly different question (wrong category, wrong distance); comparing to the gold result catches that, running the generated SQL alone wouldn't.
- For each validation example, buckets the outcome into `syntax_error` / `empty` / `ok_mismatch` (ran, non-empty, but different rows than gold) / `ok_match` (same `id`s as gold) and reports the percentage in each bucket, plus a per-example JSONL log (`data/samples/eval_results.jsonl`) for manual inspection of failures.
- Markdown-fence stripping (`extract_sql()`) handles the common case of a chat model wrapping its answer in \`\`\`sql ... \`\`\` despite the system prompt asking for "SQL only" — tested against real DuckDB (valid query, syntax error, empty result all handled correctly) even though the model-inference half of the script couldn't be (see below).
- **Generalization check** (not automated by this script): test on question phrasings that don't exactly match any of `03_generate_questions.py`'s hand-written templates (a paraphrase, a reordering, a synonym) — the validation split alone won't catch overfitting to the *exact* phrase templates, since validation examples were built from the same fixed phrase list as training. Would need manually-written held-out questions, not currently done.

⚠️ Same CUDA/Unsloth requirement as §3/§4's `06_finetune.py`, plus a LoRA adapter it produces — not run/tested end-to-end on this dev machine (only the pure-Python helpers, `extract_sql`/`result_ids`, were verified against the real database).

---

## 6. Merge and quantize

**Implemented**: `scripts/08_merge_and_quantize.py` (`make merge-and-quantize`).

- **Merge**: reloads the adapter via `FastLanguageModel.from_pretrained(model_name=<adapter_dir>, ...)` (Unsloth's documented way to reattach a saved adapter to its base model), then a single `model.save_pretrained_gguf(...)` call does the merge internally — no separate merge step needed, verified against Unsloth's docs rather than assumed. Unsloth dequantizes the 4-bit base weights before merging in the adapter, to avoid compounding quantization error.
- **Convert to GGUF and quantize to Q8**: same call, `quantization_method="q8_0"` — confirmed as a real, documented option (alongside `q4_k_m`, `f16`, etc.) and the specific level `DESIGN.md` names ("Quantize to GGUF Q8 (~800 MB)").

⚠️ Same CUDA/Unsloth requirement and untested status as §3/§4/§5's scripts.

### Apple Silicon alternative: `scripts/08_merge_and_quantize_mlx.py`

For an adapter from `scripts/06_finetune_mlx.py` (`make merge-and-quantize-mlx LLAMA_CPP_DIR=~/llama.cpp`). No single mlx-lm call does both merge and GGUF quantization the way Unsloth's `save_pretrained_gguf` does, so this is two steps:

1. **Merge**: `mlx_lm.fuse --model <base> --adapter-path <adapter> --save-path <fused> --dequantize` — reattaches the LoRA adapter and writes a Hugging Face-compatible directory (`config.json` + `.safetensors`). `--dequantize` matters because `06_finetune_mlx.py` trains QLoRA against a 4-bit base — fusing into still-quantized weights would compound quantization error into the merge, same reasoning as Unsloth's own dequantize-before-merge behavior. `mlx_lm.fuse --export-gguf` is deliberately not used: confirmed directly in `mlx_lm/fuse.py`'s source, it raises `ValueError` unless the model's `model_type` is `llama`/`mixtral`/`mistral` — Qwen3 isn't.
2. **Convert + quantize**: llama.cpp's `convert_hf_to_gguf.py` — the thing that actually serves the model (`llama-server`, §7) only ever loads GGUF, never a raw HF directory. Its `--outtype` (confirmed from the script's own argparse) accepts `q8_0` directly, so this one call produces the DESIGN.md target ("Quantize to GGUF Q8 (~800 MB)") without a separate `llama-quantize` binary — that's only needed for quantization levels `convert_hf_to_gguf.py` doesn't support directly (e.g. `q4_k_m`).

Requires a local clone of `ggml-org/llama.cpp` (`--llama-cpp-dir`/`LLAMA_CPP_DIR`) — `convert_hf_to_gguf.py` isn't published as an installable package on its own, unlike `mlx-lm`.

⚠️ Not run/tested, for two compounding reasons: no Apple Silicon Mac available in this environment, and no adapter exists yet from either fine-tuning path (CUDA or MLX) to feed it.

---

## 7. Serve

- Run the quantized model via **llama-server** (llama.cpp's inference server) — takes a natural-language question, returns SQL. Not launched by any script here; a manual CLI command (`llama-server -m <gguf> --port 8080`), documented in `app/README.md`.
- **Implemented (the client side)**: `app/backend` (FastAPI + LangChain) calls llama-server's OpenAI-compatible API, executes the returned SQL against DuckDB, and returns GeoJSON; `app/frontend` (Vue + MapLibre) is the UI that calls the backend and renders results on a map. See `app/README.md` for the full picture.
- This is the boundary into `DESIGN.md` STEP 6 (deployment): packaging the GGUF model together with DuckDB, the GeoParquet data, and the Python pipeline into the Docker image described there — `app/` is that "Python pipeline", not yet containerized.
- ⚠️ Same untested status as §3-§6: `app/backend`'s DuckDB→GeoJSON conversion was verified against real data, but the LLM call itself needs a running llama-server serving a real fine-tuned GGUF — neither exists yet on this dev machine.

---

## Open questions (not yet decided)

- **Hyperparameters** (rank, epochs, LR) are `DESIGN.md` defaults, not tuned against this project's actual dataset size/composition (currently far below the 10k-70k target — see chat history on `ROWS` sizing).
- **System message wording** (`scripts/05_format_for_training.py --system`) — a first reasonable default, not tested against actual model behavior.
- **`scripts/06_finetune.py`, `scripts/07_evaluate.py`, and `scripts/08_merge_and_quantize.py` are unverified end-to-end** — written against Unsloth's documented API/model catalog, but never executed (no CUDA GPU on this dev machine). Running them for the first time on real GPU hardware may surface API mismatches (Unsloth/TRL/PEFT version differences, e.g. `SFTConfig` field availability) the way `TEMPLATES.md`'s SQL templates surfaced real bugs the first time *they* ran against real data. `07_evaluate.py`'s pure-Python parts (SQL cleanup, DuckDB execution/comparison) were tested in isolation; the model-loading/generation half wasn't.
- **Generalization check with hand-written held-out questions** — not automated (§5).

## Next step, once a first model exists

**Compound/composite queries** — a question combining two spatial relations at once (e.g. "restaurants north of Lisbon *and* along the coast") isn't supported: every template in `TEMPLATES.md` encodes exactly one relation, and the model has no compositional training signal for combining two. Deliberately accepted as a v1 limitation (Option A) rather than building a combinatorial family of paired-relation templates (Option B) or an orchestrator that splits and merges two single-relation calls (Option C) upfront — see `TEMPLATES.md` "Known limitation — no compound/composite queries" for the full reasoning. Revisit **after** a first model has been trained and evaluated on the current single-relation templates (§4/§5 above) — whether compound queries are worth the extra work (B's template combinatorics, or C's orchestration complexity) should be judged against real usage, not decided before there's even a working v1.
