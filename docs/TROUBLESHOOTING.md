# Troubleshooting & known pitfalls

Problems actually hit while running the pipeline, what they meant, and what was done about them. For the exact commands, see [`PIPELINE.md`](PIPELINE.md); for the fine-tuning rationale, [`FINETUNING.md`](FINETUNING.md).

---

## Data preparation (steps 01-05)

### Scripts fail with `Could not set lock on file ... Conflicting lock is held`

The `duckdb` container (or the DuckDB UI, `make up`) keeps `data/db/llaici.duckdb` open read-write, which locks out every other process, read-only ones included. Stop it first: `docker compose stop duckdb` / `make down`. This applies to steps 01, 02 and 07 (evaluate), which all query the database.

### FR questions read "où se trouve Espagne", "frontalières de Maroc"

Since step 02 swaps local names for exonyms ("España" → "Espagne"), the missing French article became visible. Fixed in step 03: countries get their article from a hand-written ISO code → gender table (`FR_COUNTRY_GENDER` in `03_generate_questions.py`), with contractions and elision ("du Maroc", "de l'Espagne", "des Pays-Bas", "d'Orléans", "au Havre"). The SQL keeps the bare name, so the model learns to drop the article: without these examples, a user typing "du Maroc" risked the model copying it into `ILIKE '%du Maroc%'` — valid SQL, silently empty result.

- Needs `place_country_code` from step 01: entities sampled before this fix have none, so re-run step 01 (at least `bordering` and `show_division`).
- A few genders depend on the exact FR name Overture uses (CD, CG, MM): check them if those countries show up.
- Rivers still get no article ("le long de Seine"), and EN has the same issue in smaller form ("show me Netherlands").

### Questions typed without accents fail ("ponts pres de Madrid")

The dataset only had correctly accented questions, so for the model "pres" was an unknown word, and the SQL's `ILIKE '%Seville%'` wouldn't have matched "Séville" anyway. Fixed: step 02's name match is accent-insensitive (`strip_accents` on both sides), and step 03 adds accent-less copies of a share of the questions (`NO_ACCENT_RATIO`, default 0.3). Needs steps 02-05 re-run (not 01) and a new training: existing SQL doesn't have `strip_accents`.

### `ROWS` is not the number of training pairs

`ROWS` is the number of sampled *entities*, split evenly across templates. Each entity yields ~3-4 (question, SQL) pairs (FR + EN, several phrasings), minus what steps 02/03 filter out. Observed: `ROWS=50` → 45 entities → 167 pairs. So `ROWS=70000` gives roughly 230k pairs, not 70k — about 7x longer training than `DESIGN.md`'s "30 min - 1 h for ~70k pairs" estimate (which rather matches `ROWS≈20000`). Use `MAX_PER_TEMPLATE` to cap the volume, and try an intermediate size (e.g. `ROWS=5000`) to measure real timings first.

---

## Fine-tuning (step 06)

### "Do I need to download the model / run `make finetune-venv` first?"

No to both. `make finetune` builds `.venv-finetune/` itself if missing, and the first `FastLanguageModel.from_pretrained(...)` call downloads the base model from Hugging Face (public, no token) into `~/.cache/huggingface/`, reused afterwards. Without `MODEL=`, `make finetune` lists the models already cached and lets you pick one (`scripts/model_picker.py`).

### The model picker lists two Qwen3 0.6B

```
1) unsloth/Qwen3-0.6B-unsloth-bnb-4bit  592.1M  (default)
2) unsloth/qwen3-0.6b  1.2G
```

Not two models: the same Qwen3 0.6B in two formats. The 4-bit one is the training base (QLoRA), downloaded by the first `make finetune`. The 16-bit one is what step 08 merges the adapter into (merging into already-compressed weights would add rounding error on top), downloaded by the first `make merge-and-quantize`, which derives its name from the 4-bit one by dropping the `-unsloth-bnb-4bit` suffix. The picker lists every LLM in the Hugging Face cache, so it shows both.

Train on **1** (the default). Choosing 2 would likely work too (Unsloth would compress it to 4-bit on load) but loads slower for no gain. Keep both cached: deleting the 16-bit one only makes step 08 download it again (1.2 GB).

### Evaluate / merge can't find the adapter, or use the wrong base model

Output directories are named after the base model (`models/llaici-<model>-lora/`). If you trained with a non-default `MODEL` (on the command line or picked interactively), pass the same `MODEL=` to `make evaluate` and `make merge-and-quantize`: they default to Qwen3 0.6B otherwise. Changing model family (not Qwen3) also needs a matching `CHAT_TEMPLATE`.

### Warnings during training

| Warning | Meaning | Action |
|---|---|---|
| `Unsloth should be imported before [trl, transformers, peft]` | Our script imported `trl` first, so Unsloth's patches may not apply (slower / more memory). | Fixed: `06_finetune.py` now imports `unsloth` first. |
| `torch/utils/_pytree.py ... register_constant() on Enum subclasses is deprecated` | PyTorch deprecation, triggered inside a library Unsloth uses. | None, harmless. |
| `torch._dynamo.config.inline_inbuilt_nn_modules is deprecated` | Unsloth sets a PyTorch option that no longer does anything. | None, harmless. |
| `'has_cudnn' / 'has_mps' / 'has_mkldnn' is deprecated` | Unsloth reads PyTorch attributes marked obsolete; they still return the right value. | None, harmless — for Unsloth to fix upstream. |
| Unsloth banner shows `CUDA: 12.0` | That's the GPU's *compute capability* (sm_120 = Blackwell), not the CUDA version (shown as `CUDA Toolkit`). | None. |
| `FA2 = False` | Flash Attention 2 not installed; Unsloth uses xformers instead. | None needed for a 0.6B model. |

### Training finishes in seconds

Expected on a test dataset: `ROWS=50` → ~111 training examples → 28 steps (111 / (batch 2 × 4 accumulation) × 2 epochs), ~15 s on a recent consumer GPU. See "`ROWS` is not the number of training pairs" above for the real-size estimate.

### Low `eval_loss` doesn't mean correct SQL

`06_finetune.py` computes the loss over the whole example — system prompt, question and SQL — so part of the drop is the model memorizing the fixed system prompt. A falling `eval_loss` (no rise vs training loss) rules out overfitting, but only `make evaluate` tells whether the SQL is right.

---

## Evaluation (step 07)

### Evaluation slows down or seems stuck on one question

Some questions are much heavier for DuckDB than others: "left/right bank", "along" and "within X km of" compute geometry over the `water` table. When several fall in a row, progress slows (GPU at 0%, DuckDB busy on the CPU). A badly filtered generated query (e.g. a spatial join over a whole table) could also run for a very long time.

Every query now runs in a child process that's killed past a time limit (`con.interrupt()` isn't reliable on some spatial queries, same finding as step 01):
- the **gold** query runs first and is timed; past `--gold-timeout` (300 s) the question is **skipped**, left out of the score (not the model's fault);
- the **generated** query gets **5x the gold query's time**, at least `--query-timeout` (30 s); past it, the answer counts as **"too slow (timed out)"**.

A fixed limit wouldn't fit: heavy templates can legitimately take a minute with 2 threads, light ones a fraction of a second. Each line of `eval_results.jsonl` records `gold_seconds` and `gen_seconds`, and the file is flushed per question, so `wc -l models/llaici-<model>-lora/eval_results.jsonl` shows live progress. `THREADS=10` speeds DuckDB up a lot on the heavy templates.

### `make evaluate` takes hours

Each validation question is a full generation (up to ~650 tokens of SQL), run one at a time. At `ROWS=5000` there are ~2,000 validation questions, so a full evaluation can take 2-3 hours, longer than training itself. Use a sample while iterating:

```bash
make evaluate MAX_EVAL=300
```

It evaluates a random sample of 300 questions, drawn with a fixed seed (`--seed`, default 42): two models evaluated with the same `MAX_EVAL` see the same questions, so their scores are directly comparable. The verdict box says when a sample was used ("300 validation questions (random sample of 2000)"). With 300 questions the % correct is accurate to roughly ±5 points: enough to compare two trainings, not to split hairs. Keep the full evaluation for the final model.

### Many `syntax_error`, some with `Parser Error: syntax error at end of input`

Generation was capped at `--max-new-tokens 256`, but the templates' SQL is long (7-language `ILIKE` chains): median ~200 tokens, up to ~530. On a 56-example run, 15 gold queries exceeded 256 tokens and 10 generated queries were cut mid-statement, scored as syntax errors. Fixed: default raised to 1024.

### `Both max_new_tokens (=...) and max_length (=40960) seem to have been set` — and evaluation was random

Qwen3's `generation_config.json` is tuned for chat: `max_length: 40960`, which clashes with `--max-new-tokens` (a warning on every example), and **sampling** (`do_sample: true`, temperature 0.6, top_p/top_k). So evaluation drew its SQL at random: two runs gave different scores, and some errors were bad luck rather than the model. Fixed in `07_evaluate.py`: after loading, the generation config is switched to greedy decoding (`do_sample=False`, sampling params cleared) and `max_length` is unset. Runs are now reproducible and the warning is gone. Scores measured before this fix aren't comparable with later ones.

### `The attention mask is not set and cannot be inferred from input because pad token is same as eos token`

`07_evaluate.py` passed only the token ids to `model.generate()`. Qwen uses the same token for padding and end-of-sequence, so `generate()` can't rebuild the attention mask (which tokens to attend to) on its own. With one unpadded question at a time the mask is all ones anyway, so results weren't affected. Fixed: the chat template is now applied with `return_dict=True`, which returns the `attention_mask` alongside the ids, and both are passed to `generate()`.

### Low score after a small test run

A model trained on ~111 examples (~7 per template) scored ~7% `ok_match`. The per-example detail (`models/llaici-qwen3-0.6b-lora/eval_results.jsonl`) shows it learned the SQL *shape* but not the schema yet:

- invented columns (`p.categories` instead of `p.category_hierarchy`, `subtype` on `places`);
- invented `subtype`/`class` values (`'local_entrance'` / `'green_hut_parking'` instead of `'transit'` / `'parking'`);
- wrong template (lodging sent to `infrastructures` instead of `places`, "periphery" answered as "proximity");
- place name altered (`'%Creixomil%'` for "Creixomil e Mariz", `'%El Vilar%'` for "el Vilar").

Expected at this size — not a sign the approach fails. Scale the dataset up before judging quality.

### Reading the verdict, and what to do when it's not good

`make evaluate` ends with a colored verdict on the share of **correct** answers (same rows as the expected query): green ≥ 80%, yellow 50-80%, red < 50%. These thresholds are a judgment call, not a standard. Below green, it prints advice aimed at the most frequent failure. In order of what to try:

1. **More training data, first.** By far the most common cause. Each template needs to be seen many times with many values to learn its exact columns and `subtype`/`class` values. Under ~5,000 training examples, regenerate a bigger dataset (`make generate-dataset ROWS=5000 THREADS=10` or more, see "`ROWS` is not the number of training pairs" above) before touching anything else.
2. **Look at the failures** in `models/llaici-<model>-lora/eval_results.jsonl` (each line: question, expected SQL, generated SQL, outcome). The dominant failure says what's missing:
   - **SQL error**: invented columns or tables → the model hasn't learned the schema yet → more data, or more epochs (`make finetune EPOCHS=3`). If the verdict says answers were **cut off**, raise `--max-new-tokens` instead: the SQL was fine but incomplete.
   - **Empty result**: usually the place name was altered in the `ILIKE` (truncated, re-capitalized) or a category was invented → more data; if it's always the same phrasing, the question templates in step 03 may need more variety.
   - **Wrong rows**: the right shape but the wrong template (proximity instead of periphery) or the wrong category → more data, more epochs.
3. **Watch the loss curves from step 06.** If `eval_loss` rises while training loss keeps falling, the model is overfitting: fewer epochs, not more. If both are still falling at the end, more epochs can help.
4. **Then hyperparameters**: a higher LoRA rank (`RANK=64`) gives the adapter more capacity; a bigger base model (`MODEL=unsloth/Qwen3-1.7B-unsloth-bnb-4bit`) understands questions better but is slower and bigger to serve.

Change one thing at a time and re-run `make finetune` + `make evaluate`, so you know what helped. Use the same `MAX_EVAL` for every run (see "`make evaluate` takes hours" above): same sample, comparable scores.

### Known gaps in `07_evaluate.py`

- **No `<think>…</think>` stripping**: not needed so far (the `qwen3-instruct` template produced none), but would turn correct SQL into `syntax_error` if a model emitted reasoning tags.

---

## Merge and export (step 08)

### `Permission denied: 'models/llaici-<model>-gguf/model.safetensors'` (first version of step 08)

The first version of `08_merge_and_quantize.py` used Unsloth's `save_pretrained_gguf`. To merge, Unsloth downloads the **16-bit** version of the base model (e.g. `unsloth/qwen3-0.6b`, ~1.2 GB, on top of the 4-bit one used for training), copies its `model.safetensors` into the output directory, then writes the adapter into that copy in place. Files in the Hugging Face cache can be read-only (mode `444`), and the copy keeps that mode, so the in-place write fails.

Rather than changing file permissions, step 08 was rewritten: the merge is now done with PEFT, which writes fresh files (`models/llaici-<model>-merged/`), and the GGUF conversion with llama.cpp's `convert_hf_to_gguf.py` (see `FINETUNING.md` §6). A failed run of the old version may have left a read-only `models/llaici-<model>-gguf/model.safetensors` behind (plus its `config.json`, tokenizer files...): they're not used anymore and can be deleted.

The `Cache check failed: tokenizer.model not found in local cache` line of that old version was harmless: Qwen3 has no `tokenizer.model` file (it uses `tokenizer.json`).

### `convert_hf_to_gguf.py` fails with `ModuleNotFoundError`

The converter runs with the fine-tuning venv's Python, which has `torch`, `transformers` and `numpy` but maybe not everything llama.cpp's converter imports. Install its requirements into the venv: `uv pip install --python .venv-finetune/bin/python -r ~/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt`. If that tries to change the `torch` version, install only the missing module the error named instead.

---

## Demo app

### The map stays blank: `GPUInitializationError: WebGL2 is required to display this map`

MapLibre GL 6 draws the map with **WebGL2**, and the browser doesn't provide it: hardware acceleration is turned off, or the browser has blocklisted the GPU driver (common on Linux, especially with NVIDIA drivers). Nothing to do with the project or the model; the page is blank from the first load, before any question.

1. **Check**: `chrome://gpu` (Chrome/Chromium, "WebGL2" line) or `about:support` (Firefox, "Graphics" section).
2. **Enable**:
   - Chrome/Chromium: Settings → System → "Use graphics acceleration when available", restart. If still disabled: `chrome://flags` → "Override software rendering list" (`#ignore-gpu-blocklist`), restart.
   - Firefox: `about:config` → `webgl.disabled` = `false`; if still disabled, `webgl.force-enabled` = `true`, restart.
3. **Verify** at https://get.webgl.org/webgl2/ (a spinning cube), then hard-reload the app (`Ctrl+Shift+R`).

Still failing: try another browser (they don't share the same driver blocklist) or update the GPU driver.

### "The model's answer was cut off after 1024 tokens without finishing"

The model never ended its answer: under greedy decoding (the app asks for `temperature=0`), an under-trained model often falls into a repetition loop (e.g. the same `list_contains(...)` condition over and over) until it hits the token limit. The backend reports that instead of running the cut-off SQL, which would only fail with a confusing DuckDB syntax error. Checked on the small test model: the same loop happens with the original base model + adapter (no GGUF conversion) and with the exact training prompt format, so it's the model, not the integration. Fix: train on more data.

### A question that "worked" during evaluation fails in the app

Evaluation results produced before `07_evaluate.py` switched to greedy decoding came from *random sampling*: a success there could be luck. The app always decodes greedily, like the current evaluation. Re-run `make evaluate` (or `make evaluate MAX_EVAL=300` for a quicker score) for a result that matches what the app will do.

