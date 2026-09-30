# LLaIci

A toolkit to train a small LLM (eg. Qwen3 0.6B) to turn a geographic question in plain language, French or English ("restaurants in Madrid", "villes frontalières du Maroc"), into a spatial DuckDB SQL query over [Overture Maps](https://overturemaps.org/) data. The model targets a custom schema: logical views built on top of the Overture extracts (see [`docs/SCHEMA.md`](docs/SCHEMA.md)), not the raw Overture tables.

It covers the whole chain:
- **Dataset generation**: real Overture entities fill hand-written SQL templates, each query is validated by execution, and FR/EN questions are generated for it.
- **Fine-tuning with LoRA (QLoRA)**: the base model stays frozen and 4-bit quantized, and only a small adapter is trained on top, so it fits on a consumer GPU.
- **Export and demo**: the result is exported to GGUF and served in a demo app that asks the question and shows the result on a map.

> [!NOTE]
> The model only knows the view names and columns, not what's behind them. Training runs on the local Parquet extracts, but once trained, the same model could query another backend, such as a PostgreSQL/PostGIS database attached through DuckDB's `postgres` extension, as long as the views keep exposing the same columns. This hasn't been tested. Expect to convert PostGIS geometries in the views, and note that spatial filters (`ST_Within`, `ST_DWithin_Spheroid`...) aren't pushed down to PostgreSQL: DuckDB pulls the rows and computes them itself, which can be slow on large tables.

## How it works, in plain terms

**1. Build a dataset of (question, SQL) pairs.** Nobody writes them by hand: real places are sampled from Overture ("Lyon", "the Seine", "bus stops"), plugged into hand-written SQL templates ("bus stops in {place}"), and each query is **executed** to keep only the ones that return something. Then several French and English phrasings of the question are generated for each query. This is steps 01-05.

**2. Teach an existing model with a LoRA adapter.** We don't train a model from scratch. We start from a small open model (Qwen3 0.6B, downloaded automatically from Hugging Face the first time) that already understands language, and teach it our SQL. Its ~600M parameters stay untouched (**frozen**). Next to them, LoRA adds small extra matrices, the **adapter** (~20M parameters, about 3% of the model, ~80 MB), and only those are trained. The base model is also loaded compressed to 4-bit (**QLoRA**), which is what lets training fit on a consumer GPU. Think of the base model as a printed map and the adapter as a transparent overlay with your notes: the overlay is useless alone, and the map alone doesn't know your notes. This is step 06.

**3. Check that the generated SQL is actually right.** The training loss only says the model is learning something. The real test (step 07) asks the model questions it never saw, runs its SQL in DuckDB, and compares the rows returned to the expected answer: correct, wrong rows, empty, or broken SQL.

**4. Package it as a single file.** An adapter only works on top of its base model. Step 08 **merges** the two and **quantizes** the result into one GGUF file (~800 MB), which `llama-server` can serve on its own to the demo app.

## Documentation

| Document | What's in it |
|---|---|
| [`docs/DESIGN.md`](docs/DESIGN.md) | The full project design and rationale |
| [`docs/SCHEMA.md`](docs/SCHEMA.md) | The DuckDB logical data model (the views the model queries) |
| [`docs/TEMPLATES.md`](docs/TEMPLATES.md) | The SQL query patterns the model learns to generate |
| [`docs/FINETUNING.md`](docs/FINETUNING.md) | The fine-tuning steps in detail — model choice, LoRA config, evaluation |
| [`docs/PIPELINE.md`](docs/PIPELINE.md) | The exact command, flags, and defaults for every pipeline step below |
| [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) | Problems hit while running the pipeline (warnings, low scores, locks...) and what they mean |
| [`app/README.md`](app/README.md) | How to run the demo app (backend + frontend + llama-server) |
| [`app/backend/README.md`](app/backend/README.md) | Backend-specific setup and API reference |

## The pipeline, step by step

Each step is a script under `scripts/`, numbered in the order it runs.

| Step | What it does |
|---|---|
| _Setup_ | Download the Overture Maps data (`make download-overture`) |
| **00** | Build the DuckDB logical views |
| **01** | Sample real entities to fill the SQL templates |
| **02** | Fill in and validate those templates by executing them |
| **03** | Generate natural-language questions (French + English) for each validated query |
| **04** | Split the dataset into train / validation sets |
| **05** | Format the data for fine-tuning |
| **06** | Fine-tune the model (QLoRA) |
| **07** | Evaluate the fine-tuned model |
| **08** | Merge and quantize the model to GGUF |

Steps 00-05 (data preparation) run fine on a regular machine and have all been tested against real data. Steps 06-08 (fine-tuning) need an NVIDIA GPU with CUDA. Steps 06-08 have been tested end to end on an NVIDIA RTX 50xx (Blackwell) GPU with a small test dataset (the pipeline works; the model's quality at full dataset size hasn't been measured yet) — see [`docs/FINETUNING.md`](docs/FINETUNING.md) for details.

👉 **For the exact command for each step, see [`docs/PIPELINE.md`](docs/PIPELINE.md).**

### Run it end to end

Once the Overture data is downloaded and the DuckDB views are built:

```bash
make download-overture          # once: download the Overture Maps extracts
docker compose up duckdb        # once: builds the DuckDB views (00_init.sql) on first start
```

> [!IMPORTANT]
> Stop the `duckdb` container (`docker compose stop duckdb`) before running the pipeline. It keeps `data/db/llaici.duckdb` open in read-write mode, which locks out every other process — even read-only ones. The scripts then fail with `Could not set lock on file ... Conflicting lock is held in duckdb` (see [DuckDB concurrency](https://duckdb.org/docs/stable/connect/concurrency)).

...the rest of the pipeline (steps 01-08) is just two commands:

```bash
make generate-dataset ROWS=70000 THREADS=10   # steps 01-05: sample, validate, generate questions, split, format

make finetune-venv                            # optional: the CUDA targets below build this environment themselves if missing (see below)
make finetune-pipeline-cuda LLAMA_CPP_DIR=~/llama.cpp   # steps 06-08: fine-tune, evaluate, merge/quantize — NVIDIA GPU (CUDA)
# or, on a Mac:
make finetune-pipeline-mlx LLAMA_CPP_DIR=~/llama.cpp   # steps 06+08 — Apple Silicon (mlx-lm), see FINETUNING.md
```

Both paths export the final GGUF with [llama.cpp](https://github.com/ggml-org/llama.cpp)'s converter, so they need a local clone of it: `git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp`, then pass its path with `LLAMA_CPP_DIR`.

`THREADS` defaults to **2** (kept low on purpose, to stay light on whatever machine this runs on) — raise it to match your CPU's actual core count (e.g. `THREADS=10` on a 10-core machine) for a much faster run; see `docs/PIPELINE.md` for details.

`make finetune-venv` creates a dedicated Python environment in `.venv-finetune/` (with [uv](https://docs.astral.sh/uv/)) holding Unsloth and a CUDA build of PyTorch, then checks that the GPU is actually usable from it. The CUDA targets (`finetune`, `evaluate`, `merge-and-quantize`) run inside that environment and build it automatically if it's missing, so this step is optional — running it first just surfaces install problems early. The PyTorch CUDA version defaults to `cu130`; override it with `TORCH_BACKEND` (e.g. `make finetune-venv TORCH_BACKEND=cu128`) if your driver doesn't support CUDA 13.0. To rebuild from scratch: `rm -rf .venv-finetune`.

Either way, you end up with a quantized GGUF model under `models/`, ready to serve with `llama-server` (see [`app/README.md`](app/README.md)).

### What `make finetune` (step 06) actually does

It teaches a small existing LLM to answer a question with our SQL. The model isn't trained from scratch: the base model stays as-is, and only a small add-on (a **LoRA adapter**) is trained on top of it.

1. **Loads the base model**: the model to fine-tune, here Qwen3 0.6B Instruct, already compressed to 4-bit (Hugging Face checkpoint `unsloth/Qwen3-0.6B-unsloth-bnb-4bit`). Training LoRA on a 4-bit model is what **QLoRA** means, and it's what makes training fit on a consumer GPU.
   **Choosing another model**: without `MODEL`, `make finetune` lists the models already on the machine and lets you pick one (no download), or offers the default one if none is there. Or name it directly: `make finetune MODEL=<hugging-face-id>` (see [`docs/PIPELINE.md`](docs/PIPELINE.md) step 06, including `CHAT_TEMPLATE` for other model families).
   **No manual download needed**: the first `make finetune` downloads the model from Hugging Face by itself (a few hundred MB). It's public, so no account or token is required, but that first run needs internet access. The model is then cached in `~/.cache/huggingface/` and reused by later runs, including `make evaluate`. On a Mac, `make finetune-mlx` does the same with `mlx-community/Qwen3-0.6B-4bit`. To download it ahead of time (e.g. before going offline): `.venv-finetune/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('unsloth/Qwen3-0.6B-unsloth-bnb-4bit')"`.
2. **Reads the training files** from step 05: `data/training/train_formatted.jsonl` and `val_formatted.jsonl`. Each example is a short conversation (system instruction → question → SQL), rendered with Qwen's own chat template (`qwen3-instruct`, no `<think>` reasoning).
3. **Trains the LoRA adapter** (Unsloth + TRL `SFTTrainer`): the model sees every question and learns to produce the matching SQL. Defaults: rank 32, 2 epochs, learning rate 2e-4, batch 2 × 4 gradient-accumulation steps.
4. **Evaluates on the validation set at the end of each epoch** and logs the loss every 10 steps. Watch the **validation loss** (`eval_loss`): the dataset is template-generated, so training loss can drop just by memorizing SQL skeletons. Only the validation loss tells whether the model generalizes.
5. **Saves the adapter** to `models/llaici-<model>-lora/` (`llaici-qwen3-0.6b-lora/` by default; each model gets its own folder) (a few tens of MB, plus intermediate `checkpoint-*` folders). This isn't a usable model on its own yet: step 07 (`make evaluate`) checks its SQL against DuckDB, and step 08 (`make merge-and-quantize`) merges it into the base model and exports the GGUF for `llama-server`.

Good to know:
- **Needs an NVIDIA GPU with CUDA** (runs in `.venv-finetune/`, built automatically). On a Mac, use `make finetune-mlx` instead: same job via mlx-lm.
- **Dataset size matters**: a test run like `ROWS=50` (~100 examples) only checks that the chain works; for a model worth using, generate the dataset with `ROWS=70000` (~30 min to 1 h of training, per `DESIGN.md`'s estimate).
- **Hyperparameters**: pass them to make, e.g. `make finetune EPOCHS=3 RANK=16 LR=1e-4`. Any you leave out keep their default. Full list (`EPOCHS`, `RANK`, `LORA_ALPHA`, `LORA_DROPOUT`, `LR`, `BATCH_SIZE`, `GRAD_ACCUM`, `MAX_SEQ_LENGTH`, `SEED`, `MODEL`, `CHAT_TEMPLATE`, plus `ITERS`/`LORA_SCALE` for `finetune-mlx`) in [`docs/PIPELINE.md`](docs/PIPELINE.md) step 06. The same variables also work with `make finetune-pipeline-cuda` / `finetune-pipeline-mlx`.
- Details and rationale: [`docs/FINETUNING.md`](docs/FINETUNING.md) §3-4.

## Run the DuckDB UI

Opens an interactive DuckDB session in the browser, with the logical views ready to query.

```bash
make up            # start
make build-up      # rebuild the image first, then start
make down          # stop
```

While the UI is running it holds a write lock on the database: stop it before running any pipeline script (see the note above).

## Run the demo app

A small web app to actually ask the fine-tuned model a question and see the result on a map (FastAPI + LangChain backend, Vue + MapLibre frontend).

```bash
make app-up
```

See [`app/README.md`](app/README.md) for the full picture, including how to run `llama-server` alongside it.

> [!IMPORTANT]
> The map needs a browser with **WebGL2** (MapLibre GL 6 requires it). Without it the page stays blank and the browser console shows `GPUInitializationError: WebGL2 is required to display this map`. This usually means hardware acceleration is off or the GPU driver is blocklisted (common on Linux): enable hardware acceleration in the browser settings, then check at https://get.webgl.org/webgl2/. Details in [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md).
