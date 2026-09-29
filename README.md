# LLaIci

Ask a geographic question in plain language ("restaurants in Madrid", "cities bordering Morocco") and get a map back. Under the hood: a small LLM fine-tuned to translate the question into spatial SQL, run against Overture Maps data in DuckDB.

## Documentation

| Document | What's in it |
|---|---|
| [`docs/DESIGN.md`](docs/DESIGN.md) | The full project design and rationale |
| [`docs/SCHEMA.md`](docs/SCHEMA.md) | The DuckDB logical data model (the views the model queries) |
| [`docs/TEMPLATES.md`](docs/TEMPLATES.md) | The SQL query patterns the model learns to generate |
| [`docs/FINETUNING.md`](docs/FINETUNING.md) | The fine-tuning steps in detail — model choice, LoRA config, evaluation |
| [`docs/PIPELINE.md`](docs/PIPELINE.md) | The exact command, flags, and defaults for every pipeline step below |
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

Steps 00-05 (data preparation) run fine on a regular machine and have all been tested against real data. Steps 06-08 (fine-tuning) need an NVIDIA GPU with CUDA and haven't been run yet in this environment — see [`docs/FINETUNING.md`](docs/FINETUNING.md) for details.

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

make finetune-venv                            # once: sets up the fine-tuning environment (CUDA path only, see below)
make finetune-pipeline-cuda                   # steps 06-08: fine-tune, evaluate, merge/quantize — NVIDIA GPU (CUDA)
# or, on a Mac:
make finetune-pipeline-mlx LLAMA_CPP_DIR=~/llama.cpp   # steps 06+08 — Apple Silicon (mlx-lm), see FINETUNING.md
```

`THREADS` defaults to **2** (kept low on purpose, to stay light on whatever machine this runs on) — raise it to match your CPU's actual core count (e.g. `THREADS=10` on a 10-core machine) for a much faster run; see `docs/PIPELINE.md` for details.

`make finetune-venv` creates a dedicated Python environment in `.venv-finetune/` (with [uv](https://docs.astral.sh/uv/)) holding Unsloth and a CUDA build of PyTorch, then checks that the GPU is actually usable from it. The CUDA targets (`finetune`, `evaluate`, `merge-and-quantize`) run inside that environment and build it automatically if it's missing, so this step is optional — running it first just surfaces install problems early. The PyTorch CUDA version defaults to `cu130`; override it with `TORCH_BACKEND` (e.g. `make finetune-venv TORCH_BACKEND=cu128`) if your driver doesn't support CUDA 13.0. To rebuild from scratch: `rm -rf .venv-finetune`.

Either way, you end up with a quantized GGUF model under `models/`, ready to serve with `llama-server` (see [`app/README.md`](app/README.md)).

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
