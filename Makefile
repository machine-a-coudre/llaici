BBOX ?= -9.72,35.91,3.59,43.82 # or -9.84,35.55,3.43,44.43
PARQUET_DIR := data/parquet
SAMPLES_DIR := data/samples
# Final fine-tuning inputs only (scripts/05 output), kept apart from the intermediate files above
TRAINING_DIR := data/training
ROWS ?= 100
THREADS ?= 2
TEMPLATE ?= all

# Nice, hard-to-miss reminder shown before every DuckDB-backed script launch (see
# sample-entities/validate-samples/evaluate below): the default is intentionally
# low, and staying on it will make a large run slow.
define PRINT_THREADS
@printf "\n\033[1;36m⚙  DuckDB threads: %s\033[0m\n\033[33m   → default is 2, kept low on purpose. Raise it with THREADS=N to match your\033[0m\n\033[33m     CPU's core count (e.g. THREADS=10) — otherwise this run will be slow.\033[0m\n\n" "$(THREADS)"
endef

# Dedicated venv for the CUDA/Unsloth fine-tuning path (see `finetune-venv`).
FINETUNE_VENV := .venv-finetune
FINETUNE_PY := $(FINETUNE_VENV)/bin/python
# PyTorch CUDA build. Must be >= cu128 for Blackwell GPUs (RTX 50xx, sm_120) and <=
# the max CUDA version the driver supports. cu132+ has no recent xformers wheels,
# which Unsloth needs, so cu130 is the newest usable one (-> torch 2.12).
TORCH_BACKEND ?= cu130

.PHONY: download-overture download-overture-places up build-up down sample-entities merge-samples validate-samples generate-questions split-dataset format-for-training generate-dataset finetune-venv finetune finetune-mlx evaluate merge-and-quantize merge-and-quantize-mlx finetune-pipeline-cuda finetune-pipeline-mlx app-up app-build-up app-down

download-overture:
	mkdir -p $(PARQUET_DIR)
	uvx overturemaps download --connect_timeout 60 --request_timeout 300 --no-stac --bbox=$(BBOX) -f geoparquet --type=division -o $(PARQUET_DIR)/eu_divisions.parquet
	uvx overturemaps download --connect_timeout 60 --request_timeout 300 --no-stac --bbox=$(BBOX) -f geoparquet --type=division_area -o $(PARQUET_DIR)/eu_division_areas.parquet
	uvx overturemaps download --connect_timeout 60 --request_timeout 300 --no-stac --bbox=$(BBOX) -f geoparquet --type=infrastructure -o $(PARQUET_DIR)/eu_infrastructures.parquet
	uvx overturemaps download --connect_timeout 60 --request_timeout 300 --no-stac --bbox=$(BBOX) -f geoparquet --type=water -o $(PARQUET_DIR)/eu_water.parquet
	uvx overturemaps download --connect_timeout 60 --request_timeout 300 --no-stac --bbox=$(BBOX) -f geoparquet --type=place -o $(PARQUET_DIR)/eu_places.parquet

# STEP 4 sampling (see DESIGN.md / TEMPLATES.md). Each template samples into its own
# entities_<name>.jsonl (see scripts/01_sample_entities.py); with the default
# TEMPLATE=all, entities.jsonl is then rebuilt by merging all of them.
# Examples:
#   make sample-entities ROWS=70000 THREADS=10       # regenerate + merge everything
#   make sample-entities TEMPLATE=along ROWS=500     # regenerate only entities_along.jsonl
#                                                     # (entities.jsonl is left untouched — see `merge-samples`)
sample-entities:
	mkdir -p $(SAMPLES_DIR)
	$(PRINT_THREADS)
	uvx --with duckdb python3 scripts/01_sample_entities.py --rows $(ROWS) --threads $(THREADS) --template $(TEMPLATE) --out $(SAMPLES_DIR)/entities.jsonl --yes

# STEP 4: rebuild entities.jsonl by merging every entities_<name>.jsonl as-is, with
# no resampling — e.g. after `make sample-entities TEMPLATE=along` above.
merge-samples:
	uvx --with duckdb python3 scripts/01_sample_entities.py --template merge --out $(SAMPLES_DIR)/entities.jsonl

# STEP 4: fill templates with sampled entities and validate by execution (see scripts/02_fill_and_validate.py)
validate-samples:
	$(PRINT_THREADS)
	uvx --with duckdb python3 scripts/02_fill_and_validate.py --in $(SAMPLES_DIR)/entities.jsonl --out $(SAMPLES_DIR)/validated_samples.jsonl --threads $(THREADS) $(if $(LOCAL_NAME_RATIO),--local-name-ratio $(LOCAL_NAME_RATIO))

# STEP 4: generate NL question formulations (FR+EN) for validated pairs -> dataset.jsonl (see scripts/03_generate_questions.py)
generate-questions:
	python3 scripts/03_generate_questions.py --in $(SAMPLES_DIR)/validated_samples.jsonl --out $(SAMPLES_DIR)/dataset.jsonl $(if $(MAX_PER_TEMPLATE),--max-per-template $(MAX_PER_TEMPLATE))

# STEP 5 (fine-tuning) prep, step 1c: split dataset.jsonl into train/val, stratified by template (see FINETUNING.md, scripts/04_split_train_val.py)
split-dataset:
	python3 scripts/04_split_train_val.py --dataset $(SAMPLES_DIR)/dataset.jsonl --validated $(SAMPLES_DIR)/validated_samples.jsonl --train-out $(SAMPLES_DIR)/train.jsonl --val-out $(SAMPLES_DIR)/val.jsonl $(if $(VAL_RATIO),--val-ratio $(VAL_RATIO))

# STEP 5 (fine-tuning) prep, steps 1a/1b: format train/val pairs as Qwen chat messages (see FINETUNING.md, scripts/05_format_for_training.py)
format-for-training:
	python3 scripts/05_format_for_training.py --dir $(SAMPLES_DIR) --out-dir $(TRAINING_DIR)

# STEP 4 + STEP 5 prep, full pipeline: sample -> validate -> generate questions ->
# split train/val -> format for training (see DESIGN.md/FINETUNING.md). Chains the
# 5 targets above in order so `data/training/train_formatted.jsonl` and
# `val_formatted.jsonl` come out ready for scripts/06_finetune.py. Example: make generate-dataset ROWS=70000
generate-dataset: sample-entities validate-samples generate-questions split-dataset format-for-training

# STEP 5 (fine-tuning), setup for the CUDA path: creates $(FINETUNE_VENV) with Unsloth
# and a CUDA build of PyTorch, then checks the GPU is usable. The unsloth floor makes
# an unresolvable TORCH_BACKEND fail loudly instead of silently picking an ancient,
# dependency-less unsloth release. Only reruns if the venv is missing
# (`rm -rf $(FINETUNE_VENV)` to rebuild). Example: make finetune-venv TORCH_BACKEND=cu128
finetune-venv: $(FINETUNE_VENV)/.ready

$(FINETUNE_VENV)/.ready:
	uv venv --python 3.12 $(FINETUNE_VENV)
	uv pip install --python $(FINETUNE_PY) --torch-backend=$(TORCH_BACKEND) "unsloth>=2026.9" duckdb==1.5.5
	$(FINETUNE_PY) -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('torch', torch.__version__, '-', torch.cuda.get_device_name(0), 'sm_%d%d' % torch.cuda.get_device_capability(0)); torch.ones(1, device='cuda')"
	$(FINETUNE_PY) -c "import unsloth; print('unsloth', unsloth.__version__)"
	touch $@

# Fine-tuning hyperparameters: each one is only passed when set, so an unset one
# keeps the script's own default (shown in brackets; `finetune` / `finetune-mlx` when
# they differ). Shared by both paths where the flag exists in both.
#   RANK          LoRA rank [32]
#   LORA_DROPOUT  LoRA dropout [0.0]
#   LR            learning rate [2e-4]
#   BATCH_SIZE    per-device batch size [2 / 4]
#   MAX_SEQ_LENGTH  max tokens per example [2048]
#   SEED          random seed [3407]
# `finetune` only:
#   EPOCHS        training epochs [2]
#   LORA_ALPHA    LoRA alpha [2 x RANK]
#   GRAD_ACCUM    gradient accumulation steps [4]
# `finetune-mlx` only (mlx-lm counts steps, not epochs):
#   ITERS         training iterations [600]
#   LORA_SCALE    mlx-lm's LoRA alpha/scale [20.0]
#
# Base model (Hugging Face id, downloaded on first use and cached in ~/.cache/huggingface/):
#   MODEL          [unsloth/Qwen3-0.6B-unsloth-bnb-4bit / mlx-community/Qwen3-0.6B-4bit]
#                  unset on `finetune`/`finetune-mlx`: interactive pick among the models
#                  already in the cache (scripts/model_picker.py), else the default
#                  also passed to `evaluate`, `merge-and-quantize` and `merge-and-quantize-mlx`:
#                  they must load the same base model the adapter was trained on, and
#                  output dirs are named after it (models/llaici-<model>-lora/-gguf...)
#   CHAT_TEMPLATE  Unsloth chat template name, `finetune` + `evaluate` only [qwen3-instruct]
#                  — change it along with MODEL when switching to another model family
UNSLOTH_MODEL_ARGS = $(if $(MODEL),--model-name $(MODEL)) $(if $(CHAT_TEMPLATE),--chat-template $(CHAT_TEMPLATE))
FINETUNE_COMMON_ARGS = $(if $(RANK),--rank $(RANK)) $(if $(LORA_DROPOUT),--lora-dropout $(LORA_DROPOUT)) \
	$(if $(MAX_SEQ_LENGTH),--max-seq-length $(MAX_SEQ_LENGTH)) $(if $(SEED),--seed $(SEED))

# STEP 5 (fine-tuning), §3/§4: QLoRA fine-tune Qwen3-0.6B (see FINETUNING.md, scripts/06_finetune.py)
# ⚠️ Requires a CUDA GPU — runs in the venv set up by `finetune-venv` (built automatically).
# Example: make finetune EPOCHS=3 RANK=16 LR=1e-4
#          make finetune MODEL=unsloth/Qwen3-1.7B-unsloth-bnb-4bit
finetune: $(FINETUNE_VENV)/.ready
	$(FINETUNE_PY) scripts/06_finetune.py $(FINETUNE_COMMON_ARGS) $(UNSLOTH_MODEL_ARGS) \
		$(if $(EPOCHS),--epochs $(EPOCHS)) $(if $(LR),--lr $(LR)) $(if $(LORA_ALPHA),--lora-alpha $(LORA_ALPHA)) \
		$(if $(BATCH_SIZE),--per-device-batch-size $(BATCH_SIZE)) $(if $(GRAD_ACCUM),--gradient-accumulation-steps $(GRAD_ACCUM))

# STEP 5, §3/§4 Apple Silicon alternative: same job as `finetune`, but via mlx-lm
# instead of Unsloth, since Unsloth requires a CUDA GPU (see FINETUNING.md).
# ⚠️ Requires `pip install "mlx-lm[train]"` on a Mac.
# Example: make finetune-mlx ITERS=1000 RANK=16
finetune-mlx:
	python3 scripts/06_finetune_mlx.py $(FINETUNE_COMMON_ARGS) $(if $(MODEL),--model $(MODEL)) \
		$(if $(ITERS),--iters $(ITERS)) $(if $(LR),--learning-rate $(LR)) $(if $(LORA_SCALE),--lora-scale $(LORA_SCALE)) \
		$(if $(BATCH_SIZE),--batch-size $(BATCH_SIZE))

# STEP 5, §5: evaluate the fine-tuned model's generated SQL against real DuckDB (see FINETUNING.md, scripts/07_evaluate.py)
# ⚠️ Same CUDA/Unsloth requirement as `finetune` — needs an adapter from that step first.
evaluate: $(FINETUNE_VENV)/.ready
	$(PRINT_THREADS)
	$(FINETUNE_PY) scripts/07_evaluate.py --threads $(THREADS) $(UNSLOTH_MODEL_ARGS)

# STEP 5, §6: merge the LoRA adapter and export to quantized GGUF (see FINETUNING.md, scripts/08_merge_and_quantize.py)
# ⚠️ Same CUDA/Unsloth requirement as `finetune` — needs an adapter from that step first.
merge-and-quantize: $(FINETUNE_VENV)/.ready
	$(FINETUNE_PY) scripts/08_merge_and_quantize.py $(if $(MODEL),--model-name $(MODEL))

# STEP 5, §6 Apple Silicon alternative: merge (mlx_lm.fuse) + convert/quantize to
# GGUF (llama.cpp's convert_hf_to_gguf.py) for an adapter from `finetune-mlx`.
# ⚠️ Requires `pip install "mlx-lm[train]"` and a local ggml-org/llama.cpp clone
# (see LLAMA_CPP_DIR). Example: make merge-and-quantize-mlx LLAMA_CPP_DIR=~/llama.cpp
merge-and-quantize-mlx:
	python3 scripts/08_merge_and_quantize_mlx.py --llama-cpp-dir $(LLAMA_CPP_DIR) $(if $(MODEL),--base-model $(MODEL))

# STEP 5, full pipeline (CUDA/Unsloth path): fine-tune -> evaluate -> merge/quantize
# to GGUF (FINETUNING.md §3-6). Assumes data/training/{train,val}_formatted.jsonl
# already exist (see `generate-dataset`). Chains the 3 targets above in order.
# ⚠️ Requires a CUDA GPU (venv set up automatically, see `finetune-venv`).
finetune-pipeline-cuda: finetune evaluate merge-and-quantize

# STEP 5, full pipeline (Apple Silicon/mlx-lm path): fine-tune -> merge/quantize to
# GGUF (FINETUNING.md §3-6 MLX alternatives). No MLX equivalent of `evaluate` is
# chained here: scripts/07_evaluate.py loads the model via Unsloth (CUDA-only),
# so it can't run against an mlx-lm adapter as-is (see FINETUNING.md §5).
# ⚠️ Requires `pip install "mlx-lm[train]"` and a local ggml-org/llama.cpp clone.
# Example: make finetune-pipeline-mlx LLAMA_CPP_DIR=~/llama.cpp
finetune-pipeline-mlx: finetune-mlx merge-and-quantize-mlx

build-up:
	docker compose build
	docker compose up

up:
	docker compose up

down:
	docker compose down -v

# Run the demo app (backend + frontend, see app/README.md) together, both with
# hot reload via bind-mounted source (docker-compose.yml). Needs llama-server
# running separately on the host — see app/README.md.
app-up:
	docker compose up backend frontend

app-build-up:
	docker compose build backend frontend
	docker compose up backend frontend

app-down:
	docker compose stop backend frontend
