"""Backend configuration, all overridable via environment variables.

Defaults assume: llama-server running locally (FINETUNING.md §7) and this backend
started from the repo root (so the relative DuckDB path resolves the same way
every scripts/*.py in this project already assumes).
"""

import os

# llama-server exposes an OpenAI-compatible API under /v1 when started with
# `llama-server -m <gguf> --port 8080` (see FINETUNING.md §7).
LLAMA_SERVER_URL = os.environ.get("LLAMA_SERVER_URL", "http://localhost:8080/v1")

# The model name llama-server reports doesn't have to match anything specific —
# llama-server serves whichever single GGUF it was started with regardless of
# this value — but the OpenAI client requires a non-empty string.
MODEL_NAME = os.environ.get("LLAICI_MODEL_NAME", "llaici-qwen3-0.6b")

# Same relative path convention as scripts/01-05 (DB_PATH = "data/db/llaici.duckdb"):
# assumes the process is started from the repo root.
DB_PATH = os.environ.get("LLAICI_DB_PATH", "data/db/llaici.duckdb")
