# LLaIci backend

FastAPI service: takes a natural-language question, gets SQL from the fine-tuned
model via a local `llama-server` (LangChain `ChatOpenAI` client against its
OpenAI-compatible API), executes it against the project's DuckDB, and returns the
result as GeoJSON.

## Prerequisites

- `llama-server` running with the GGUF model from `scripts/08_merge_and_quantize.py`
  (see `docs/FINETUNING.md` §7), e.g.:
  ```bash
  llama-server -m ../../models/llaici-qwen3-0.6b-gguf/<file>.gguf --port 8080
  ```
- `data/db/llaici.duckdb` populated (see the repo root `README.md`).

## Run

From the **repo root** (not this directory — the default DB path is relative to it):

```bash
python3 -m venv app/backend/.venv
source app/backend/.venv/bin/activate
pip install -r app/backend/requirements.txt
uvicorn app.main:app --app-dir app/backend --reload --port 8000
```

Environment variables (all optional, see `app/config.py`):

| Variable | Default | Purpose |
|---|---|---|
| `LLAMA_SERVER_URL` | `http://localhost:8080/v1` | llama-server's OpenAI-compatible endpoint |
| `LLAICI_MODEL_NAME` | `llaici-qwen3-0.6b` | Sent to the OpenAI client; llama-server ignores it (serves whichever GGUF it was started with) |
| `LLAICI_DB_PATH` | `data/db/llaici.duckdb` | Path to the DuckDB file, relative to the process's working directory |
| `LLAICI_DB_THREADS` | `2` | DuckDB `SET threads=N` per connection — kept low by default, raise it up to roughly the host's CPU core count for lower query latency |

## API

`POST /query` `{"question": "restaurants in Madrid"}` →
```json
{
  "question": "restaurants in Madrid",
  "sql": "SELECT ...",
  "geojson": {"type": "FeatureCollection", "features": [...]},
  "feature_count": 12
}
```

`GET /health` → `{"status": "ok"}`

⚠️ Not run/tested end-to-end — same reason as `scripts/06-08`: no fine-tuned GGUF model exists yet to point `llama-server` at (see `docs/FINETUNING.md`).
