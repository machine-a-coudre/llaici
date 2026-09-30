# LLaIci app

The demo application: ask a geographic question in natural language, see the
result on a map. Three pieces, run together:

```
┌─────────────┐   HTTP (question)   ┌──────────────┐   OpenAI-compatible API   ┌──────────────┐
│  frontend    │ ──────────────────> │   backend    │ ────────────────────────> │ llama-server  │
│ Vue + MapLibre│ <────────────────── │   FastAPI    │ <──────────────────────── │ (GGUF model) │
└─────────────┘   GeoJSON + count    └──────┬───────┘        SQL text           └──────────────┘
                                             │
                                             ▼ SQL
                                      DuckDB (data/db/llaici.duckdb)
```

1. **llama-server** serves the fine-tuned model (`scripts/08_merge_and_quantize.py`'s
   GGUF output) via an OpenAI-compatible API — see `docs/FINETUNING.md` §7.
2. **`backend/`** (FastAPI + LangChain): takes a question, calls llama-server for
   SQL, executes it against the project's DuckDB, returns GeoJSON. See
   `backend/README.md`.
3. **`frontend/`** (Vue 3 + TypeScript + MapLibre GL): a map with a prompt panel
   on the right. Submits the question to the backend, renders the returned
   GeoJSON, shows the feature count and the generated SQL.

## Run everything

**1. llama-server, on the host** (not in Docker). It serves exactly one GGUF, the one given with `-m`; there's no model list to configure. It comes from [llama.cpp](https://github.com/ggml-org/llama.cpp), built from the clone step 08 already uses (a CPU build is plenty for a 0.6B model; add `-DGGML_CUDA=ON` for GPU, which needs the CUDA toolkit):

```bash
cmake -S ~/llama.cpp -B ~/llama.cpp/build && cmake --build ~/llama.cpp/build --config Release -j --target llama-server   # once
~/llama.cpp/build/bin/llama-server -m models/llaici-qwen3-0.6b-gguf/llaici-qwen3-0.6b.q8_0.gguf --port 8080 --host 0.0.0.0 --jinja
```

- `--jinja`: applies the chat template embedded in the GGUF (the one the model was trained with).
- `--host 0.0.0.0`: the backend container reaches the host through Docker's bridge, so llama-server can't listen on `127.0.0.1` only (its default). This also exposes it on your local network: fine for a local test, or use `--host 172.17.0.1` (the `docker0` address) to stay off the LAN.

**2. Backend + frontend, in Docker**: stop the DuckDB UI container first (`make down`: it holds a write lock on the database), then:

```bash
make app-up          # or make app-build-up after changing requirements/Dockerfiles
```

Then open `http://localhost:5173` in a browser with **WebGL2** enabled (required by MapLibre GL 6, see "The map stays blank" in `docs/TROUBLESHOOTING.md`). The backend reaches llama-server at `http://host.docker.internal:8080/v1` (`LLAMA_SERVER_URL` in `docker-compose.yml`), mapped to the host on Linux too (`extra_hosts`).

**Without Docker**, for the backend and frontend:

```bash
# backend (from the REPO ROOT — see backend/README.md)
python3 -m venv app/backend/.venv && source app/backend/.venv/bin/activate
pip install -r app/backend/requirements.txt
uvicorn app.main:app --app-dir app/backend --reload --port 8000

# frontend
cd app/frontend && npm install && npm run dev
```

## Status

- **Frontend**: `npm install`, `vue-tsc` type-check, and `npm run build` all verified to pass (0 vulnerabilities after bumping `maplibre-gl` to 6.10.0 — the installed 4.x line had a critical XSS advisory, [GHSA-jrc7-96c5-q579](https://github.com/advisories/GHSA-jrc7-96c5-q579)). Not run against a live backend (no dev server started, no visual check).
- **Backend**: the DuckDB→GeoJSON conversion (`duckdb_service.py`) was tested directly against the real database and works. The LLM call (`llm.py`) has not been tested — it needs a running `llama-server`, which needs a fine-tuned GGUF model, which doesn't exist yet (see `docs/FINETUNING.md` §3-§6, none of which have been run in this development environment — no CUDA GPU available).

## Known limitations, not addressed here

- No auth, no rate limiting — this is a local demo app, not hardened for exposure beyond `localhost`.
- CORS is wide open (`allow_origins=["*"]`) in the backend — fine for local dev via the Vite proxy, not for any real deployment.
- The basemap is [OpenFreeMap](https://openfreemap.org)'s "Liberty" style — free, no API key, full OSM detail (countries, regions, cities, roads). No rate limits documented by the provider as of writing, but it's a third-party public service outside this project's control — self-host OpenFreeMap's tiles (or switch to a paid provider) if reliability becomes a concern.
- No handling yet for a query the model generates that references a column other than `geometry`/`id`/`name` in unexpected ways, or a non-SELECT statement slipping through despite fine-tuning (the backend's `UnsafeQueryError` check is defense-in-depth, not exhaustively tested against adversarial input).
