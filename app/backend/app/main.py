"""FastAPI backend: question in, GeoJSON out.

Route handlers live in app.api.query; this module only wires up the FastAPI app
(middleware, router registration).

Run from the repo root (so the default relative DB path resolves correctly):
    uvicorn app.main:app --app-dir app/backend --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.query import router as query_router

app = FastAPI(title="LLaIci API")

# Dev convenience: the Vite dev server proxies /api to this backend (see
# frontend/vite.config.ts), so CORS shouldn't normally matter in dev. Left open
# here mainly for the case of calling the API directly (e.g. curl, a built
# frontend served from a different origin) — tighten allow_origins before any
# real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query_router)
