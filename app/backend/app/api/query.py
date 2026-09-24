"""Routes: question in, GeoJSON out.

Flow: user question -> llama-server generates SQL (services.llm) -> SQL executed
against DuckDB, result converted to GeoJSON (services.duckdb_service) -> returned
to the Vue/MapLibre frontend.
"""

import random

from fastapi import APIRouter, HTTPException

from ..models.schemas import QueryRequest, QueryResponse
from ..services import duckdb_service, llm

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.post("/query", response_model=QueryResponse)
async def query(payload: QueryRequest) -> QueryResponse:
    try:
        sql = await llm.generate_sql(payload.question)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}") from e

    try:
        geojson = duckdb_service.run_query(sql)
    except duckdb_service.UnsafeQueryError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except duckdb_service.QueryExecutionError as e:
        raise HTTPException(status_code=422, detail=f"Generated SQL failed to execute: {e}") from e

    return QueryResponse(
        question=payload.question,
        sql=sql,
        geojson=geojson,
        feature_count=len(geojson["features"]),
    )


# ── Temporary: lets the frontend (map highlight/zoom behavior) be built and
# tested without a running llama-server or fine-tuned model (see FINETUNING.md
# status — nothing in scripts/06-08 has been run yet). Skips services.llm
# entirely and feeds a fixed, hand-written SQL query straight to
# duckdb_service.run_query — the same function the real /query route uses — so
# the DuckDB/GeoJSON path is exercised for real, only the LLM call is faked.
# Remove once /query can be exercised for real, or keep around as a frontend
# dev fixture — either way, the frontend should stop calling this once the
# real pipeline works (see src/api.ts's askQuestionMock).
_MOCK_SQL_CHOICES = [
    "SELECT id, name, geometry FROM divisions WHERE name ILIKE '%Bilbao%' LIMIT 1",
    "SELECT id, name, geometry FROM divisions WHERE name ILIKE '%Madrid%' LIMIT 1",
    "SELECT id, name, geometry FROM divisions WHERE name ILIKE '%Santander%' LIMIT 1",
    # division_areas (unlike divisions) holds polygon geometry — administrative
    # boundaries — so this one exercises the frontend's polygon rendering path.
    # Exact name match + largest-area tie-break: `ILIKE '%Madrid%' LIMIT 1` with
    # no ordering picked an arbitrary match (e.g. "Madridejos", or a tiny
    # sub-locality a few dozen meters across) rather than the actual city.
    "SELECT id, name, geometry FROM division_areas WHERE name = 'Madrid' "
    "ORDER BY ST_Area(geometry) DESC LIMIT 1",
]


@router.post("/query/mock", response_model=QueryResponse)
async def query_mock(payload: QueryRequest) -> QueryResponse:
    sql = random.choice(_MOCK_SQL_CHOICES)
    try:
        geojson = duckdb_service.run_query(sql)
    except duckdb_service.UnsafeQueryError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except duckdb_service.QueryExecutionError as e:
        raise HTTPException(status_code=422, detail=f"Mock SQL failed to execute: {e}") from e

    return QueryResponse(
        question=payload.question,
        sql=sql,
        geojson=geojson,
        feature_count=len(geojson["features"]),
    )
