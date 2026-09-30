"""Request/response bodies of the /query routes (api/query.py). Field names match
the frontend's `QueryResponse` type (app/frontend/src/types.ts)."""

from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, description="natural-language geographic question, FR or EN")


class QueryResponse(BaseModel):
    question: str
    sql: str = Field(description="the SQL the model generated, as executed")
    geojson: dict[str, Any] = Field(description="GeoJSON FeatureCollection of the result rows")
    feature_count: int
