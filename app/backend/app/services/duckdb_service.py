"""Executes model-generated SQL against the read-only llaici DuckDB and converts
the result to GeoJSON for the frontend map.

Every template in TEMPLATES.md SELECTs a `geometry` column — this module assumes
that convention holds for model-generated SQL too (the model was fine-tuned only
on that shape of query, see FINETUNING.md).
"""

import json

import duckdb

from .. import config


class UnsafeQueryError(Exception):
    """Raised when the generated SQL isn't a read-only SELECT/WITH statement.

    Defense-in-depth, not the primary safety mechanism: the model is only ever
    fine-tuned on SELECT queries (TEMPLATES.md), so this should never actually
    trigger in normal operation — but the DB connection is also opened
    read_only=True regardless, which is the real guarantee.
    """


class QueryExecutionError(Exception):
    """Raised when the generated SQL is syntactically invalid or fails to run."""


def _connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(config.DB_PATH, read_only=True)
    con.execute(f"SET threads={config.DB_THREADS}")  # see scripts/01_sample_entities.py's connect() note
    con.execute("LOAD spatial")
    return con


def _validate(sql: str) -> None:
    stripped = sql.strip().lower()
    if not (stripped.startswith("select") or stripped.startswith("with")):
        raise UnsafeQueryError("Only SELECT/WITH queries are allowed")


def run_query(sql: str) -> dict:
    """Executes `sql` and returns a GeoJSON FeatureCollection.

    Wraps the model's query rather than requiring it to call ST_AsGeoJSON itself
    (it was never trained to) — `ST_AsGeoJSON(geometry)` is computed here in an
    outer SELECT, and every other selected column becomes a GeoJSON Feature
    property.
    """
    _validate(sql)

    wrapped = f"""
        SELECT * EXCLUDE (geometry), ST_AsGeoJSON(geometry) AS __geojson__
        FROM ({sql.rstrip(';')}) AS __llaici_result__
    """

    con = _connect()
    try:
        relation = con.sql(wrapped)
        columns = relation.columns
        rows = relation.fetchall()
    except duckdb.Error as e:
        raise QueryExecutionError(str(e)) from e
    finally:
        con.close()

    geo_index = columns.index("__geojson__")
    prop_indices = [i for i, name in enumerate(columns) if name != "__geojson__"]
    prop_names = [columns[i] for i in prop_indices]

    features = []
    for row in rows:
        geometry = json.loads(row[geo_index])
        properties = {prop_names[j]: row[prop_indices[j]] for j in range(len(prop_indices))}
        features.append({"type": "Feature", "geometry": geometry, "properties": properties})

    return {"type": "FeatureCollection", "features": features}
