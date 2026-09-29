#!/usr/bin/env python3
"""STEP 4, steps 2-3 — fill templates with sampled entities and validate by execution.

Reads `data/samples/entities.jsonl` (produced by `sample_entities.py` /
`make sample-entities`), fills in the matching SQL template from TEMPLATES.md with
each row's sampled values, executes the result on DuckDB, and keeps only the pairs
that return a non-empty result (DESIGN.md STEP 4: "Empty result -> discard the pair").

Not implemented here (later STEP 4 sub-steps, see DESIGN.md): generating NL question
formulations, deduplication/balancing, or writing the final dataset.jsonl. This
script's output is {template, params, sql, country} per validated row — the
geometric result itself is never kept (DESIGN.md: "used only for validation, NEVER
included in the dataset").

Since sampling is join-driven (every sampled row already comes from a real match),
validated pairs are expected to be ~100% of input rows — this step is a safety net
confirming that, not the primary filter DESIGN.md originally envisioned for a
random-then-discard sampling strategy (which this project doesn't use; see DESIGN.md
STEP 4 "Decisions").

Run:
    uvx --with duckdb python3 scripts/02_fill_and_validate.py
    uvx --with duckdb python3 scripts/02_fill_and_validate.py --in data/samples/entities.jsonl --out data/samples/validated.jsonl
"""

import argparse
import json
import sys

import duckdb

DB_PATH = "data/db/llaici.duckdb"

# The 7 name columns TEMPLATES.md spells out explicitly (template #1); every other
# template abbreviates the same set with "/* ...other name_* columns */".
NAME_COLUMNS = ["name", "name_fr", "name_es", "name_it", "name_de", "name_en", "name_pt"]


def esc(value: str) -> str:
    """Escape a value for interpolation into a SQL string literal."""
    return value.replace("'", "''")


def name_match(alias: str, term: str) -> str:
    term = esc(term)
    return " OR ".join(f"{alias}.{col} ILIKE '%{term}%'" for col in NAME_COLUMNS)


def name_match_bare(term: str) -> str:
    """Same as name_match, but for a CTE with no table alias (templates #3, #8, left/right bank)."""
    term = esc(term)
    return " OR ".join(f"{col} ILIKE '%{term}%'" for col in NAME_COLUMNS)


def azimuth_expr() -> str:
    return (
        "degrees(ST_Azimuth("
        "ST_Point2D(ST_X(ref.geometry), ST_Y(ref.geometry)), "
        "ST_Point2D(ST_X(ST_Centroid(e.geometry)), ST_Y(ST_Centroid(e.geometry)))"
        "))"
    )


def angle_condition(direction: str, angle_min: float, angle_max: float) -> str:
    expr = azimuth_expr()
    if direction == "north":
        return f"({expr} >= 315 OR {expr} < 45)"
    return f"{expr} BETWEEN {angle_min} AND {angle_max}"


def build_containment(p: dict) -> str:
    return f"""
        SELECT e.id, e.name, e.geometry
        FROM infrastructures e
        JOIN division_areas da ON ST_Within(e.geometry, da.geometry)
        WHERE e.subtype = '{esc(p['subtype'])}' AND e.class = '{esc(p['class'])}'
          AND ({name_match('da', p['place'])})
    """


def build_proximity(p: dict) -> str:
    return f"""
        SELECT e.id, e.name, e.geometry
        FROM infrastructures e, divisions ref
        WHERE e.subtype = '{esc(p['subtype'])}' AND e.class = '{esc(p['class'])}'
          AND ({name_match('ref', p['place'])})
          AND ST_DWithin_Spheroid(
                ST_Point2D(ST_X(ST_Centroid(e.geometry)), ST_Y(ST_Centroid(e.geometry))),
                ST_Point2D(ST_X(ST_ClosestPoint(ref.geometry, e.geometry)), ST_Y(ST_ClosestPoint(ref.geometry, e.geometry))),
                {p['distance_m']})
    """


def build_along(p: dict) -> str:
    return f"""
        WITH segments AS (
            SELECT geometry AS line
            FROM water
            WHERE class IN ('river', 'stream', 'canal')
              AND ST_GeometryType(geometry) = 'LINESTRING'
              AND ({name_match_bare(p['feature'])})
        ),
        bbox AS (
            SELECT line,
                   ST_Expand(
                     ST_Envelope(line),
                     {p['corridor_m']} / (111320.0 * cos(radians(
                       (ST_YMin(ST_Envelope(line)) + ST_YMax(ST_Envelope(line))) / 2
                     )))
                   ) AS box
            FROM segments
        ),
        candidates AS (
            SELECT e.id, e.name, e.geometry, ST_Centroid(e.geometry) AS cgeom, b.line,
                   row_number() OVER (PARTITION BY e.id ORDER BY ST_Distance(b.line, ST_Centroid(e.geometry))) AS rn
            FROM infrastructures e, bbox b
            WHERE e.subtype = '{esc(p['subtype'])}' AND e.class = '{esc(p['class'])}'
              AND ST_Intersects(e.geometry, b.box)
        )
        SELECT id, name, geometry
        FROM candidates
        WHERE rn = 1
          AND ST_DWithin_Spheroid(
                ST_Point2D(ST_X(cgeom), ST_Y(cgeom)),
                ST_Point2D(ST_X(ST_LineInterpolatePoint(line, ST_LineLocatePoint(line, cgeom))), ST_Y(ST_LineInterpolatePoint(line, ST_LineLocatePoint(line, cgeom)))),
                {p['corridor_m']})
    """


def build_center(p: dict) -> str:
    return f"""
        SELECT e.id, e.name, e.geometry
        FROM infrastructures e
        JOIN division_areas da ON ({name_match('da', p['place'])})
        WHERE e.subtype = '{esc(p['subtype'])}' AND e.class = '{esc(p['class'])}'
          AND ST_DWithin_Spheroid(
                ST_Point2D(ST_X(ST_Centroid(e.geometry)), ST_Y(ST_Centroid(e.geometry))),
                ST_Point2D(ST_X(ST_Centroid(da.geometry)), ST_Y(ST_Centroid(da.geometry))),
                GREATEST(300, LEAST(5000, 0.15 * SQRT(ST_Area_Spheroid(da.geometry)))))
    """


def build_periphery(p: dict) -> str:
    return f"""
        SELECT e.id, e.name, e.geometry
        FROM infrastructures e
        JOIN division_areas da ON ({name_match('da', p['place'])})
        WHERE e.subtype = '{esc(p['subtype'])}' AND e.class = '{esc(p['class'])}'
          AND ST_Within(e.geometry, da.geometry)
          AND ST_DWithin_Spheroid(
                ST_Point2D(ST_X(ST_Centroid(e.geometry)), ST_Y(ST_Centroid(e.geometry))),
                ST_Point2D(ST_X(ST_ClosestPoint(ST_Boundary(da.geometry), e.geometry)), ST_Y(ST_ClosestPoint(ST_Boundary(da.geometry), e.geometry))),
                {p['band_m']})
    """


def build_direction(p: dict) -> str:
    return f"""
        SELECT e.id, e.name, e.geometry
        FROM infrastructures e, divisions ref
        WHERE e.subtype = '{esc(p['subtype'])}' AND e.class = '{esc(p['class'])}'
          AND ({name_match('ref', p['place'])})
          AND {angle_condition(p['direction'], p['angle_min'], p['angle_max'])}
          AND ST_Distance_Spheroid(
                ST_Point2D(ST_X(ref.geometry), ST_Y(ref.geometry)),
                ST_Point2D(ST_X(ST_Centroid(e.geometry)), ST_Y(ST_Centroid(e.geometry)))
              ) <= 100000
    """


def build_direction_distance(p: dict) -> str:
    return f"""
        SELECT e.id, e.name, e.geometry
        FROM infrastructures e, divisions ref
        WHERE e.subtype = '{esc(p['subtype'])}' AND e.class = '{esc(p['class'])}'
          AND ({name_match('ref', p['place'])})
          AND {angle_condition(p['direction'], p['angle_min'], p['angle_max'])}
          AND ST_Distance_Spheroid(
                ST_Point2D(ST_X(ref.geometry), ST_Y(ref.geometry)),
                ST_Point2D(ST_X(ST_Centroid(e.geometry)), ST_Y(ST_Centroid(e.geometry)))
              ) BETWEEN {p['distance_m']} - GREATEST(200, LEAST(0.20 * {p['distance_m']}, 10000))
                    AND {p['distance_m']} + GREATEST(200, LEAST(0.20 * {p['distance_m']}, 10000))
    """


def build_area_distance(p: dict) -> str:
    return f"""
        WITH ref AS (
            SELECT geometry
            FROM water
            WHERE ({name_match_bare(p['place'])})
            LIMIT 1
        ),
        bbox AS (
            SELECT geometry AS line,
                   ST_Expand(
                     ST_Envelope(geometry),
                     {p['distance_m']} / (111320.0 * cos(radians(
                       (ST_YMin(ST_Envelope(geometry)) + ST_YMax(ST_Envelope(geometry))) / 2
                     )))
                   ) AS box
            FROM ref
        )
        SELECT e.id, e.name, e.geometry
        FROM infrastructures e, bbox
        WHERE e.subtype = '{esc(p['subtype'])}' AND e.class = '{esc(p['class'])}'
          AND ST_Intersects(e.geometry, bbox.box)
          AND ST_DWithin_Spheroid(
                ST_Point2D(ST_X(ST_Centroid(e.geometry)), ST_Y(ST_Centroid(e.geometry))),
                ST_Point2D(ST_X(ST_ClosestPoint(bbox.line, e.geometry)), ST_Y(ST_ClosestPoint(bbox.line, e.geometry))),
                {p['distance_m']})
    """


def build_left_right_bank(p: dict) -> str:
    return f"""
        WITH segments AS (
            SELECT geometry AS line
            FROM water
            WHERE class IN ('river', 'stream', 'canal')
              AND ST_GeometryType(geometry) = 'LINESTRING'
              AND ({name_match_bare(p['feature'])})
        ),
        bbox AS (
            SELECT line,
                   ST_Expand(
                     ST_Envelope(line),
                     {p['distance_m']} / (111320.0 * cos(radians(
                       (ST_YMin(ST_Envelope(line)) + ST_YMax(ST_Envelope(line))) / 2
                     )))
                   ) AS box
            FROM segments
        ),
        candidates AS (
            SELECT e.id, e.name, ST_Centroid(e.geometry) AS cgeom, b.line,
                   row_number() OVER (PARTITION BY e.id ORDER BY ST_Distance(b.line, ST_Centroid(e.geometry))) AS rn
            FROM infrastructures e, bbox b
            WHERE e.subtype = '{esc(p['subtype'])}' AND e.class = '{esc(p['class'])}'
              AND ST_Intersects(e.geometry, b.box)
        ),
        nearest AS (
            SELECT id, name, cgeom, line, ST_LineLocatePoint(line, cgeom) AS f
            FROM candidates WHERE rn = 1
        )
        SELECT id, name
        FROM nearest
        WHERE ST_DWithin_Spheroid(
                ST_Point2D(ST_X(cgeom), ST_Y(cgeom)),
                ST_Point2D(ST_X(ST_LineInterpolatePoint(line, f)), ST_Y(ST_LineInterpolatePoint(line, f))),
                {p['distance_m']})
          AND sign(
            (ST_X(ST_LineInterpolatePoint(line, LEAST(f + 0.0001, 1))) - ST_X(ST_LineInterpolatePoint(line, GREATEST(f - 0.0001, 0))))
          * (ST_Y(cgeom) - ST_Y(ST_LineInterpolatePoint(line, f)))
          - (ST_Y(ST_LineInterpolatePoint(line, LEAST(f + 0.0001, 1))) - ST_Y(ST_LineInterpolatePoint(line, GREATEST(f - 0.0001, 0))))
          * (ST_X(cgeom) - ST_X(ST_LineInterpolatePoint(line, f)))
        ) = {p['side_sign']}
    """


def build_places_containment(p: dict) -> str:
    return f"""
        SELECT p.id, p.name, p.geometry
        FROM places p
        JOIN division_areas da ON ST_Within(p.geometry, da.geometry)
        WHERE list_contains(p.category_hierarchy, '{esc(p['category'])}')
          AND ({name_match('da', p['place'])})
    """


def build_places_proximity(p: dict) -> str:
    return f"""
        SELECT p.id, p.name, p.geometry
        FROM places p, divisions ref
        WHERE list_contains(p.category_hierarchy, '{esc(p['category'])}')
          AND ({name_match('ref', p['place'])})
          AND ST_DWithin_Spheroid(
                ST_Point2D(ST_X(p.geometry), ST_Y(p.geometry)),
                ST_Point2D(ST_X(ref.geometry), ST_Y(ref.geometry)),
                {p['distance_m']})
    """


def build_bordering(p: dict) -> str:
    return f"""
        WITH ref AS (
            SELECT geometry
            FROM division_areas
            WHERE ({name_match_bare(p['place'])})
            LIMIT 1
        ),
        bbox AS (
            SELECT geometry AS area,
                   ST_Expand(
                     ST_Envelope(geometry),
                     {p['distance_m']} / (111320.0 * cos(radians(
                       (ST_YMin(ST_Envelope(geometry)) + ST_YMax(ST_Envelope(geometry))) / 2
                     )))
                   ) AS box
            FROM ref
        )
        SELECT d.id, d.name, d.geometry
        FROM divisions d, bbox
        WHERE d.subtype = 'locality'
          AND ST_Intersects(d.geometry, bbox.box)
          AND ST_DWithin_Spheroid(
                ST_Point2D(ST_X(d.geometry), ST_Y(d.geometry)),
                ST_Point2D(ST_X(ST_ClosestPoint(ST_Boundary(bbox.area), d.geometry)), ST_Y(ST_ClosestPoint(ST_Boundary(bbox.area), d.geometry))),
                {p['distance_m']})
    """


def _build_city_direction(p: dict, distance_condition: str) -> str:
    return f"""
        WITH ref AS (
            SELECT geometry
            FROM divisions
            WHERE ({name_match_bare(p['place'])})
            ORDER BY population DESC NULLS LAST
            LIMIT 1
        ),
        city AS (
            SELECT e.id, e.name, e.geometry
            FROM divisions e, ref
            WHERE e.subtype = 'locality'
              AND {angle_condition(p['direction'], p['angle_min'], p['angle_max'])}
              AND {distance_condition}
        ),
        ranked_area AS (
            SELECT da.division_id, da.geometry,
                   row_number() OVER (PARTITION BY da.division_id ORDER BY ST_Area(da.geometry) DESC) AS rn
            FROM division_areas da
            WHERE da.is_land = true
              AND da.division_id IN (SELECT id FROM city)
        )
        SELECT city.id, city.name, COALESCE(area.geometry, city.geometry) AS geometry
        FROM city
        LEFT JOIN ranked_area area ON area.division_id = city.id AND area.rn = 1
    """


def build_city_direction(p: dict) -> str:
    distance_condition = f"""
        ST_Distance_Spheroid(
            ST_Point2D(ST_X(ref.geometry), ST_Y(ref.geometry)),
            ST_Point2D(ST_X(e.geometry), ST_Y(e.geometry))
        ) <= 100000
    """
    return _build_city_direction(p, distance_condition)


def build_city_direction_distance(p: dict) -> str:
    distance_condition = f"""
        ST_Distance_Spheroid(
            ST_Point2D(ST_X(ref.geometry), ST_Y(ref.geometry)),
            ST_Point2D(ST_X(e.geometry), ST_Y(e.geometry))
        ) BETWEEN {p['distance_m']} - GREATEST(200, LEAST(0.20 * {p['distance_m']}, 10000))
              AND {p['distance_m']} + GREATEST(200, LEAST(0.20 * {p['distance_m']}, 10000))
    """
    return _build_city_direction(p, distance_condition)


def build_show_division(p: dict) -> str:
    return f"""
        WITH d AS (
            SELECT id, name, geometry
            FROM divisions
            WHERE ({name_match_bare(p['place'])})
            ORDER BY population DESC NULLS LAST
            LIMIT 1
        ),
        area AS (
            SELECT da.id, da.name, da.geometry
            FROM division_areas da
            JOIN d ON da.division_id = d.id
            WHERE da.is_land = true
            ORDER BY ST_Area(da.geometry) DESC
            LIMIT 1
        )
        SELECT id, name, geometry FROM d
        UNION ALL
        SELECT id, name, geometry FROM area
    """


BUILDERS = {
    "containment": build_containment,
    "proximity": build_proximity,
    "along": build_along,
    "center": build_center,
    "periphery": build_periphery,
    "direction": build_direction,
    "direction_distance": build_direction_distance,
    "area_distance": build_area_distance,
    "left_right_bank": build_left_right_bank,
    "places_containment": build_places_containment,
    "places_proximity": build_places_proximity,
    "bordering": build_bordering,
    "show_division": build_show_division,
    "city_direction": build_city_direction,
    "city_direction_distance": build_city_direction_distance,
}


def normalize_sql(sql: str) -> str:
    """Collapse the builder functions' multi-line/indented SQL into a single line."""
    return " ".join(sql.split())


def connect(threads: int = 2) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(DB_PATH, read_only=True)
    # Default low on purpose, to stay light on the user's machine — see the same
    # note in sample_entities.py's connect(). More RAM doesn't speed this up; more
    # CPU cores (and raising --threads, up to roughly the machine's core count) would.
    con.execute(f"SET threads={threads}")
    con.execute("INSTALL spatial")  # no-op once installed; host runs lack the image's pre-install
    con.execute("LOAD spatial")
    return con


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", default="data/samples/entities.jsonl")
    parser.add_argument("--out", dest="out_path", default="data/samples/validated.jsonl")
    parser.add_argument(
        "--threads",
        type=int,
        default=2,
        help="DuckDB SET threads=N (default 2, kept low on purpose); tune to roughly "
        "the machine's CPU core count for a faster large run (e.g. --threads 10)",
    )
    args = parser.parse_args()

    con = connect(args.threads)
    kept, discarded, errors = 0, 0, 0

    with open(args.in_path, encoding="utf-8") as fin, open(args.out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            row = json.loads(line)
            template = row["template"]
            params = {k: v for k, v in row.items() if k not in ("template", "country")}
            sql = BUILDERS[template](params).strip()

            try:
                result = con.execute(sql).fetchone()
            except duckdb.Error as e:
                errors += 1
                print(f"# ERROR ({template}, {row.get('place') or row.get('feature')}): {e}", file=sys.stderr)
                continue

            if result is None:
                discarded += 1
                continue

            kept += 1
            fout.write(json.dumps({
                "template": template,
                "params": params,
                "sql": normalize_sql(sql),
                "country": row.get("country"),
            }, ensure_ascii=False) + "\n")

    print(f"# kept: {kept}, discarded (empty result): {discarded}, errors: {errors}", file=sys.stderr)
    print(f"# written to {args.out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
