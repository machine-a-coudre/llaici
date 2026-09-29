#!/usr/bin/env python3
"""STEP 4 sampling — draws real, join-valid parameters for the SQL templates in TEMPLATES.md.

This only implements the *sampling* part of STEP 4 (DESIGN.md): for a given template,
produce real (subtype, class, place/feature name, distance...) combinations that are
guaranteed to yield a non-empty result when substituted into that template's SQL,
because they were derived from an actual spatial join rather than guessed at random.

Not implemented here (later STEP 4 sub-steps, see DESIGN.md): filling the SQL template
strings, generating NL question formulations, deduplication/balancing, or writing
dataset.jsonl. This script only prints sampled parameter sets as JSON lines.

Sampling strategy (see DESIGN.md STEP 4 "Decisions"):
- join-driven: every parameter set comes from a real spatial join (e.g. a real bus
  stop that is really within a real city's polygon), not from randomly guessing a
  (subtype, place) combo and hoping it matches something.
- country-stratified: a random country is picked first (from `division_areas`,
  subtype='country'), then entities are sampled inside it, so densely-mapped
  countries/cities (Paris, Barcelona kept coming up in ad hoc testing) don't
  dominate the sample.
- bbox-prefiltered: every join against `water`/`division_areas` geometries is
  restricted to a bounding box first (see TEMPLATES.md "Bounding-box prefilter"),
  to avoid the OOM crash documented there (an unfiltered join against the full
  ~13M-row `infrastructures` view exhausted 19+ GiB of memory).
- per-attempt hard timeout: even with the bbox prefilters and plain (non-random)
  `LIMIT 1` on every final join against `infrastructures`, a small fraction of random
  (country, place/segment) picks can still take unexpectedly long — observed
  non-reproducibly across several different samplers during testing, each hanging
  for minutes at ~200% CPU. A `con.interrupt()`-based watchdog was tried first and
  didn't work: DuckDB only checks for an interrupt at certain internal checkpoints,
  which the slow query shapes never reached, so the watchdog fired without actually
  stopping anything. Each attempt now runs in its own subprocess
  (`attempt_with_hard_timeout`), killed outright after QUERY_TIMEOUT_S if it hasn't
  returned — killing the OS process is the one thing guaranteed to work regardless
  of what DuckDB is doing internally. A killed or otherwise-failing attempt is
  counted as a miss like any other.

Run (or via `make sample-entities ROWS=100`, see Makefile):
    uvx --with duckdb python3 scripts/01_sample_entities.py --rows 100
    uvx --with duckdb python3 scripts/01_sample_entities.py --rows 70000 --out data/samples/entities.jsonl
    uvx --with duckdb python3 scripts/01_sample_entities.py --template along --rows 20
"""

import argparse
import json
import multiprocessing
import random
import sys
from pathlib import Path

import duckdb

DB_PATH = "data/db/llaici.duckdb"
QUERY_TIMEOUT_S = 8

# Expands a geometry's bounding box by `dist_m` meters, converted to degrees.
# See TEMPLATES.md "Bounding-box prefilter" for why the longitude (larger) degree
# delta is deliberately used for both axes: over-inclusive is safe for a prefilter,
# the exact ST_DWithin_Spheroid check afterward catches anything let through.
BBOX_SQL = """
    ST_Expand(
        ST_Envelope({geom}),
        {dist_m} / (111320.0 * cos(radians(
            (ST_YMin(ST_Envelope({geom})) + ST_YMax(ST_Envelope({geom}))) / 2
        )))
    )
"""


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(DB_PATH, read_only=True)
    # Hardcoded low on purpose, to stay light on the user's machine (see chat history
    # around the OOM/hang incidents this script went through). This — not RAM — is
    # the main lever for speed: DuckDB won't use more CPU cores than this regardless
    # of available memory. Raise it if more CPU cores are available and speed matters
    # more than staying light (e.g. for a real ROWS=70000 run on a beefier machine).
    con.execute("SET threads=2")
    con.execute("INSTALL spatial")  # no-op once installed; host runs lack the image's pre-install
    con.execute("LOAD spatial")
    return con


def _attempt_worker(template_name: str, countries: list[tuple[str, str]], q: "multiprocessing.Queue") -> None:
    con = connect()
    country_id, country_name = random.choice(countries)
    result = SAMPLERS[template_name](con, country_id)
    if result is not None:
        result["country"] = country_name
    q.put(result)


def attempt_with_hard_timeout(template_name: str, countries: list[tuple[str, str]]) -> dict | None:
    """Run one full attempt (pick a country, run the sampler) in its own process,
    killed after QUERY_TIMEOUT_S if it hasn't returned.

    See module docstring "per-attempt timeout": ORDER BY random() LIMIT 1 occasionally
    hits an unexpectedly large candidate set for an otherwise-ordinary random pick.
    A `con.interrupt()`-based watchdog was tried first but proved unreliable — DuckDB
    only checks for an interrupt at certain internal checkpoints, which some query
    shapes never reach, so the watchdog fired without the query actually stopping
    (observed directly: a run kept accumulating CPU time well past the timeout).
    Killing the whole OS process is the one thing guaranteed to actually stop it.
    A timed-out or otherwise-failing attempt returns None (counted as a miss).
    """
    q: multiprocessing.Queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=_attempt_worker, args=(template_name, countries, q))
    p.start()
    p.join(QUERY_TIMEOUT_S)
    if p.is_alive():
        p.terminate()
        p.join()
        return None
    try:
        return q.get_nowait()
    except Exception:
        return None


def fetch_countries(con: duckdb.DuckDBPyConnection) -> list[tuple[str, str]]:
    """Fetches every (division_areas.id, name) with subtype='country' once, up front.

    Every attempt used to re-run `ORDER BY random() LIMIT 1` on this table itself
    (in `pick_country()`, now removed) — cheap per call on its own, but each attempt
    already pays for spawning a fresh subprocess and DuckDB connection (see
    `attempt_with_hard_timeout`), so across a `ROWS=70000` run that's thousands of
    redundant queries (plus a sort) for a result set that never changes. Fetching
    once and picking with `random.choice()` in Python moves the randomness out of
    SQL entirely — same distribution, one query total instead of one per attempt.
    """
    return con.execute("SELECT id, name FROM division_areas WHERE subtype = 'country'").fetchall()


# ── Template 1: Containment ─────────────────────────────────────────────────
def sample_containment(con, country_id: str) -> dict | None:
    bbox = BBOX_SQL.format(geom="da.geometry", dist_m=0)
    row = con.execute(
        f"""
        WITH country AS (SELECT geometry FROM division_areas WHERE id = ?),
        place AS (
            SELECT da.id, da.name, da.geometry, {bbox} AS box
            FROM division_areas da, country
            WHERE da.subtype = 'locality'
              AND da.name IS NOT NULL
              AND ST_Within(da.geometry, country.geometry)
            ORDER BY random() LIMIT 1
        )
        SELECT place.name, e.subtype, e.class
        FROM place, infrastructures e
        WHERE ST_Intersects(e.geometry, place.box)
          AND ST_Within(e.geometry, place.geometry)
        LIMIT 1
        """,
        [country_id],
    ).fetchone()
    if row is None:
        return None
    place, subtype, cls = row
    return {"template": "containment", "place": place, "subtype": subtype, "class": cls}


# ── Template 2: Generic proximity (reference = divisions, a point) ─────────
def sample_proximity(con, country_id: str) -> dict | None:
    bbox = BBOX_SQL.format(geom="d.geometry", dist_m=5000)
    row = con.execute(
        f"""
        WITH country AS (SELECT geometry FROM division_areas WHERE id = ?),
        ref AS (
            SELECT d.id, d.name, d.geometry, {bbox} AS box
            FROM divisions d, country
            WHERE d.subtype IN ('locality', 'neighborhood')
              AND d.name IS NOT NULL
              AND ST_Within(d.geometry, country.geometry)
            ORDER BY random() LIMIT 1
        )
        SELECT ref.name, e.subtype, e.class,
               ST_Distance_Spheroid(
                   ST_Point2D(ST_X(ref.geometry), ST_Y(ref.geometry)),
                   ST_Point2D(ST_X(ST_Centroid(e.geometry)), ST_Y(ST_Centroid(e.geometry)))
               ) AS actual_distance_m
        FROM ref, infrastructures e
        WHERE ST_Intersects(e.geometry, ref.box)
          AND ST_DWithin_Spheroid(
                  ST_Point2D(ST_X(ref.geometry), ST_Y(ref.geometry)),
                  ST_Point2D(ST_X(ST_Centroid(e.geometry)), ST_Y(ST_Centroid(e.geometry))),
                  5000)
        LIMIT 1
        """,
        [country_id],
    ).fetchone()
    if row is None:
        return None
    place, subtype, cls, dist = row
    return {
        "template": "proximity",
        "place": place,
        "subtype": subtype,
        "class": cls,
        "distance_m": 5000,
        "actual_distance_m": round(dist, 1),
    }


# ── Template 3: Along (reference = water, a linestring) ────────────────────
def sample_along(con, country_id: str) -> dict | None:
    bbox = BBOX_SQL.format(geom="line", dist_m=500)
    row = con.execute(
        f"""
        WITH country AS (SELECT geometry FROM division_areas WHERE id = ?),
        segment AS (
            SELECT w.name, w.geometry AS line
            FROM water w, country
            WHERE w.class IN ('river', 'stream', 'canal')
              AND w.name IS NOT NULL
              AND ST_GeometryType(w.geometry) = 'LINESTRING'
              AND ST_Intersects(w.geometry, country.geometry)
              AND ST_Length_Spheroid(w.geometry) < 50000
            ORDER BY random() LIMIT 1
        ),
        bbox AS (
            SELECT name, line, {bbox} AS box
            FROM segment
        )
        SELECT bbox.name, e.subtype, e.class
        FROM bbox, infrastructures e
        WHERE ST_Intersects(e.geometry, bbox.box)
          AND ST_DWithin_Spheroid(
                  ST_Point2D(ST_X(ST_Centroid(e.geometry)), ST_Y(ST_Centroid(e.geometry))),
                  ST_Point2D(
                      ST_X(ST_LineInterpolatePoint(bbox.line, ST_LineLocatePoint(bbox.line, ST_Centroid(e.geometry)))),
                      ST_Y(ST_LineInterpolatePoint(bbox.line, ST_LineLocatePoint(bbox.line, ST_Centroid(e.geometry))))
                  ),
                  500)
        LIMIT 1
        """,
        [country_id],
    ).fetchone()
    if row is None:
        return None
    feature, subtype, cls = row
    return {"template": "along", "feature": feature, "subtype": subtype, "class": cls, "corridor_m": 500}


# ── Template 4: Center of (reference = division_areas, radius from area) ───
def sample_center(con, country_id: str) -> dict | None:
    # radius_m is a computed column, not a python literal, so the bbox expression
    # references it by name rather than substituting a number (see BBOX_SQL).
    bbox = BBOX_SQL.format(geom="geometry", dist_m="radius_m")
    row = con.execute(
        f"""
        WITH country AS (SELECT geometry FROM division_areas WHERE id = ?),
        place AS (
            SELECT da.name, da.geometry,
                   GREATEST(300, LEAST(5000, 0.15 * SQRT(ST_Area_Spheroid(da.geometry)))) AS radius_m
            FROM division_areas da, country
            WHERE da.subtype = 'locality'
              AND da.name IS NOT NULL
              AND ST_Within(da.geometry, country.geometry)
            ORDER BY random() LIMIT 1
        ),
        bbox AS (
            SELECT name, geometry, radius_m, {bbox} AS box
            FROM place
        )
        SELECT bbox.name, e.subtype, e.class, bbox.radius_m
        FROM bbox, infrastructures e
        WHERE ST_Intersects(e.geometry, bbox.box)
          AND ST_DWithin_Spheroid(
                  ST_Point2D(ST_X(ST_Centroid(e.geometry)), ST_Y(ST_Centroid(e.geometry))),
                  ST_Point2D(ST_X(ST_Centroid(bbox.geometry)), ST_Y(ST_Centroid(bbox.geometry))),
                  bbox.radius_m)
        LIMIT 1
        """,
        [country_id],
    ).fetchone()
    if row is None:
        return None
    place, subtype, cls, radius_m = row
    return {"template": "center", "place": place, "subtype": subtype, "class": cls, "center_radius_m": round(radius_m, 1)}


# ── Template 5: Periphery (reference = division_areas boundary) ────────────
def sample_periphery(con, country_id: str) -> dict | None:
    bbox = BBOX_SQL.format(geom="da.geometry", dist_m=500)
    row = con.execute(
        f"""
        WITH country AS (SELECT geometry FROM division_areas WHERE id = ?),
        place AS (
            SELECT da.name, da.geometry, {bbox} AS box
            FROM division_areas da, country
            WHERE da.subtype = 'locality'
              AND da.name IS NOT NULL
              AND ST_Within(da.geometry, country.geometry)
            ORDER BY random() LIMIT 1
        )
        SELECT place.name, e.subtype, e.class
        FROM place, infrastructures e
        WHERE ST_Intersects(e.geometry, place.box)
          AND ST_Within(e.geometry, place.geometry)
          AND ST_DWithin_Spheroid(
                  ST_Point2D(ST_X(ST_Centroid(e.geometry)), ST_Y(ST_Centroid(e.geometry))),
                  ST_Point2D(
                      ST_X(ST_ClosestPoint(ST_Boundary(place.geometry), e.geometry)),
                      ST_Y(ST_ClosestPoint(ST_Boundary(place.geometry), e.geometry))
                  ),
                  500)
        LIMIT 1
        """,
        [country_id],
    ).fetchone()
    if row is None:
        return None
    place, subtype, cls = row
    return {"template": "periphery", "place": place, "subtype": subtype, "class": cls, "band_m": 500}


# ── Templates 6/7: Cardinal direction (reference = divisions, a point) ─────
# degrees(ST_Azimuth(...)) returns [0, 360), never negative (verified: bearing due
# north/east/south/west from a point returns exactly 0.0/90.0/180.0/270.0) — see
# TEMPLATES.md "Distance and angle conventions" for the earlier, broken [-180,180)
# assumption this replaced. North straddles the 0/360 wraparound, so it needs an
# OR of its two halves rather than a plain (lo, hi) range like the other three.
DIRECTIONS = [("east", 45, 135), ("south", 135, 225), ("west", 225, 315)]


def classify_bearing(bearing: float) -> tuple[str, float, float]:
    if bearing >= 315 or bearing < 45:
        return "north", 315, 45  # wraps: matches >= 315 OR < 45, not a plain BETWEEN
    return next((d, lo, hi) for d, lo, hi in DIRECTIONS if lo <= bearing < hi)


def sample_direction(con, country_id: str, with_distance: bool) -> dict | None:
    bbox = BBOX_SQL.format(geom="d.geometry", dist_m=100000)
    row = con.execute(
        f"""
        WITH country AS (SELECT geometry FROM division_areas WHERE id = ?),
        ref AS (
            SELECT d.name, d.geometry, {bbox} AS box
            FROM divisions d, country
            WHERE d.subtype IN ('locality', 'neighborhood')
              AND d.name IS NOT NULL
              AND ST_Within(d.geometry, country.geometry)
            ORDER BY random() LIMIT 1
        )
        SELECT ref.name, e.subtype, e.class,
               degrees(ST_Azimuth(
                   ST_Point2D(ST_X(ref.geometry), ST_Y(ref.geometry)),
                   ST_Point2D(ST_X(ST_Centroid(e.geometry)), ST_Y(ST_Centroid(e.geometry)))
               )) AS bearing,
               ST_Distance_Spheroid(
                   ST_Point2D(ST_X(ref.geometry), ST_Y(ref.geometry)),
                   ST_Point2D(ST_X(ST_Centroid(e.geometry)), ST_Y(ST_Centroid(e.geometry)))
               ) AS actual_distance_m
        FROM ref, infrastructures e, country
        WHERE ST_Intersects(e.geometry, ref.box)
          AND ST_Within(e.geometry, country.geometry)
          AND ST_DWithin_Spheroid(
                  ST_Point2D(ST_X(ref.geometry), ST_Y(ref.geometry)),
                  ST_Point2D(ST_X(ST_Centroid(e.geometry)), ST_Y(ST_Centroid(e.geometry))),
                  100000)
        LIMIT 1
        """,
        [country_id],
    ).fetchone()
    if row is None:
        return None
    place, subtype, cls, bearing, dist = row
    direction, angle_min, angle_max = classify_bearing(bearing)
    result = {
        "template": "direction_distance" if with_distance else "direction",
        "place": place,
        "subtype": subtype,
        "class": cls,
        "direction": direction,
        "angle_min": angle_min,
        "angle_max": angle_max,
        "actual_bearing": round(bearing, 1),
    }
    if with_distance:
        result["distance_m"] = round(dist, 1)
    return result


# ── Template 8: Area-based distance (reference = water, any geometry type) ─
def sample_area_distance(con, country_id: str) -> dict | None:
    bbox = BBOX_SQL.format(geom="line", dist_m=5000)
    row = con.execute(
        f"""
        WITH country AS (SELECT geometry FROM division_areas WHERE id = ?),
        ref AS (
            SELECT w.name, w.geometry AS line
            FROM water w, country
            WHERE w.class IN ('lake', 'reservoir')
              AND w.name IS NOT NULL
              AND ST_Intersects(w.geometry, country.geometry)
            ORDER BY random() LIMIT 1
        ),
        bbox AS (
            SELECT name, line, {bbox} AS box
            FROM ref
        )
        SELECT bbox.name, e.subtype, e.class
        FROM bbox, infrastructures e
        WHERE ST_Intersects(e.geometry, bbox.box)
          AND ST_DWithin_Spheroid(
                  ST_Point2D(ST_X(ST_Centroid(e.geometry)), ST_Y(ST_Centroid(e.geometry))),
                  ST_Point2D(
                      ST_X(ST_ClosestPoint(bbox.line, e.geometry)),
                      ST_Y(ST_ClosestPoint(bbox.line, e.geometry))
                  ),
                  5000)
        LIMIT 1
        """,
        [country_id],
    ).fetchone()
    if row is None:
        return None
    place, subtype, cls = row
    return {"template": "area_distance", "place": place, "subtype": subtype, "class": cls, "distance_m": 5000}


# ── Left bank / right bank ──────────────────────────────────────────────────
def sample_left_right_bank(con, country_id: str) -> dict | None:
    bbox = BBOX_SQL.format(geom="line", dist_m=3000)
    row = con.execute(
        f"""
        WITH country AS (SELECT geometry FROM division_areas WHERE id = ?),
        segment AS (
            SELECT w.name, w.geometry AS line
            FROM water w, country
            WHERE w.class IN ('river', 'stream', 'canal')
              AND w.name IS NOT NULL
              AND ST_GeometryType(w.geometry) = 'LINESTRING'
              AND ST_Intersects(w.geometry, country.geometry)
              AND ST_Length_Spheroid(w.geometry) < 50000
            ORDER BY random() LIMIT 1
        ),
        bbox AS (
            SELECT name, line, {bbox} AS box
            FROM segment
        ),
        candidate AS (
            SELECT bbox.name, e.subtype, e.class, ST_Centroid(e.geometry) AS cgeom, bbox.line,
                   ST_LineLocatePoint(bbox.line, ST_Centroid(e.geometry)) AS f
            FROM bbox, infrastructures e
            WHERE ST_Intersects(e.geometry, bbox.box)
              AND ST_DWithin_Spheroid(
                      ST_Point2D(ST_X(ST_Centroid(e.geometry)), ST_Y(ST_Centroid(e.geometry))),
                      ST_Point2D(
                          ST_X(ST_LineInterpolatePoint(bbox.line, ST_LineLocatePoint(bbox.line, ST_Centroid(e.geometry)))),
                          ST_Y(ST_LineInterpolatePoint(bbox.line, ST_LineLocatePoint(bbox.line, ST_Centroid(e.geometry))))
                      ),
                      3000)
            LIMIT 1
        )
        SELECT name, subtype, class,
               sign(
                   (ST_X(ST_LineInterpolatePoint(line, LEAST(f + 0.0001, 1))) - ST_X(ST_LineInterpolatePoint(line, GREATEST(f - 0.0001, 0))))
                 * (ST_Y(cgeom) - ST_Y(ST_LineInterpolatePoint(line, f)))
                 - (ST_Y(ST_LineInterpolatePoint(line, LEAST(f + 0.0001, 1))) - ST_Y(ST_LineInterpolatePoint(line, GREATEST(f - 0.0001, 0))))
                 * (ST_X(cgeom) - ST_X(ST_LineInterpolatePoint(line, f)))
               ) AS side_sign
        FROM candidate
        """,
        [country_id],
    ).fetchone()
    if row is None:
        return None
    feature, subtype, cls, side_sign = row
    return {
        "template": "left_right_bank",
        "feature": feature,
        "subtype": subtype,
        "class": cls,
        "distance_m": 3000,
        "side_sign": int(side_sign),
        "side": "left" if side_sign == 1 else "right",
    }


# ── Template 10: Bordering a place (draft, not yet tested — see TEMPLATES.md) ──
def sample_bordering(con, country_id: str) -> dict | None:
    bbox = BBOX_SQL.format(geom="area", dist_m=20000)
    row = con.execute(
        f"""
        WITH ref AS (SELECT name, geometry FROM division_areas WHERE id = ?),
        bbox AS (
            SELECT name, geometry AS area, {bbox} AS box
            FROM ref
        )
        SELECT bbox.name, d.name
        FROM bbox, divisions d
        WHERE d.subtype = 'locality'
          AND ST_Intersects(d.geometry, bbox.box)
          AND ST_DWithin_Spheroid(
                  ST_Point2D(ST_X(d.geometry), ST_Y(d.geometry)),
                  ST_Point2D(
                      ST_X(ST_ClosestPoint(ST_Boundary(bbox.area), d.geometry)),
                      ST_Y(ST_ClosestPoint(ST_Boundary(bbox.area), d.geometry))
                  ),
                  20000)
        ORDER BY random() LIMIT 1
        """,
        [country_id],
    ).fetchone()
    if row is None:
        return None
    place_name, city_name = row
    return {"template": "bordering", "place": place_name, "city": city_name, "distance_m": 20000}


# ── Template 12: Cities in a direction (candidate = divisions locality, not
# infrastructures — see TEMPLATES.md "12. Cities in a direction") ──────────
def sample_city_direction(con, country_id: str, with_distance: bool) -> dict | None:
    bbox = BBOX_SQL.format(geom="d.geometry", dist_m=100000)
    row = con.execute(
        f"""
        WITH country AS (SELECT geometry FROM division_areas WHERE id = ?),
        ref AS (
            SELECT d.name, d.geometry, {bbox} AS box
            FROM divisions d, country
            WHERE d.subtype IN ('locality', 'neighborhood')
              AND d.name IS NOT NULL
              AND ST_Within(d.geometry, country.geometry)
            ORDER BY random() LIMIT 1
        )
        SELECT ref.name,
               degrees(ST_Azimuth(
                   ST_Point2D(ST_X(ref.geometry), ST_Y(ref.geometry)),
                   ST_Point2D(ST_X(e.geometry), ST_Y(e.geometry))
               )) AS bearing,
               ST_Distance_Spheroid(
                   ST_Point2D(ST_X(ref.geometry), ST_Y(ref.geometry)),
                   ST_Point2D(ST_X(e.geometry), ST_Y(e.geometry))
               ) AS actual_distance_m
        FROM ref, divisions e
        WHERE e.subtype = 'locality'
          AND ST_Intersects(e.geometry, ref.box)
          -- excludes ref matching itself (or a near-duplicate point) as its own result
          AND ST_Distance_Spheroid(
                ST_Point2D(ST_X(ref.geometry), ST_Y(ref.geometry)),
                ST_Point2D(ST_X(e.geometry), ST_Y(e.geometry))
              ) BETWEEN 500 AND 100000
        LIMIT 1
        """,
        [country_id],
    ).fetchone()
    if row is None:
        return None
    place, bearing, dist = row
    direction, angle_min, angle_max = classify_bearing(bearing)
    result = {
        "template": "city_direction_distance" if with_distance else "city_direction",
        "place": place,
        "direction": direction,
        "angle_min": angle_min,
        "angle_max": angle_max,
        "actual_bearing": round(bearing, 1),
    }
    if with_distance:
        result["distance_m"] = round(dist, 1)
    return result


# ── Template 11: Show a division (no infrastructures join — just resolves a name) ──
def sample_show_division(con, country_id: str) -> dict | None:
    """Picks either the country itself (~30% of the time) or a random locality
    inside it, so training examples cover both "show me France" and "show me Lyon"
    phrasings (TEMPLATES.md template #11). No infrastructures/water join needed —
    this template only resolves a name to divisions/division_areas geometry."""
    if random.random() < 0.3:
        row = con.execute(
            "SELECT name FROM divisions WHERE id = (SELECT division_id FROM division_areas WHERE id = ?)",
            [country_id],
        ).fetchone()
    else:
        row = con.execute(
            """
            WITH country AS (SELECT geometry FROM division_areas WHERE id = ?)
            SELECT d.name
            FROM divisions d, country
            WHERE d.subtype = 'locality'
              AND d.name IS NOT NULL
              AND ST_Within(d.geometry, country.geometry)
            ORDER BY random() LIMIT 1
            """,
            [country_id],
        ).fetchone()
    if row is None:
        return None
    return {"template": "show_division", "place": row[0]}


# ── Template 9: Places by category (requires the `places` view — see
# scripts/init_places.sql, not yet populated; these samplers will error until
# `make download-overture-places` + init_places.sql have been run) ─────────
PLACE_CATEGORIES = ["restaurant", "lodging", "hotel", "school", "hospital", "shopping_mall", "grocery_store"]


def sample_places_containment(con, country_id: str) -> dict | None:
    category = random.choice(PLACE_CATEGORIES)
    row = con.execute(
        """
        WITH country AS (SELECT geometry FROM division_areas WHERE id = ?),
        place AS (
            SELECT da.name, da.geometry
            FROM division_areas da, country
            WHERE da.subtype = 'locality'
              AND da.name IS NOT NULL
              AND ST_Within(da.geometry, country.geometry)
            ORDER BY random() LIMIT 1
        )
        SELECT place.name
        FROM place, places p
        WHERE list_contains(p.category_hierarchy, ?)
          AND ST_Within(p.geometry, place.geometry)
        LIMIT 1
        """,
        [country_id, category],
    ).fetchone()
    if row is None:
        return None
    return {"template": "places_containment", "place": row[0], "category": category}


def sample_places_proximity(con, country_id: str) -> dict | None:
    category = random.choice(PLACE_CATEGORIES)
    row = con.execute(
        """
        WITH country AS (SELECT geometry FROM division_areas WHERE id = ?),
        ref AS (
            SELECT d.name, d.geometry
            FROM divisions d, country
            WHERE d.subtype IN ('locality', 'neighborhood')
              AND d.name IS NOT NULL
              AND ST_Within(d.geometry, country.geometry)
            ORDER BY random() LIMIT 1
        )
        SELECT ref.name
        FROM ref, places p
        WHERE list_contains(p.category_hierarchy, ?)
          AND ST_DWithin_Spheroid(
                ST_Point2D(ST_X(p.geometry), ST_Y(p.geometry)),
                ST_Point2D(ST_X(ref.geometry), ST_Y(ref.geometry)),
                5000)
        LIMIT 1
        """,
        [country_id, category],
    ).fetchone()
    if row is None:
        return None
    return {"template": "places_proximity", "place": row[0], "category": category, "distance_m": 5000}


SAMPLERS = {
    "containment": lambda con, cid: sample_containment(con, cid),
    "proximity": lambda con, cid: sample_proximity(con, cid),
    "along": lambda con, cid: sample_along(con, cid),
    "center": lambda con, cid: sample_center(con, cid),
    "periphery": lambda con, cid: sample_periphery(con, cid),
    "direction": lambda con, cid: sample_direction(con, cid, with_distance=False),
    "direction_distance": lambda con, cid: sample_direction(con, cid, with_distance=True),
    "area_distance": lambda con, cid: sample_area_distance(con, cid),
    "left_right_bank": lambda con, cid: sample_left_right_bank(con, cid),
    "places_containment": lambda con, cid: sample_places_containment(con, cid),
    "places_proximity": lambda con, cid: sample_places_proximity(con, cid),
    "bordering": lambda con, cid: sample_bordering(con, cid),
    "show_division": lambda con, cid: sample_show_division(con, cid),
    "city_direction": lambda con, cid: sample_city_direction(con, cid, with_distance=False),
    "city_direction_distance": lambda con, cid: sample_city_direction(con, cid, with_distance=True),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--template", choices=[*SAMPLERS, "all"], default="all")
    parser.add_argument("--rows", type=int, default=100, help="total sample rows to produce, split evenly across templates")
    parser.add_argument("--out", default="data/samples/entities.jsonl", help="output JSONL file")
    parser.add_argument(
        "--max-attempts-factor",
        type=int,
        default=10,
        help="give up on a template after target_rows * this many failed attempts",
    )
    args = parser.parse_args()

    templates = list(SAMPLERS) if args.template == "all" else [args.template]
    target_per_template = max(1, args.rows // len(templates))

    # Fail fast: attempt_with_hard_timeout() counts any worker error as a miss, so a
    # connection problem that can never resolve itself (DB locked by the `duckdb`
    # container, missing extension, ...) would otherwise spin up thousands of doomed
    # attempts. Checked before opening --out so an existing output isn't truncated.
    # Also fetches the country list once here (see fetch_countries()) rather than
    # opening a second connection just for that.
    try:
        con = connect()
        countries = fetch_countries(con)
        con.close()
    except duckdb.Error as e:
        sys.exit(f"# cannot open {DB_PATH}: {e}\n# (DB locked? stop the container: docker compose stop duckdb)")
    if not countries:
        sys.exit("# no country found in division_areas")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_hits = 0
    with out_path.open("w", encoding="utf-8") as f:
        for name in templates:
            hits, attempts = 0, 0
            max_attempts = target_per_template * args.max_attempts_factor
            while hits < target_per_template and attempts < max_attempts:
                attempts += 1
                result = attempt_with_hard_timeout(name, countries)
                if result is None:
                    continue
                hits += 1
                total_hits += 1
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"# {name}: {hits}/{target_per_template} sampled ({attempts} attempts)", file=sys.stderr)

    print(f"# total: {total_hits} rows written to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
