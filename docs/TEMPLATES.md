# TEMPLATES.md — SQL template catalog (STEP 3)

This document is the STEP 3 deliverable from DESIGN.md: parameterized SQL templates covering the spatial relations the fine-tuned model must learn to generate. Templates are written against the logical views defined in `SCHEMA.md` (`divisions`, `division_areas`, `infrastructures`, `water`) — never against raw Overture columns.

These templates are filled with real entities, sampled directly via live spatial joins on the logical views (STEP 2, the relations graph, was skipped — see DESIGN.md), and executed on DuckDB to produce validated `(question, sql)` pairs for STEP 4. A template that returns an empty result for a given entity is discarded (DESIGN.md STEP 4).

## Confirmed available spatial functions (DuckDB `spatial` extension)

Verified directly against the running extension:

| Function | Behavior | Used for |
|---|---|---|
| `ST_DWithin_Spheroid(a, b, distance_m)` | True meter-accurate distance check on WGS84 geometries (accounts for earth curvature) — no reprojection needed | All proximity-based templates |
| `ST_Distance_Spheroid(a, b)` | Meter-accurate distance value | Directional-distance templates (range check) |
| `ST_Azimuth(a, b)` | Bearing in radians from `a` to `b`, `0 = north`, clockwise (same convention as PostGIS) | Cardinal-direction templates |
| `ST_Within` / `ST_Contains` | Standard containment predicates | Containment templates |
| `ST_Centroid`, `ST_Boundary` | Derived geometries | Center-of / periphery templates |

No meter-accurate `ST_Buffer` is available (only a planar/degree-based buffer). Templates therefore avoid materializing buffer polygons and use `ST_DWithin_Spheroid` directly for every distance-based filter — functionally equivalent for filtering purposes, and simpler.

### ⚠️ Three constraints found by testing against the real dataset (`data/db/llaici.duckdb`), not visible from the docs alone

The table above lists what the functions *do*; it doesn't say what they *accept*. Running the templates against real data (not synthetic examples) surfaced three problems that would otherwise only show up once the fine-tuned model starts generating broken SQL:

1. **`ST_DWithin_Spheroid` / `ST_Distance_Spheroid` only accept `POINT_2D`, never `GEOMETRY`.**
   Confirmed via `duckdb_functions()`: no `(GEOMETRY, GEOMETRY, DOUBLE)` overload exists, only `(POINT_2D, POINT_2D, DOUBLE)`. But every logical view column (`e.geometry`, `ref.geometry`, …) is typed `GEOMETRY('OGC:CRS84')`, not `POINT_2D`. Running template #2 as originally written against the real views throws `Binder Error: No function matches...` — reproduced directly, not theoretical.
   **Fix**: convert through `ST_Point2D(ST_X(g), ST_Y(g))` before calling any `*_Spheroid` function.

2. **`infrastructures.geometry` is not always a point** (and `water.geometry` is not always a line).
   Checked the real distribution: `infrastructures` is 57% `POINT` but 43% `LINESTRING`/`POLYGON`/`MULTIPOLYGON` (bridges as linear features, parking areas as polygons…). `water` has the same problem *within a single name match* — e.g. matching `%Tevere%` returns 17 `LINESTRING` river segments, 2 `POLYGON` river-as-area segments, plus unrelated features with "Tevere" in the name (a fountain, a waterfall, a spring) picked up by the fuzzy `ILIKE` match. A function that requires a point (`ST_X`, `ST_LineLocatePoint`'s point argument) crashes on the non-point rows.
   **Fix**: reduce every candidate/reference geometry to a representative point with `ST_Centroid(g)` before using it as a point, and restrict `water` joins to the relevant `class` and `ST_GeometryType(geometry) = 'LINESTRING'` to exclude area-labelled segments and off-topic name matches.

3. **Unfiltered spatial joins against `water`/`division_areas` don't scale.**
   `infrastructures` has ~13M rows (1.6M bridges alone). A join against even a handful of matched river segments, with no geographic prefilter, computes `ST_Distance`/`ST_DWithin_Spheroid` (each expensive on multi-vertex lines) over tens of millions of pairs — reproduced an out-of-memory crash (19+ GiB) testing the left/right-bank template. A DuckDB `RTREE` index would normally fix this, but it's not an option here: DuckDB refuses indexes on **views** (`CREATE INDEX ... USING RTREE` → `Binder Error: can only create an index on a base table`), and the local views are deliberately swappable for a direct-Overture-on-S3 read (DESIGN.md "Logical view layer") which has no local table to index at all.
   **Fix**: a plain-SQL bounding-box prefilter, cheap and portable to both backends — see "Bounding-box prefilter" below. Confirmed: without it, the left/right-bank query OOM's; with it, the same query runs in <1s.

### Bounding-box prefilter (used by templates #3, #8, and left/right bank)

Narrows candidates with a cheap `ST_Intersects` on an expanded bounding box **before** the expensive exact distance/geometry test — the standard pattern to avoid an unindexed spatial join scanning millions of rows.

```sql
-- {distance_m} converted to a degree delta: geometries are WGS84 (lon/lat degrees),
-- so meters must be converted. 1° latitude ≈ 111,320 m; 1° longitude shrinks by cos(latitude).
-- We deliberately use the longitude conversion (the larger degree delta of the two) for
-- both axes: safe to over-include here, since the exact ST_DWithin_Spheroid check afterward
-- catches anything the bbox let through by mistake.
WITH bbox AS (
    SELECT ST_Expand(
             ST_Envelope(ref.geometry),
             {distance_m} / (111320.0 * cos(radians(
               (ST_YMin(ST_Envelope(ref.geometry)) + ST_YMax(ST_Envelope(ref.geometry))) / 2
             )))
           ) AS box
    FROM ref
)
-- then: ... JOIN infrastructures e ON ST_Intersects(e.geometry, bbox.box) ...
-- followed by the exact ST_DWithin_Spheroid check on the (now small) candidate set.
```

Not needed for templates joining only against `divisions` (always a point — no multi-vertex geometry to make the exact check expensive) or when the name filter already narrows the join to one specific row (e.g. `division_areas` for a single named city in template #1/#4/#5, tested fine without it).

## Name matching: case- and accent-insensitive

The templates below show name filters as `name ILIKE '%{place}%'` for readability. The SQL actually generated (`02_fill_and_validate.py`, `name_match()`) wraps both sides in `strip_accents`:

```sql
strip_accents(da.name) ILIKE strip_accents('%{place}%') OR strip_accents(da.name_fr) ILIKE strip_accents('%{place}%') ...
```

`ILIKE` already ignores case; `strip_accents` makes "Seville" match "Séville" and "cordoba" match "Córdoba", since users often type without accents (step 03 generates such questions). The term stays exactly as typed in the question, so the model only copies it and never has to strip accents itself. Cost: longer SQL (~+50 tokens per name match), and `strip_accents` evaluated on every scanned name — not benchmarked.

## Distance and angle conventions

- All user-facing distances ("5km", "500m", "5 miles") must be normalized to meters before being substituted into a template (DESIGN.md STEP 3).
- Cardinal direction sectors (90° wide, centered on the cardinal bearing):

⚠️ **Corrected after testing**: `degrees(ST_Azimuth(...))` returns **0 to 360**, never negative — verified directly (`ST_Azimuth` from a point to another point due north/east/south/west returns exactly `0.0` / `90.0` / `180.0` / `270.0`). The `-45 to 45` / `-135 to -45` ranges below (an earlier version of this table) don't work as a plain `BETWEEN`: `North` between `-45 AND 45` only ever matches `0–45` (there are no negative azimuth values to match the `-45–0` half), silently dropping the `315–360` half of the north cone; `West` as `-135 to -45` matches nothing at all. Templates #6/#7 below now branch `North` into an `OR` of its two wrapping halves instead of a single `BETWEEN`; East/South/West stay a single `BETWEEN` in the `[0, 360)` range.

| Direction | Azimuth range (degrees, `[0, 360)`) |
|---|---|
| North | `>= 315 OR < 45` |
| East | 45 to 135 |
| South | 135 to 225 |
| West | 225 to 315 |

- Default radii when the user doesn't specify a distance:
  - Generic proximity ("near", "around"): **5000 m**
  - "Along" corridor: **500 m**
  - Directional-distance tolerance band: **±20%** of the requested distance (few entities sit at the exact requested distance)

---

## 1. Containment — "in / within a place"

**NL examples**: "restaurants in Madrid" / "restaurants à Madrid"

```sql
SELECT e.id, e.name, e.geometry
FROM infrastructures e
JOIN division_areas da ON ST_Within(e.geometry, da.geometry)
WHERE e.subtype = '{subtype}' AND e.class = '{class}'
  AND (da.name ILIKE '%{place}%' OR da.name_fr ILIKE '%{place}%' OR da.name_es ILIKE '%{place}%'
       OR da.name_it ILIKE '%{place}%' OR da.name_de ILIKE '%{place}%' OR da.name_en ILIKE '%{place}%'
       OR da.name_pt ILIKE '%{place}%');
```

Params: `{subtype}`, `{class}` (any category from `infrastructures`, or swap the base table for `water`/`divisions`), `{place}`.
Uses `division_areas` (polygon), not `divisions` (point-only) — containment requires a boundary.

---

## 2. Generic proximity — "near / close to"

**NL examples**: "bus stop near Notre-Dame" / "arrêt de bus près de Notre-Dame"

```sql
SELECT e.id, e.name, e.geometry
FROM infrastructures e, divisions ref
WHERE e.subtype = '{subtype}' AND e.class = '{class}'
  AND (ref.name ILIKE '%{place}%' OR ref.name_fr ILIKE '%{place}%' /* ...other name_* columns */)
  AND ST_DWithin_Spheroid(
        ST_Point2D(ST_X(ST_Centroid(e.geometry)), ST_Y(ST_Centroid(e.geometry))),
        ST_Point2D(ST_X(ST_ClosestPoint(ref.geometry, e.geometry)), ST_Y(ST_ClosestPoint(ref.geometry, e.geometry))),
        {distance_m});
```

Params: `{subtype}`, `{class}`, `{place}`, `{distance_m}` (default 5000 if unspecified).
Reference (`ref`) can be `divisions`, `division_areas`, or `water` depending on what the user names.

⚠️ `ST_Point2D(ST_X(...), ST_Y(...))` around both operands is required: `ST_DWithin_Spheroid` only accepts `POINT_2D`, not the `GEOMETRY` type these views expose (see "Confirmed available spatial functions" above; template fails to bind without it). `ST_Centroid(e.geometry)` handles the ~43% of `infrastructures` rows that aren't points. `ST_ClosestPoint(ref.geometry, e.geometry)` — rather than `ST_Centroid(ref.geometry)` — matters when `ref` is `division_areas`/`water`: "near" should mean near the *feature's edge*, not near its geometric center (a big country's centroid can be hundreds of km from its border). Tested against `divisions` (point ref, small cardinality) without a bbox prefilter — fine; add one (see above) if `ref` is `division_areas`/`water` and the name match isn't already narrow.

---

## 3. Along — "along a linear feature"

**NL examples**: "hotels along the Seine" / "hôtels le long de la Seine"

```sql
WITH segments AS (
    SELECT geometry AS line
    FROM water
    WHERE class IN ('river', 'stream', 'canal')
      AND ST_GeometryType(geometry) = 'LINESTRING'
      AND (name ILIKE '%{feature}%' OR name_fr ILIKE '%{feature}%' /* ...other name_* columns */)
),
bbox AS (
    SELECT line,
           ST_Expand(
             ST_Envelope(line),
             {corridor_m} / (111320.0 * cos(radians(
               (ST_YMin(ST_Envelope(line)) + ST_YMax(ST_Envelope(line))) / 2
             )))
           ) AS box
    FROM segments
),
candidates AS (
    SELECT e.id, e.name, e.geometry, ST_Centroid(e.geometry) AS cgeom, b.line,
           row_number() OVER (PARTITION BY e.id ORDER BY ST_Distance(b.line, ST_Centroid(e.geometry))) AS rn
    FROM infrastructures e, bbox b
    WHERE e.subtype = '{subtype}' AND e.class = '{class}'
      AND ST_Intersects(e.geometry, b.box)
)
SELECT id, name, geometry
FROM candidates
WHERE rn = 1
  AND ST_DWithin_Spheroid(
        ST_Point2D(ST_X(cgeom), ST_Y(cgeom)),
        ST_Point2D(ST_X(ST_LineInterpolatePoint(line, ST_LineLocatePoint(line, cgeom))), ST_Y(ST_LineInterpolatePoint(line, ST_LineLocatePoint(line, cgeom)))),
        {corridor_m});
```

Params: `{subtype}`, `{class}`, `{feature}`, `{corridor_m}` (default 500).

⚠️ Longer than the original one-liner, for reasons confirmed against real data (see "Confirmed available spatial functions" above):
- `ST_DWithin_Spheroid` cannot take `w.geometry` (a line) directly — it only accepts `POINT_2D`. The original claim that it "accepts a line geometry as the reference directly" doesn't hold: verified, it throws a `Binder Error`.
- A name match on `water` can return several segments for one river (`%Tevere%` matched 17 `LINESTRING` + 2 `POLYGON` "river-as-area" rows, plus unrelated features like a fountain whose name happens to contain the search string) — hence the `class`/`ST_GeometryType` filter and picking the *nearest* segment per candidate (`row_number() ... rn = 1`) rather than an arbitrary one.
- The `bbox` CTE is the prefilter described above — without it, this query OOM'd (19+ GiB) joining ~1.6M candidate rows against even a handful of river segments; with it, the same query ran in well under a second.

---

## 4. Center of — "in the center of a place"

**NL examples**: "in the center of Lyon" / "dans le centre de Lyon"

```sql
SELECT e.id, e.name, e.geometry
FROM infrastructures e
JOIN division_areas da ON (da.name ILIKE '%{place}%' OR da.name_fr ILIKE '%{place}%' /* ... */)
WHERE e.subtype = '{subtype}' AND e.class = '{class}'
  AND ST_DWithin_Spheroid(
        ST_Point2D(ST_X(ST_Centroid(e.geometry)), ST_Y(ST_Centroid(e.geometry))),
        ST_Point2D(ST_X(ST_Centroid(da.geometry)), ST_Y(ST_Centroid(da.geometry))),
        GREATEST(300, LEAST(5000, 0.15 * SQRT(ST_Area_Spheroid(da.geometry)))));
```

Params: `{subtype}`, `{class}`, `{place}`.

⚠️ Same `ST_Point2D`/`ST_Centroid` wrapping as template #2, required for the same reason (`ST_DWithin_Spheroid` needs `POINT_2D`, `e.geometry` isn't always a point). Tested against real data (Paris, `bus_stop`) — works, and stays fast without a bbox prefilter since the join is already narrowed to one named area via `division_areas`.

**Calibration resolved**: `{center_radius_m}` is no longer a filled-in parameter — it's computed inline from `da`'s own area, so no separate calibration step is needed at STEP 4 generation time. Formula: `radius = clamp(0.15 * sqrt(area_m²), 300m, 5000m)`. Derivation: a circle of that radius covers roughly `0.15² * π ≈ 7%` of the area's total surface — a reasonable "center zone" fraction regardless of whether the area is a hamlet or a city. The `[300m, 5000m]` clamp keeps it sane at both extremes (a tiny hamlet's "center" isn't a few meters wide; a large metro area's "center" isn't tens of km wide). Verified against real `division_areas`: Lyon (`locality`) → 1238m, Lyon (`localadmin`, the larger metro polygon) → capped at 5000m, Paris → 1890m — all plausible "city center" radii.

---

## 5. Periphery / edge / border — "on the outskirts / at the edge of a place"

**NL examples**: "on the outskirts of Berlin" / "en périphérie de Berlin" / "à la limite de Berlin" / "en bordure de Berlin"

```sql
SELECT e.id, e.name, e.geometry
FROM infrastructures e
JOIN division_areas da ON (da.name ILIKE '%{place}%' OR da.name_fr ILIKE '%{place}%' /* ... */)
WHERE e.subtype = '{subtype}' AND e.class = '{class}'
  AND ST_Within(e.geometry, da.geometry)
  AND ST_DWithin_Spheroid(
        ST_Point2D(ST_X(ST_Centroid(e.geometry)), ST_Y(ST_Centroid(e.geometry))),
        ST_Point2D(ST_X(ST_ClosestPoint(ST_Boundary(da.geometry), e.geometry)), ST_Y(ST_ClosestPoint(ST_Boundary(da.geometry), e.geometry))),
        {band_m});
```

Params: `{subtype}`, `{class}`, `{place}`, `{band_m}` (proximity band to the boundary, inside the area).

⚠️ `ST_Boundary(da.geometry)` is a ring (`LINESTRING`), not a point — same `POINT_2D`-only constraint on `ST_DWithin_Spheroid` applies, so the nearest point on that boundary ring is taken first via `ST_ClosestPoint`, then converted. Tested against real data (Paris, `bus_stop`) — works, fast (join already narrowed to one named area).

---

## 6. Cardinal direction (no distance) — "north / south / east / west of a place"

**NL examples**: "north of Barcelona" / "au nord de Barcelone"

```sql
SELECT e.id, e.name, e.geometry
FROM infrastructures e, divisions ref
WHERE e.subtype = '{subtype}' AND e.class = '{class}'
  AND (ref.name ILIKE '%{place}%' OR ref.name_fr ILIKE '%{place}%' /* ... */)
  AND {angle_condition}
  AND ST_Distance_Spheroid(
        ST_Point2D(ST_X(ref.geometry), ST_Y(ref.geometry)),
        ST_Point2D(ST_X(ST_Centroid(e.geometry)), ST_Y(ST_Centroid(e.geometry)))
      ) <= 100000;
```

Params: `{subtype}`, `{class}`, `{place}`, `{angle_condition}` — one of, from the direction sector table above:
- North: `degrees(ST_Azimuth(ST_Point2D(ST_X(ref.geometry), ST_Y(ref.geometry)), ST_Point2D(ST_X(ST_Centroid(e.geometry)), ST_Y(ST_Centroid(e.geometry))))) >= 315 OR degrees(ST_Azimuth(ST_Point2D(ST_X(ref.geometry), ST_Y(ref.geometry)), ST_Point2D(ST_X(ST_Centroid(e.geometry)), ST_Y(ST_Centroid(e.geometry))))) < 45`
- East/South/West: `degrees(ST_Azimuth(ST_Point2D(ST_X(ref.geometry), ST_Y(ref.geometry)), ST_Point2D(ST_X(ST_Centroid(e.geometry)), ST_Y(ST_Centroid(e.geometry))))) BETWEEN {angle_min} AND {angle_max}` (45/135, 135/225, 225/315 respectively)

⚠️ `ST_Azimuth` does have a `(GEOMETRY, GEOMETRY)` overload (unlike the `*_Spheroid` functions), so this one doesn't strictly need the `POINT_2D` conversion to bind — but `e.geometry` still isn't always a point, so `ST_Centroid(e.geometry)` keeps the bearing well-defined (bearing to a bridge's centroid, not to an arbitrary vertex of its linestring). Wrapping `ref.geometry` in `ST_Point2D` too keeps both templates #6 and #7 using the same convention. Tested against real data (Barcelona, `bus_stop`) — works.

**Calibration resolved**: added a fixed **100km** default max-distance bound. Without any bound, "north of Barcelona" would technically match an entity in Norway. A join constraining `ref`/`e` to the same `division_areas` (same region/country) was considered but rejected as needless complexity here — a flat 100km cutoff is simpler, doesn't depend on country size (a query near a small country's border wouldn't behave oddly), and is generous enough to still read as "roughly in that direction" for the kind of city/landmark references this template targets (`divisions` reference points, not countries).

---

## 7. Cardinal direction + distance — "X km north/south/east/west of a place"

**NL examples**: "5km north of Barcelona" / "5km au nord de Barcelone"

```sql
SELECT e.id, e.name, e.geometry
FROM infrastructures e, divisions ref
WHERE e.subtype = '{subtype}' AND e.class = '{class}'
  AND (ref.name ILIKE '%{place}%' OR ref.name_fr ILIKE '%{place}%' /* ... */)
  AND {angle_condition}
  AND ST_Distance_Spheroid(
        ST_Point2D(ST_X(ref.geometry), ST_Y(ref.geometry)),
        ST_Point2D(ST_X(ST_Centroid(e.geometry)), ST_Y(ST_Centroid(e.geometry)))
      ) BETWEEN {distance_m} - GREATEST(200, LEAST(0.20 * {distance_m}, 10000))
            AND {distance_m} + GREATEST(200, LEAST(0.20 * {distance_m}, 10000));
```

Params: `{subtype}`, `{class}`, `{place}`, `{angle_condition}` (see template #6), `{distance_m}`.

⚠️ Same fixes as template #6, plus `ST_Distance_Spheroid` — like `ST_DWithin_Spheroid` — only has a `(POINT_2D, POINT_2D)` overload, so it needs the same `ST_Point2D` wrap or it fails to bind. Tested against real data (Barcelona, `bus_stop`, 4-6km band) — works.

**Found while implementing the STEP 4 sampler**: `degrees(ST_Azimuth(...))` returns `[0, 360)`, verified directly (bearing due north/east/south/west from a point returns exactly `0.0`/`90.0`/`180.0`/`270.0`). The original `{angle_min}`/`{angle_max}` params (`North: -45 to 45`, `West: -135 to -45`) assumed a signed `[-180, 180)` range and silently broke: `BETWEEN -45 AND 45` only ever matches `0–45` (there's no negative azimuth to match `-45–0`), and `BETWEEN -135 AND -45` matches nothing at all. Replaced with `{angle_condition}` — a plain `BETWEEN` for East/South/West (whose sectors don't wrap past 360/0), and an explicit `OR` of the two halves for North (the only sector that straddles the 0/360 wraparound).

**Calibration resolved**: tolerance is now `clamp(0.20 * distance_m, 200m, 10000m)` instead of a flat ±20%. A flat percentage is fine in the middle range but breaks at both ends: at `distance_m = 200m`, ±20% is only ±40m — tighter than realistic entity spacing, likely to discard otherwise-valid pairs for no good reason; at `distance_m = 200km`, ±20% is ±40km — wide enough that "200km north" barely means anything directional anymore. The floor (200m) and ceiling (10km) keep the band meaningful at both extremes while leaving the proportional ±20% behavior for the common case (a few km to a few tens of km) unchanged.

---

## 8. Area-based distance — "within X km around a place"

**NL examples**: "restaurants within 5km around Lake Sanabria" / "restaurants à moins de 5km du Lac de Sanabria"

```sql
WITH ref AS (
    SELECT geometry
    FROM water
    WHERE (name ILIKE '%{place}%' OR name_fr ILIKE '%{place}%' /* ... */)
    LIMIT 1
),
bbox AS (
    SELECT geometry AS line,
           ST_Expand(
             ST_Envelope(geometry),
             {distance_m} / (111320.0 * cos(radians(
               (ST_YMin(ST_Envelope(geometry)) + ST_YMax(ST_Envelope(geometry))) / 2
             )))
           ) AS box
    FROM ref
)
SELECT e.id, e.name, e.geometry
FROM infrastructures e, bbox
WHERE e.subtype = '{subtype}' AND e.class = '{class}'
  AND ST_Intersects(e.geometry, bbox.box)
  AND ST_DWithin_Spheroid(
        ST_Point2D(ST_X(ST_Centroid(e.geometry)), ST_Y(ST_Centroid(e.geometry))),
        ST_Point2D(ST_X(ST_ClosestPoint(bbox.line, e.geometry)), ST_Y(ST_ClosestPoint(bbox.line, e.geometry))),
        {distance_m});
```

Same underlying relation as template #2 (Generic proximity) — this is a deliberate duplication: STEP 4 should generate multiple natural-language phrasings ("near", "around", "within Xkm of") over the *same* underlying template, rather than treating them as separate SQL patterns.

⚠️ Kept as its own block rather than pointing back to template #2's SQL: `water` features (a lake, a long river) are far more likely than `divisions` to be large multi-vertex polygons/lines, so this version keeps the bbox prefilter and `ST_ClosestPoint` (see "Confirmed available spatial functions" above) that template #2 only needs when its `ref` is `division_areas`/`water`. Tested against real data (Lake Geneva, `bus_stop`, 3km) — works, fast.

---

## 9. Places by category — "restaurants / hotels / schools... in / near a place"

**NL examples**: "restaurants in Madrid" / "restaurants à Lisbonne", "hotels near the station" / "hôtels près de la gare", "hospitals in Seville", "shopping malls near the airport", "grocery stores in Casablanca"

The `places` view is defined in `00_init.sql` (`SCHEMA.md` §5), filtered to the categories below just like `infrastructures` is — with only `id`/`name`/`category`/`category_hierarchy`/`geometry` (no `confidence`/`operating_status`, dropped from the view; no multilingual `name_*`, never needed since these templates don't search `places` by name). Tested against real data (`data/db/llaici.duckdb`): both containment and proximity variants sampled and validated 3/3, e.g. `lodging` in "La Fuente de San Esteban", `hotel` in "Cabrela", `restaurant` near "Llaneces".

### Categories covered so far

Chosen as a first, likely-to-be-asked-about subset — not exhaustive (see `SCHEMA.md` §5 for the full top-level category list found in a test extract):

| `{category}` | `category_hierarchy` path | Note |
|---|---|---|
| `restaurant` | `food_and_drink → restaurant → (cuisine)` | Excludes fast food — see caveat below |
| `lodging` | `lodging → (hotel / hostel / bed_and_breakfast / resort / guest_house...)` | Broad umbrella; use `hotel` instead of `lodging` for "hotels" specifically — both were confirmed present as real `category_hierarchy` values |
| `school` | `education → place_of_learning → school → (elementary_school...)` | Same subtype trap as `restaurant`: `elementary_school` sits *under* `school` in the hierarchy, `college_university`/`library`/`language_school` do not |
| `hospital` | `health_care → hospital` | Flat — no subtypes found under `hospital` itself in the test extract |
| `shopping_mall` | `shopping → shopping_mall` | Flat |
| `grocery_store` | `shopping → food_and_beverage_store → grocery_store` | Flat |

```sql
-- Containment: "restaurants in Madrid"
SELECT p.id, p.name, p.geometry
FROM places p
JOIN division_areas da ON ST_Within(p.geometry, da.geometry)
WHERE list_contains(p.category_hierarchy, '{category}')
  AND (da.name ILIKE '%{place}%' OR da.name_fr ILIKE '%{place}%' /* ...other name_* columns */);
```

```sql
-- Proximity: "hotels near the station"
SELECT p.id, p.name, p.geometry
FROM places p, {ref_table} ref
WHERE list_contains(p.category_hierarchy, '{category}')
  AND (ref.name ILIKE '%{place}%' OR ref.name_fr ILIKE '%{place}%' /* ...other name_* columns */)
  AND ST_DWithin_Spheroid(
        ST_Point2D(ST_X(p.geometry), ST_Y(p.geometry)),
        ST_Point2D(ST_X(ST_ClosestPoint(ref.geometry, p.geometry)), ST_Y(ST_ClosestPoint(ref.geometry, p.geometry))),
        {distance_m});
```

Params: `{category}` — `restaurant`, `lodging`, `hotel`, `school`, `hospital`, `shopping_mall`, or `grocery_store` (any value that appears in `category_hierarchy`), `{place}`, `{ref_table}` (`divisions`/`division_areas`/`water`, whichever the user names), `{distance_m}` (default 5000).

- **`list_contains(category_hierarchy, '{category}')`, not `category = '{category}'`** (i.e. not an exact match on `taxonomy.primary`): confirmed against a real extract (bbox around Geneva, ~16k places, not committed to the repo) that `taxonomy.primary` is the *most specific* category, not a general one — `italian_restaurant`, `sushi_restaurant`, `pizza_restaurant`, etc. all have their own `primary`, distinct from plain `restaurant`; same for `elementary_school` under `school`. Filtering on exact `primary = 'restaurant'` matched only **457** of the real **1,407** restaurants in the test extract; `list_contains(hierarchy, 'restaurant')` correctly matched all 1,407 (see DESIGN.md for the full breakdown). `hotel`, `hospital`, `shopping_mall`, and `grocery_store` have no such subtypes in the same extract (each a leaf node in its hierarchy) — `list_contains` is used for all of them anyway, for consistency and in case Overture adds subtypes later.
  - Caveat found in the same check: `fast_food_restaurant`'s hierarchy is `['food_and_drink', 'casual_eatery', 'fast_food_restaurant']` — it skips the `restaurant` node entirely, so `list_contains(hierarchy, 'restaurant')` does **not** include fast food. Open question, not resolved here: should "restaurants" include fast food? If yes, the filter needs an explicit `OR list_contains(category_hierarchy, 'fast_food_restaurant')` (or match on the broader `food_and_drink` node, which is far broader than "restaurant" alone and would also pull in bars/cafes).
- No `ST_Centroid` needed on `p.geometry`: `places` geometry is always a point (confirmed in the schema check), unlike the mixed-geometry `infrastructures`/`water` views.
- `confidence`/`operating_status` were considered as quality filters (e.g. excluding permanently-closed venues) but dropped from the view for simplicity — not used by these templates. Revisit if closed/low-confidence venues turn out to pollute results in practice.

---

## 10. Bordering a place — "cities bordering Morocco"

**NL examples**: "cities bordering Morocco" / "villes frontalières du Maroc", "cities near the Moroccan border"

Not the same relation as template #5 (periphery): periphery is "near the edge of the area the entity is *inside*" (requires `ST_Within`); "bordering" means near a *different* area's edge, from either side — a Spanish city near the Moroccan border counts as "bordering Morocco" too, even though it's not inside Morocco. So this drops the containment constraint template #5 has, and the reference is any named `division_areas` polygon (a country, but also a region — "cities bordering Andalusia" reads the same way), not necessarily a country.

This relation was raised when discussing what STEP 2 (the skipped relations graph, see DESIGN.md) would have covered — `ST_Touches` was on that original list. Two distinct interpretations exist for "bordering": (A) *near* a place's border regardless of which side — chosen here; (B) topological adjacency between two areas, i.e. "countries bordering Morocco" meaning countries whose polygon actually touches Morocco's (`ST_Touches`) — not implemented, deliberately deferred since it wasn't the interpretation picked, but a real gap if a question turns out to mean that instead.

```sql
WITH ref AS (
    SELECT geometry
    FROM division_areas
    WHERE (name ILIKE '%{place}%' OR name_fr ILIKE '%{place}%' /* ...other name_* columns */)
    LIMIT 1
),
bbox AS MATERIALIZED (
    SELECT ST_Boundary(geometry) AS border,
           ST_Expand(
             ST_Envelope(geometry),
             {distance_m} / (111320.0 * cos(radians(
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
        ST_Point2D(ST_X(ST_ClosestPoint(bbox.border, d.geometry)), ST_Y(ST_ClosestPoint(bbox.border, d.geometry))),
        {distance_m});
```

Params: `{place}` (the country/region name), `{distance_m}` (default **20,000m** — see below).

Tested against real data (`data/db/llaici.duckdb`) via the STEP 4 sampler: 3/3 join-driven samples succeeded (Portugal ×2, Morocco ×1 as the `{place}`, real border towns found — e.g. Ozão, Sapelos), each producing a non-empty result once filled and executed. Took more attempts than most other templates (18 attempts for 3 hits, vs. single digits elsewhere) but no hang or performance issue — the bbox prefilter does its job even on a country-sized `ST_Boundary`.

- **Correction (later full run):** the "no performance issue" above didn't hold at scale. For a country crossing the antimeridian (Russia, the US — present in `division_areas` only as polygons, see `fetch_countries()`), the bbox spans the globe, so every locality passed the prefilter; `ST_Boundary` of a ~200k-point polygon was then recomputed per row and everything sorted by `random()` — one attempt reached ~62 GB and got OOM-killed. The sampler (not the template SQL above) now computes the boundary once (`MATERIALIZED` CTE) and only runs the exact check on 200 random bbox candidates, and such countries are no longer sampled at all.
- **Boundary computed once:** `ST_Boundary(geometry)` lives in the `bbox` CTE (as `border`), and the CTE is `MATERIALIZED` so DuckDB can't inline it back into the per-row predicate. The original version called `ST_Boundary(bbox.area)` inside the `WHERE`, i.e. twice per candidate locality — for France (~80k-point polygon, ~15k localities in the bbox) that's tens of thousands of full boundary rebuilds. This is the exact SQL `02_fill_and_validate.py` (`build_bordering`) executes, so it's also what `validate-samples` runs.
- The bbox prefilter is necessary here: `division_areas` for a country is a large, complex polygon and `divisions` has ~976,733 rows EU-wide (see `SCHEMA.md`) — same unfiltered-join risk as `water`/`division_areas` joins elsewhere in this file.
- `{distance_m}` default set to **20,000m** (wider than the 5,000m generic-proximity default): a country border isn't a fixed-width concept the way "near a city" is, and 5km would likely miss most real "border town" examples. Not recalibrated further after the initial test (small sample, 3 rows) — revisit if border-adjacent results look too sparse or too broad once run at STEP 4 scale.

---

## 11. Show a division — "show me Madrid" / "where is Portugal"

**NL examples**: "show me Madrid" / "montre-moi Madrid", "where is Portugal" / "où se trouve le Portugal"

Unlike every template above, this one isn't a relation between an entity and a place — it just resolves a named place to its geometry, so the frontend map can display it. A single query returns **two rows**: the place's point (`divisions`) and, when one exists, its administrative boundary (`division_areas`) — not two geometry columns on one row, which would break every other template's "one row = one feature with one geometry" convention that `app/backend/app/duckdb_service.py` relies on. Two rows of different geometry types is already handled without any backend/frontend change: `MapView.vue` already renders a mixed-geometry `FeatureCollection` (see template #10's `divisions` results next to template #8's `water` results, for instance) via one MapLibre layer per geometry type. Works identically whether `{place}` is a city, a region, or a country — no separate handling needed, since `divisions`/`division_areas` cover every administrative level.

```sql
WITH d AS (
    SELECT id, name, geometry
    FROM divisions
    WHERE (name ILIKE '%{place}%' OR name_fr ILIKE '%{place}%' /* ...other name_* columns */)
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
SELECT id, name, geometry FROM area;
```

Params: `{place}` only.

Three things this template exists specifically to work around, each found by testing against real data:
- **`ORDER BY population DESC NULLS LAST` on the `d` CTE**: a plain `ILIKE '%{place}%' LIMIT 1` with no ordering picks an arbitrary name match — tested with "Lisbon": without this, the `divisions` name columns only ever store one "primary" language per row, and the row that happened to come back first was an unrelated "Lisbon Green Valley" neighborhood (a `NULL`-population microhood), not the city — `population DESC` reliably prefers the real, well-known place over an obscure same-named or partially-matching one.
- **Joining `area` via `division_areas.division_id = d.id`, not a second independent name match**: matching `division_areas` by name again (like every other template does) risks resolving to a *different* entity than the one `d` picked — two independent `ILIKE` matches aren't guaranteed to agree. Joining on the FK the two views already share removes that risk entirely.
- **`da.is_land = true` + `ORDER BY ST_Area(...) DESC`**: a country typically has more than one `division_areas` row for the same `division_id` — tested with Portugal, which has both a `land` polygon (the actual landmass, ~9.6° area) and a `maritime` one (its EEZ/territorial waters, ~16.5° area, `is_land = false`). Without the `is_land` filter, `ORDER BY area DESC` alone would pick the (larger) maritime boundary instead of the coastline — wrong for "show me Portugal".

Tested against real data (`data/db/llaici.duckdb`): Madrid (point + its real ~0.37°×0.33° `locality` polygon, not a random same-named micro-fragment — see the `division_areas` `LIMIT 1` bug this same issue caused in the `/query/mock` backend endpoint), Lisbon (correctly resolves via `name_pt`/`name_en`… to "Lisboa"), Portugal (point + the land-only `MULTIPOLYGON`, maritime boundary correctly excluded). A place with no matching `division_areas` row (data gap, or a `division_id` that doesn't exist) still returns the point alone — not treated as an error, just a smaller result.

---

## 12. Cities in a direction — "cities north of Paris"

**NL examples**: "cities north of Paris" / "villes au nord de Paris"

Same relation as templates #6/#7, but the candidate entity is a **city** (`divisions`, `subtype = 'locality'`), not an `infrastructures` category — and unlike every other template, the geometry returned is its **polygon** (`division_areas`), not its point, when one exists. The direction/distance math is still computed on points (a city's polygon has no single "bearing"), but the final `SELECT` swaps in the polygon via a `LEFT JOIN` on `division_id` — `LEFT`, not `INNER`, so a city with no matching `division_areas` row (a data gap, not an error) still comes back as its point instead of being silently dropped from the result.

```sql
WITH ref AS (
    SELECT geometry
    FROM divisions
    WHERE (name ILIKE '%{place}%' OR name_fr ILIKE '%{place}%' /* ...other name_* columns */)
    ORDER BY population DESC NULLS LAST
    LIMIT 1
),
city AS (
    SELECT e.id, e.name, e.geometry
    FROM divisions e, ref
    WHERE e.subtype = 'locality'
      AND {angle_condition}
      AND ST_Distance_Spheroid(
            ST_Point2D(ST_X(ref.geometry), ST_Y(ref.geometry)),
            ST_Point2D(ST_X(e.geometry), ST_Y(e.geometry))
          ) <= 100000  -- swap for the #7-style BETWEEN band for "X km north of..."
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
LEFT JOIN ranked_area area ON area.division_id = city.id AND area.rn = 1;
```

Params: `{place}`, `{angle_condition}` (see template #6 — same table, same North wraparound handling). For the distance-band variant ("X km north of Paris"), replace the `<= 100000` line with template #7's `BETWEEN {distance_m} - tolerance AND {distance_m} + tolerance` clause — no other change needed, same as #6 → #7.

Three points carried over or new here:
- **`ref` needs `ORDER BY population DESC NULLS LAST LIMIT 1`, unlike templates #6/#7's `ref`**: those join `ref` inline without disambiguation because their candidate table (`infrastructures`) is unrelated to `divisions`, so an ambiguous `ref` match just risks the wrong *reference* point, not a self-referential mess. Here the candidate table is `divisions` too (`e`), so an undisambiguated `ref` measured against real data actually broke: matching "Paris" without ordering picked an obscure micro-place (a `microhood` or low-population hamlet also named "Paris" in the fuzzy `ILIKE` sense) over the real city — same failure shape as template #11's Lisbon/Madrid fix. `ORDER BY population DESC` fixes it the same way.
- **`ranked_area`'s `is_land = true` + largest-area tie-break**: identical reasoning to template #11 — a `division_areas` row can have a maritime/EEZ duplicate or multiple admin-level polygons for the same `division_id`; `is_land` + `ORDER BY ST_Area(...) DESC` picks the actual coastline/city boundary.
- **`LEFT JOIN` + `COALESCE`, not `INNER JOIN`**: tested against real data (cities north of Madrid, 100km): 217/381 matches had no `division_areas` row at all (small villages Overture hasn't mapped a boundary for) — an `INNER JOIN` would have silently dropped 57% of otherwise-correct results. `COALESCE(area.geometry, city.geometry)` keeps them, falling back to the point.

Tested against real data (`data/db/llaici.duckdb`): "Paris" itself has no exact-name match in this dataset extract (a pre-existing data gap, unrelated to this template — confirmed separately, not a bug here), so verified against Madrid instead: 381 cities matched north of Madrid within 100km, 154 `POLYGON` + 10 `MULTIPOLYGON` + 217 `POINT` (fallback), correct geometry-type mix, ran in well under a second (no bbox prefilter needed — same reasoning as templates #6/#7: the `ref` match narrows to one row, so the join is a single-row-vs-`locality`-subtype scan, not a combinatorial one).

---

## Left bank / right bank (no longer deferred to STEP 2)

**Left bank / right bank** ("rive gauche" / "rive droite", "left bank" / "right bank") was originally thought to need a precomputed `side: 'left' | 'right'` attribute in an offline relations graph (STEP 2), because determining which side of a waterway a point falls on seemed to require:

1. A correctly-oriented waterway geometry (source → mouth) — not guaranteed in Overture data as-is.
2. A cross-product test between the river's local direction vector and the vector to the candidate point.

⚠️ **Revised approach**: like [geoblocks/etter](https://github.com/geoblocks/etter), we skip the reorientation step entirely and trust the geometry's existing digitization order — OSM convention states waterway ways "should point in the direction of water flow." Unlike `etter` (a Python library, free to use Shapely's `offset_curve`), the generated SQL here must be plain SQL the fine-tuned LLM outputs directly — and DuckDB's spatial extension has no `ST_OffsetCurve` (verified against `duckdb_functions()`). So the side is computed live with a cross-product test built from `ST_LineLocatePoint`/`ST_LineInterpolatePoint` instead of a one-sided buffer:

```sql
-- Left bank / right bank of a waterway
-- Trusts waterway digitization order (OSM convention: drawn in flow direction).
-- "side" = sign of the 2D cross product between the river's local tangent
-- at the point closest to the candidate, and the vector to the candidate.
WITH segments AS (
    SELECT geometry AS line
    FROM water
    WHERE class IN ('river', 'stream', 'canal')
      AND ST_GeometryType(geometry) = 'LINESTRING'
      AND (name ILIKE '%{feature}%' OR name_fr ILIKE '%{feature}%' /* ...other name_* columns */)
),
bbox AS (
    SELECT line,
           ST_Expand(
             ST_Envelope(line),
             {distance_m} / (111320.0 * cos(radians(
               (ST_YMin(ST_Envelope(line)) + ST_YMax(ST_Envelope(line))) / 2
             )))
           ) AS box
    FROM segments
),
candidates AS (
    SELECT e.id, e.name, ST_Centroid(e.geometry) AS cgeom, b.line,
           row_number() OVER (PARTITION BY e.id ORDER BY ST_Distance(b.line, ST_Centroid(e.geometry))) AS rn
    FROM infrastructures e, bbox b
    WHERE e.subtype = '{subtype}' AND e.class = '{class}'
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
        {distance_m})
  AND sign(
    (ST_X(ST_LineInterpolatePoint(line, LEAST(f + 0.0001, 1))) - ST_X(ST_LineInterpolatePoint(line, GREATEST(f - 0.0001, 0))))
  * (ST_Y(cgeom) - ST_Y(ST_LineInterpolatePoint(line, f)))
  - (ST_Y(ST_LineInterpolatePoint(line, LEAST(f + 0.0001, 1))) - ST_Y(ST_LineInterpolatePoint(line, GREATEST(f - 0.0001, 0))))
  * (ST_X(cgeom) - ST_X(ST_LineInterpolatePoint(line, f)))
) = {side_sign};  -- {side_sign}: 1 for left, -1 for right
```

Params: `{feature}`, `{subtype}`, `{class}`, `{distance_m}`, `{side_sign}` (1 = left, -1 = right).

Verified end-to-end against the real dataset (river "Tevere", `bridge` candidates, 3km): left/right results were disjoint (no id in both sets), and non-empty on both sides. Longer than the other templates, but still amounts to one `ST_Distance`/`ST_Intersects` pass over a bbox-narrowed candidate set — no precomputation or graph needed. The small fine-tuned model just reproduces this fixed block with placeholders filled in, same as any other template.

Every non-obvious piece here exists because of something that broke when tested against the real data, not by design upfront:
- `class IN (...)` + `ST_GeometryType(...) = 'LINESTRING'`: a plain name match on `water` also returns polygon "river-as-area" segments and unrelated same-named features (a fountain, a spring) — confirmed on "Tevere" — none of which have a meaningful flow direction.
- `row_number() ... rn = 1`: a named river is usually split into many segments (17 for "Tevere") — picking an arbitrary one instead of the nearest gave wrong/undefined results for candidates near a different segment.
- `bbox`: without it, this query OOM'd (19+ GiB) on the full `infrastructures` table; with it, well under a second.
- `ST_Point2D(...)` wraps: `ST_DWithin_Spheroid` only accepts `POINT_2D` (see above).

Known limitation, unrelated to the above and not mitigated: this trusts OSM's digitization convention ("should" point in flow direction, not enforced), so minor/unnamed streams are more likely to be mis-oriented than major named rivers. Revisit if left/right bank queries turn out unreliable in practice (e.g. restrict to named/major waterway `class` values).

This is why STEP 2 (the relations graph) was skipped entirely — see DESIGN.md.

## Calibration items (resolved)

All three were resolved inline in their respective templates rather than left as free parameters to tune during STEP 4 — each is now a fixed formula baked into the SQL, so there's nothing left to calibrate at dataset-generation time:

- `{center_radius_m}` (template #4): computed from the area's own extent — `clamp(0.15 * sqrt(area_m²), 300m, 5000m)`.
- Direction without distance (template #6): fixed 100km default bound, `ST_Distance_Spheroid(...) <= 100000`.
- Tolerance band (template #7): `clamp(0.20 * distance_m, 200m, 10000m)` instead of a flat ±20%.

## Known limitation — no compound/composite queries

Every template above encodes exactly **one** spatial relation. A question combining two at once — e.g. "restaurants north of Lisbon **and** along the coast" (direction relative to one place + proximity to a different water feature, in the same query) — has no matching template and no training examples that combine two relations this way.

**Why this matters more than it might look**: this isn't just "a template we haven't written yet" — the fine-tuned model only ever sees single-relation SQL shapes during training (`DOC.md`/`FINETUNING.md`), so it has no compositional training signal to draw on. Unlike a large general-purpose LLM (which has broad SQL competence from pretraining and can often improvise a combined `WHERE` clause reasonably), a small model fine-tuned narrowly on fixed templates is unlikely to generalize to an unseen compound structure — it would more likely drop one of the two constraints (probably whichever phrasing is less template-like) or produce syntactically plausible but semantically wrong SQL, rather than correctly `AND`-ing two independent relations together.

**Decision (Option A, chosen over building compound templates now)**: accept this limitation for v1. A user asking a compound question gets an answer to (at best) one of the two constraints, or a wrong/empty result — not handled specially, and not silently corrected. Reformulating as two separate single-relation questions is the workaround for now.

**Considered and rejected for now (Option B)**: a dedicated template family for common relation *pairs* (direction + water-proximity, containment + category, etc.), each needing its own SQL pattern, sampler, and training examples. Rejected because it's combinatorial — every new pair of relations is a new template to design, test against real data, and generate examples for (the same amount of work as adding one of the 10 templates above, repeated per combination) — premature before a first model has even been trained on the single-relation case.

**Considered, not built now (Option C — orchestration instead of new templates)**: rather than teaching the model compound queries, decompose the question *before* it reaches the model:
1. A router detects the question is compound (keyword heuristic, or a lightweight LLM classification call) and splits it into two single-relation sub-questions, carrying the shared subject (e.g. "restaurants") into both branches.
2. Each sub-question goes through the existing fine-tuned model unchanged, producing two single-relation SQL queries — no new training data needed, unlike Option B.
3. The two are merged with a plain SQL `INTERSECT` rather than an application-level join: every template already `SELECT`s `id` first (`app/backend/app/duckdb_service.py` relies on this same convention for its GeoJSON conversion), so the two result sets are already shape-compatible for `INTERSECT` as long as both queries select the same columns.

The hard part isn't the merge (step 3 is close to free, given the existing `id`-first convention) — it's step 1: reliably detecting a compound question, splitting it, and correctly carrying the shared subject into both halves is its own non-trivial NLP problem, arguably no less fragile than what Option B tries to avoid, just moved to a different component (an orchestrator) instead of the model itself.

**Next step**: revisit Options B and C once STEP 5 (fine-tuning) has produced and evaluated a first model on the current single-relation templates (see `FINETUNING.md`). Whether compound queries are worth building — and which of the two approaches — should be judged against real usage, not decided upfront.
