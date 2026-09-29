# SCHEMA.md — Logical schema contract (DuckDB views)

This document describes the **logical schema** the fine-tuned LLM is trained on, and against which its generated SQL queries run. This is the "contract" mentioned in DESIGN.md (`schema.sql`).

Principle (see DESIGN.md, "Logical view layer" section): the LLM **never** sees raw Overture columns (nested structs, `names.common` as a `MAP`, S3 partitions by theme/type). It generates SQL against a fixed set of **flat views**, defined once and for all. These views can be backed either by:

- local **GeoParquet files** (`data/parquet/*.parquet` — this repo's current mode),
- or directly by **Overture on S3** (`read_parquet('s3://overturemaps-us-west-2/release/.../theme=.../type=...')`),

without the model's generated SQL ever needing to change.

Raw sources covered in this document: `eu_divisions`, `eu_division_areas`, `eu_infrastructures`, `eu_water` (Overture extracts, EU scope).

---

## 1. View `divisions`

Raw source: `eu_divisions.parquet` (Overture theme `divisions`, type `division`).
Geometry: representative **point** (country, region, city, neighborhood...).

### Selected columns

| View column | Origin (raw column) | Type | Rationale |
|---|---|---|---|
| `id` | `id` | VARCHAR | Stable identifier (GERS) |
| `name` | `names.primary` | VARCHAR | Primary display name |
| `name_fr`, `name_es`, `name_it`, `name_de`, `name_en`, `name_pt`, `name_ru`, `name_uk`, `name_nl`, `name_pl`, `name_th`, `name_zh`, `name_ar` | `names.common['fr']`, `['es']`, `['it']`, `['de']`, `['en']`, `['pt']`, `['ru']`, `['uk']`, `['nl']`, `['pl']`, `['th']`, `['zh']`, `['ar']` | VARCHAR | Multilingual name search — the LLM does not translate, it searches the term as-is in these columns (see DESIGN.md STEP 3) |
| `country` | `country` | VARCHAR | ISO 3166-1 alpha-2 code (`FR`, `ES`...) of the country the entity belongs to — lets `01_sample_entities.py` pick references per country with a plain equality instead of an `ST_Within` against the country polygon |
| `subtype` | `subtype` | VARCHAR | Distinguishes `country`/`region`/`county`/`localadmin`/`locality`/`neighborhood`/`microhood`/`macrohood`/`dependency` |
| `class` | `class` | VARCHAR | Sub-category of `locality` (`hamlet`, `village`, `town`, `city`) — often `NULL` for other subtypes |
| `admin_level` | `admin_level` | INTEGER | Administrative hierarchy level |
| `population` | `population` | INTEGER | Enables "largest city" questions, sorting by population |
| `geometry` | `geometry` | GEOMETRY (Point) | Representative position of the entity |

### Observed `subtype`/`class` distribution (EU)

```
locality/hamlet        508,031      neighborhood/NULL    116,984
locality/village        214,002      microhood/NULL         69,382
macrohood/NULL           26,481      localadmin/NULL        11,718
locality/NULL             10,844      locality/town          10,158
county/NULL                7,318      locality/city              940
region/NULL                   842      country/NULL                  30
dependency/NULL                 3
```

### View definition

```sql
-- Local mode
CREATE OR REPLACE VIEW divisions AS
SELECT
    id,
    names.primary          AS name,
    names.common['fr']     AS name_fr,
    names.common['es']     AS name_es,
    names.common['it']     AS name_it,
    names.common['de']     AS name_de,
    names.common['en']     AS name_en,
    names.common['pt']     AS name_pt,
    names.common['ru']     AS name_ru,
    names.common['uk']     AS name_uk,
    names.common['nl']     AS name_nl,
    names.common['pl']     AS name_pl,
    names.common['th']     AS name_th,
    names.common['zh']     AS name_zh,
    names.common['ar']     AS name_ar,
    country,
    subtype,
    class,
    admin_level,
    population,
    geometry
FROM read_parquet('data/parquet/eu_divisions.parquet');

-- Direct-Overture mode (S3) — same columns, same names
CREATE OR REPLACE VIEW divisions AS
SELECT
    id,
    names.primary          AS name,
    names.common['fr']     AS name_fr,
    names.common['es']     AS name_es,
    names.common['it']     AS name_it,
    names.common['de']     AS name_de,
    names.common['en']     AS name_en,
    names.common['pt']     AS name_pt,
    names.common['ru']     AS name_ru,
    names.common['uk']     AS name_uk,
    names.common['nl']     AS name_nl,
    names.common['pl']     AS name_pl,
    names.common['th']     AS name_th,
    names.common['zh']     AS name_zh,
    names.common['ar']     AS name_ar,
    country,
    subtype,
    class,
    admin_level,
    population,
    geometry
FROM read_parquet('s3://overturemaps-us-west-2/release/.../theme=divisions/type=division/*');
```

---

## 2. View `division_areas`

Raw source: `eu_division_areas.parquet` (theme `divisions`, type `division_area`).
Geometry: **polygon** (administrative boundary). Complementary to `divisions` (joinable via `division_id` ↔ `divisions.id`): `divisions` serves distance/proximity queries (point), `division_areas` serves containment queries (`ST_Contains`, `ST_Within`).

### Selected columns

| View column | Origin | Type | Rationale |
|---|---|---|---|
| `id` | `id` | VARCHAR | Area identifier |
| `division_id` | `division_id` | VARCHAR | Join key to `divisions.id` |
| `name` | `names.primary` | VARCHAR | Primary name |
| `name_fr`, `name_es`, `name_it`, `name_de`, `name_en`, `name_pt`, `name_ru`, `name_uk`, `name_nl`, `name_pl`, `name_th`, `name_zh`, `name_ar` | `names.common[...]` | VARCHAR | Multilingual search, consistent with `divisions` |
| `country` | `country` | VARCHAR | ISO 3166-1 alpha-2 code (`FR`, `ES`...) of the country the area belongs to — lets `01_sample_entities.py` pick references per country with a plain equality instead of an `ST_Within` against the country polygon |
| `subtype` | `subtype` | VARCHAR | Same nomenclature as `divisions` |
| `class` | `class` | VARCHAR | `land` / `maritime` |
| `is_land` | `is_land` | BOOLEAN | Distinguishes land area |
| `is_territorial` | `is_territorial` | BOOLEAN | Distinguishes territorial waters |
| `geometry` | `geometry` | GEOMETRY (Polygon) | Boundary for spatial containment |

### Observed `subtype`/`class` distribution (EU)

```
locality/land       131,315      microhood/land       54,515
neighborhood/land     36,412      localadmin/land       11,729
county/land             7,417      macrohood/land          6,954
region/land               879      country/land               41
region/maritime            44      country/maritime           26
dependency/land              3      dependency/maritime            3
county/maritime              2
```

### View definition

```sql
CREATE OR REPLACE VIEW division_areas AS
SELECT
    id,
    division_id,
    names.primary          AS name,
    names.common['fr']     AS name_fr,
    names.common['es']     AS name_es,
    names.common['it']     AS name_it,
    names.common['de']     AS name_de,
    names.common['en']     AS name_en,
    names.common['pt']     AS name_pt,
    names.common['ru']     AS name_ru,
    names.common['uk']     AS name_uk,
    names.common['nl']     AS name_nl,
    names.common['pl']     AS name_pl,
    names.common['th']     AS name_th,
    names.common['zh']     AS name_zh,
    names.common['ar']     AS name_ar,
    country,
    subtype,
    class,
    is_land,
    is_territorial,
    geometry
FROM read_parquet('data/parquet/eu_division_areas.parquet');
-- S3 variant: theme=divisions/type=division_area/*
```

---

## 3. View `infrastructures`

Raw source: `eu_infrastructures.parquet` (theme `base`, type `infrastructure`).
Very heterogeneous dataset (street furniture, power, barriers, transport...). **Filtered** to categories relevant for natural-language geocoding — the rest (power poles, fences, walls, kerbs, street lamps, benches, traffic signs...) is excluded since it is never referenced by a natural-language question and would bloat the view without adding value for the model.

### Selected categories (`subtype`/`class`)

| subtype | class | EU volume | Why kept |
|---|---|---|---|
| `transit` | `parking` | 2,288,241 | Parking lots |
| `transit` | `parking_space` | 1,793,586 | Individual parking spaces |
| `bridge` | `bridge` | 1,649,078 | Bridges — named geographic landmarks |
| `transit` | `bus_stop` | 1,277,917 | Bus stops |
| `transit` | `bicycle_parking` | 413,375 | Bicycle parking |
| `pedestrian` | `toilets` | 118,999 | Public toilets |
| `waste_management` | `waste_basket` | 628,825 | Waste baskets |
| `waste_management` | `recycling` | 400,974 | Recycling points |
| `waste_management` | `waste_disposal` | 115,599 | Waste disposal points |
| `transit` | `railway_station` | 19,479 | Train stations |
| `transit` | `subway_station` | 2,712 | Metro/subway stations |
| `transit` | `railway_halt` | 14,602 | Minor train stops |
| `transit` | `bus_station` | 8,941 | Bus stations (terminals, distinct from `bus_stop`) |
| `aerialway` | `aerialway_station` | 24,097 | Cable-car/gondola stations |
| `airport` | `airport` | 3,213 | Airports |
| `airport` | `international_airport` | 226 | Airports (international) |
| `airport` | `regional_airport` | 45 | Airports (regional) |
| `airport` | `municipal_airport` | 14 | Airports (municipal) |
| `airport` | `military_airport` | 231 | Airports (military) |
| `airport` | `private_airport` | 169 | Airports (private) |
| `airport` | `heliport` | 185 | Heliports |
| `airport` | `helipad` | 12,788 | Helipads |
| `airport` | `runway` | 6,768 | Runways |
| `airport` | `taxiway` | 38,690 | Taxiways |
| `airport` | `taxilane` | 2,403 | Taxilanes |
| `airport` | `apron` | 8,324 | Aprons |
| `airport` | `airport_gate` | 3,529 | Airport gates |
| `airport` | `airstrip` | 983 | Airstrips |
| `airport` | `stopway` | 495 | Stopways |
| `communication` | `communication_tower` | 85,200 | Communication towers |
| `communication` | `mobile_phone_tower` | 65,794 | Mobile phone towers |
| `communication` | `communication_pole` | 1,195 | Communication poles |
| `communication` | `communication_line` | 5,026 | Communication lines |

**Excluded**: `power` (poles, lines, generators, substations), `barrier` (fences, walls, hedges, gates, kerbs), `transportation/street_lamp`, `transportation/give_way`, `transportation/stop` (traffic signs), `pedestrian/bench`, `pedestrian/information` — cartographic noise with no value for a natural-language query. Also excluded: `transportation/crossing`, `emergency/fire_hydrant`, and `transit/stop_position` — dropped after review, not a good fit for NL geocoding questions. `airport/launchpad` (2 rows EU-wide) also excluded as negligible.

⚠️ To validate/adjust: this list is an initial proposal. If a use case specifically targets one of these excluded categories (e.g. street-furniture questions), add it to the view's `WHERE` clause.

### Selected columns

| View column | Origin | Type | Rationale |
|---|---|---|---|
| `id` | `id` | VARCHAR | Identifier |
| `name` | `names.primary` | VARCHAR | Name (often NULL for this theme, e.g. a named bus stop) |
| `name_fr`, `name_es`, `name_it`, `name_de`, `name_en`, `name_pt`, `name_ru`, `name_uk`, `name_nl`, `name_pl`, `name_th`, `name_zh`, `name_ar` | `names.common[...]` | VARCHAR | Multilingual, consistent with other views |
| `subtype` | `subtype` | VARCHAR | Broad category |
| `class` | `class` | VARCHAR | Precise sub-category |
| `level` | `level` | INTEGER | Level (useful to disambiguate underground/surface/elevated, e.g. stacked stations/bridges) |
| `geometry` | `geometry` | GEOMETRY | Position |

### View definition

```sql
CREATE OR REPLACE VIEW infrastructures AS
SELECT
    id,
    names.primary          AS name,
    names.common['fr']     AS name_fr,
    names.common['es']     AS name_es,
    names.common['it']     AS name_it,
    names.common['de']     AS name_de,
    names.common['en']     AS name_en,
    names.common['pt']     AS name_pt,
    names.common['ru']     AS name_ru,
    names.common['uk']     AS name_uk,
    names.common['nl']     AS name_nl,
    names.common['pl']     AS name_pl,
    names.common['th']     AS name_th,
    names.common['zh']     AS name_zh,
    names.common['ar']     AS name_ar,
    subtype,
    class,
    level,
    geometry
FROM read_parquet('data/parquet/eu_infrastructures.parquet')
WHERE (subtype, class) IN (
    ('transit',        'parking'),
    ('transit',        'parking_space'),
    ('bridge',         'bridge'),
    ('transit',        'bus_stop'),
    ('transit',        'bicycle_parking'),
    ('transit',        'railway_station'),
    ('transit',        'subway_station'),
    ('transit',        'railway_halt'),
    ('transit',        'bus_station'),
    ('aerialway',      'aerialway_station'),
    ('pedestrian',     'toilets'),
    ('waste_management', 'waste_basket'),
    ('waste_management', 'recycling'),
    ('waste_management', 'waste_disposal'),
    ('airport',        'airport'),
    ('airport',        'international_airport'),
    ('airport',        'regional_airport'),
    ('airport',        'municipal_airport'),
    ('airport',        'military_airport'),
    ('airport',        'private_airport'),
    ('airport',        'heliport'),
    ('airport',        'helipad'),
    ('airport',        'runway'),
    ('airport',        'taxiway'),
    ('airport',        'taxilane'),
    ('airport',        'apron'),
    ('airport',        'airport_gate'),
    ('airport',        'airstrip'),
    ('airport',        'stopway'),
    ('communication',  'communication_tower'),
    ('communication',  'mobile_phone_tower'),
    ('communication',  'communication_pole'),
    ('communication',  'communication_line')
);
-- S3 variant: theme=base/type=infrastructure/*
```

---

## 4. View `water`

Raw source: `eu_water.parquet` (theme `base`, type `water`).
Geometry: line (waterway) or polygon (water body) depending on `subtype`.

### Selected columns

| View column | Origin | Type | Rationale |
|---|---|---|---|
| `id` | `id` | VARCHAR | Identifier |
| `name` | `names.primary` | VARCHAR | Primary name (e.g. "Lago de Sanabria") |
| `name_fr`, `name_es`, `name_it`, `name_de`, `name_en`, `name_pt`, `name_ru`, `name_uk`, `name_nl`, `name_pl`, `name_th`, `name_zh`, `name_ar` | `names.common[...]` | VARCHAR | Multilingual — key use case from the doc (`Lago de Sanabria` / `Lake Sanabria`) |
| `subtype` | `subtype` | VARCHAR | `river`, `lake`, `canal`, `stream`, `pond`, `reservoir`, `spring`, `water`, `human_made`, `physical`, `wastewater` |
| `class` | `class` | VARCHAR | Precise sub-category |
| `is_salt` | `is_salt` | BOOLEAN | Distinguishes salt/fresh water |
| `is_intermittent` | `is_intermittent` | BOOLEAN | Seasonal vs. permanent waterway |
| `geometry` | `geometry` | GEOMETRY (Line/Polygon) | Water body/waterway geometry |

### Observed `subtype`/`class` distribution (EU)

```
stream/stream            3,693,110    human_made/swimming_pool   1,671,854
canal/ditch                1,026,883    water/water                    755,672
pond/pond                     561,776    canal/drain                    485,878
river/river                    215,298    canal/canal                    160,097
reservoir/reservoir           126,101    spring/spring                   108,405
reservoir/basin               106,231    water/wastewater                 54,156
lake/lake                        18,571    physical/waterfall               18,242
physical/bay                       9,589    human_made/salt_pond              8,531
physical/cape                      6,891    lake/oxbow                          3,738
reservoir/water_storage            3,236    lake/lagoon                         2,596
wastewater/sewage                   2,203    human_made/fish_pass                2,102
pond/fishpond                        1,406    water/tidal_channel                   931
canal/moat                              844
```

⚠️ As with `infrastructures`, some classes (`human_made/swimming_pool`, `wastewater/sewage`, `canal/ditch`, `canal/drain`) are large in volume but of low relevance for natural-language geocoding ("swimming pool", "sewage"). Not filtered for now (unlike `infrastructures`) — to be decided if the view's size/relevance becomes an issue.

### View definition

```sql
CREATE OR REPLACE VIEW water AS
SELECT
    id,
    names.primary          AS name,
    names.common['fr']     AS name_fr,
    names.common['es']     AS name_es,
    names.common['it']     AS name_it,
    names.common['de']     AS name_de,
    names.common['en']     AS name_en,
    names.common['pt']     AS name_pt,
    names.common['ru']     AS name_ru,
    names.common['uk']     AS name_uk,
    names.common['nl']     AS name_nl,
    names.common['pl']     AS name_pl,
    names.common['th']     AS name_th,
    names.common['zh']     AS name_zh,
    names.common['ar']     AS name_ar,
    subtype,
    class,
    is_salt,
    is_intermittent,
    geometry
FROM read_parquet('data/parquet/eu_water.parquet');
-- S3 variant: theme=base/type=water/*
```

---

## 5. View `places`

Raw source: `eu_places.parquet` (theme `places`, type `place`) — downloaded as part of `make download-overture` and defined in `00_init.sql` alongside the other views. Much denser than the other themes (a small test bbox around Geneva alone returned 16,030 rows, 24% of it generic `services_and_business`), so — like `infrastructures` — the view filters down to the categories actually covered by a template (see `WHERE list_intersect(...)` in `00_init.sql`): `restaurant`, `lodging`, `school`, `hospital`, `shopping_mall`, `grocery_store`. Geometry: always a **point**.

Restaurants, hotels, shops, schools, hospitals... — see DESIGN.md "`places` theme — not yet ingested, findings for later" and `TEMPLATES.md` "9. Places by category" for the taxonomy findings this view is based on.

### Selected columns

| View column | Origin | Type | Rationale |
|---|---|---|---|
| `id` | `id` | VARCHAR | Identifier |
| `name` | `names.primary` | VARCHAR | Primary name only — **not** flattened into `name_fr`/`name_es`/... like every other view. Templates #9 never search `places` by name (they filter by category and join spatially); the multilingual columns would only ever be displayed, never queried, so they're skipped here. |
| `category` | `taxonomy.primary` | VARCHAR | The *most specific* category (e.g. `italian_restaurant`) — **not** a general one. Filtering on this directly misses subtypes; see `category_hierarchy`. |
| `category_hierarchy` | `taxonomy.hierarchy` | VARCHAR[] | Full path from general to specific (e.g. `['food_and_drink', 'restaurant', 'italian_restaurant']`). Queries should use `list_contains(category_hierarchy, '...')`, not `category = '...'` — confirmed on real data that the latter misses most matches for categories with subtypes (`category = 'restaurant'` matched 457/1,407 real restaurants; `list_contains(category_hierarchy, 'restaurant')` matched all 1,407). |
| `geometry` | `geometry` | GEOMETRY (Point) | Position — always a point, no `ST_Centroid` needed |

`confidence`/`operating_status` were considered (as quality filters — e.g. excluding permanently-closed venues) but dropped from the view: not used by any template, kept the view minimal.

Like `infrastructures`, this view is filtered to a category allowlist rather than exposing every `taxonomy.hierarchy` value (see `00_init.sql`'s `WHERE list_intersect(...)`): `restaurant`, `lodging`, `school`, `hospital`, `shopping_mall`, `grocery_store` — chosen as a first, likely-to-be-asked-about subset with a template in `TEMPLATES.md`, not an exhaustive list (see the full category breakdown in DESIGN.md: `services_and_business`, `shopping`, `food_and_drink`, `health_care`, `lifestyle_services`, `community_and_government`, `education`, `travel_and_transportation`, `sports_and_recreation`, `arts_and_entertainment`, `cultural_and_historic`, `lodging`, `geographic_entities`). Extending coverage later means adding the new category to both this `WHERE` clause and a new/updated template.

### View definition

```sql
CREATE OR REPLACE VIEW places AS
SELECT
    id,
    names.primary          AS name,
    taxonomy."primary"     AS category,
    taxonomy.hierarchy     AS category_hierarchy,
    geometry
FROM read_parquet('data/parquet/eu_places.parquet')
WHERE len(list_intersect(taxonomy.hierarchy, [
    'restaurant',
    'lodging',
    'school',
    'hospital',
    'shopping_mall',
    'grocery_store'
])) > 0;
-- S3 variant: theme=places/type=place/*
```

⚠️ The raw parquet also still ships the older `categories.primary/alternate` struct alongside `taxonomy`, currently holding identical values (see DESIGN.md) — `taxonomy` is used here since it's the documented current field, but worth re-checking if a future Overture release drops `categories` or lets the two diverge.

---

## 6. Summary — columns common to all views

So the LLM learns a uniform pattern regardless of which view is queried, every view consistently exposes:

```
id, name, name_fr, name_es, name_it, name_de, name_en, name_pt, name_ru, name_uk, name_nl, name_pl, name_th, name_zh, name_ar, subtype, class, geometry
```

**Except `places`**, the one deliberate exception: only `id`, `name` (primary only, no `name_fr`/etc.), `geometry`, plus its own `category`/`category_hierarchy` in place of `subtype`/`class` — see §5 for why (never searched by name, so the multilingual columns would go unused).

View-specific columns:

| View | Specific columns |
|---|---|
| `divisions` | `admin_level`, `population` |
| `division_areas` | `division_id`, `is_land`, `is_territorial` |
| `infrastructures` | `level` (+ restricted `subtype`/`class` filter) |
| `water` | `is_salt`, `is_intermittent` |
| `places` | `category`, `category_hierarchy` — no `subtype`/`class`, no multilingual `name_*`, no `confidence`/`operating_status` (see §5) |

## 7. Open questions

- Languages flattened: `fr`, `es`, `it`, `de`, `en`, `pt`, `ru`, `uk`, `nl`, `pl`, `th`, `zh`, `ar`. Note: this list is a manual selection, not derived from the actual `names.common` key frequency measured on the EU extracts — a real frequency count showed dozens of language keys present (e.g. `sr-Cyrl`, `el`, `hu`, `cs`, `ko`, `oc`... some far more frequent than `pt`), plus non-language keys like `genitive`. Revisit this list against real frequency data if search recall for under-covered languages becomes a problem.
- Decide whether `water` should be filtered like `infrastructures` (excluding swimming pools/sewage/drainage ditches).
- Verify the stability of `names.common` (type `MAP(VARCHAR, VARCHAR)`) across Overture releases before generalizing the `names.common['fr']` syntax to direct-S3 mode.
