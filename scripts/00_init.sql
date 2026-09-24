-- init.sql
-- Loaded on container startup — creates the logical view layer described in SCHEMA.md.
-- Views use paths relative to the repo root (data/parquet/*.parquet, data/db/...), so
-- they resolve the same way whether DuckDB's working directory is the container's /app
-- (with data/ and scripts/ bind-mounted there, see docker-compose.yml) or the repo root
-- on the host (e.g. connecting from DBeaver with the repo as the working directory).

LOAD spatial;
LOAD httpfs;
LOAD json;

ATTACH IF NOT EXISTS 'data/db/llaici.duckdb' AS llaici;
USE llaici;

-- ─────────────────────────────────────────────────────────────
-- divisions (point) — countries, regions, cities, neighborhoods...
-- ─────────────────────────────────────────────────────────────
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
    subtype,
    class,
    admin_level,
    population,
    geometry
FROM read_parquet('data/parquet/eu_divisions.parquet');

-- ─────────────────────────────────────────────────────────────
-- division_areas (polygon) — administrative boundaries
-- ─────────────────────────────────────────────────────────────
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
    subtype,
    class,
    is_land,
    is_territorial,
    geometry
FROM read_parquet('data/parquet/eu_division_areas.parquet');

-- ─────────────────────────────────────────────────────────────
-- infrastructures — filtered to categories relevant for NL geocoding
-- (see SCHEMA.md §3 for the excluded-categories rationale)
-- ─────────────────────────────────────────────────────────────
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

-- ─────────────────────────────────────────────────────────────
-- water — rivers, lakes, canals...
-- ─────────────────────────────────────────────────────────────
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

-- ─────────────────────────────────────────────────────────────
-- places — restaurants, hotels, shops, schools, hospitals...
-- filtered to categories covered by TEMPLATES.md "9. Places by category"
-- (see SCHEMA.md §5 for why: `places` is far denser than the other themes —
-- a ~16k-row test extract was 24% generic services_and_business alone —
-- so it's filtered at ingestion just like `infrastructures` is)
-- Only the primary name is kept (unlike every other view): templates #9 never
-- search `places` by name (they filter by category and join spatially), the
-- multilingual name_* columns would only ever be displayed, never queried.
-- ─────────────────────────────────────────────────────────────
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

SELECT 'llaici views ready: divisions, division_areas, infrastructures, water, places' AS status;
