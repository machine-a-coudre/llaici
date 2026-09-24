# LLaIci GeoAI model

### 🎯 Objective

Build a geocoding service that turns a natural language question (FR + EN) into a valid geometry (GeoJSON), relying on:

A small fine-tuned LLM (< 1B parameters) that generates spatial SQL.

Open geographic data (Overture, OSM, Natural Earth, WOF…)

An embedded SQL engine (DuckDB + spatial extension)

Multilingual fuzzy matching to find entities

No traditional database (everything in memory, deployable on CPU)

Packaged as a lightweight Docker container (~4 GB).

## 🏗️ System Architecture (runtime)

```text
User types a question (FR or EN)
        │
        ▼
┌─────────────────────────────────────────────┐
│ STEP 1 — Extract place names                │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│ STEP 2 — Multilingual fuzzy matching        │
│ (name, name_fr, name_es, name_it…)          │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│ STEP 3 — Load into memory (DuckDB)          │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│ STEP 4 — Generate SQL                       │
│ → 🤖 THE LLM (Qwen 0.8B fine-tuned)         │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│ STEP 5 — Execute SQL (DuckDB)               │
└─────────────────────────────────────────────┘
        │
        ▼
   Geometry (GeoJSON) returned

````

Key point: the LLM only intervenes at step 4. Everything else is deterministic.

## 📋 The 6 Construction Steps

### STEP 1 — Prepare geographic data
Download Overture Maps (Divisions + Places) or OSM, Natural Earth, Who's on First

Convert to GeoParquet

Flatten multilingual columns from names.common → name_fr, name_es, name_it…

Load into DuckDB with the spatial extension

⚠️ Logical view layer (schema stability + zero-install mode):
Instead of committing the LLM to Overture's raw nested schema (names.common structs, per-theme/type S3 partitions, which can change between Overture releases), define a fixed set of logical DuckDB views (`places`, `divisions`, `rivers`, `lakes`…) that expose flat, stable columns (`name`, `name_fr`, `name_es`, `geometry`…). The LLM only ever generates SQL against these view names — never against raw Overture columns.

These views can point at two different backends without changing a single line of generated SQL:

- Local mode: view built on a locally downloaded/ingested GeoParquet table.
- Direct-Overture mode: view built on `read_parquet('s3://overturemaps-us-west-2/release/.../theme=places/type=place/*')` via `httpfs`, with the multilingual flattening expressed inline in the view definition (e.g. `map_extract(names.common, 'fr')` — exact extraction syntax must be validated against the current Overture release schema, since `names.common`'s structure has changed between releases).

```sql
-- schema.sql (the contract) — same view names/columns in both modes
CREATE OR REPLACE VIEW places AS
SELECT
    id,
    names.primary                     AS name,
    map_extract(names.common, 'fr')   AS name_fr,
    map_extract(names.common, 'es')   AS name_es,
    categories.primary                AS category,
    geometry
FROM read_parquet('s3://overturemaps-us-west-2/release/.../theme=places/type=place/*');
```

This lets users who don't want to build/host a local database run the exact same generated SQL directly against Overture on S3 (no ingestion, near-zero storage — see "🔌 Local vs Overture" below), while users who want speed/frequent-query performance can materialize the same views locally. Ship both view variants (`schema.local.sql` / `schema.overture.sql`) as part of the model distribution.

Trade-off to flag to users: direct-Overture mode re-scans S3 on every query (network-dependent latency, more data transfer) vs. local mode's fast repeated queries — same trade-off already described in "🔌 Local vs Overture: Two Connection Modes".

Deliverable: DuckDB database with entities + multilingual names, exposed through the logical view layer (local and/or direct-Overture variant).

### STEP 2 — ~~Build the spatial relations graph~~ (skipped)
Originally planned: precompute `ST_Touches, ST_Contains, ST_Within, ST_Intersects, ST_Overlaps` for entity pairs and store them in a relations table, to be reused when filling SQL templates in STEP 4 (now STEP 3).

⚠️ **Why skipped**: a persistent relations graph only pays for itself when a relation is expensive to compute per-pair and reused often. The one relation that fit that bar — `left_bank`/`right_bank` (see TEMPLATES.md "Deferred to STEP 2") — turned out to be cheap after all: following the same approach as [geoblocks/etter](https://github.com/geoblocks/etter), we trust the waterway geometry's digitization order (OSM convention: ways should be drawn in flow direction) and use `offset_curve` directly, no reorientation step needed. Every other relation (containment, proximity, direction) is already computed live in the STEP 3 templates. With no relation left that justifies precomputing, entity pairs for STEP 4 dataset generation are sampled directly via live spatial joins on the logical views (generate → execute → discard if empty), rather than pre-materialized in a separate table. This removes a whole deliverable and its maintenance cost.

Step number kept as a placeholder so the rest of the pipeline's numbering stays stable.

No deliverable for this step.

### STEP 3 — Create SQL templates
Cover 3 dimensions:

Spatial relations: in, near, north of, along, within…

Locations: countries, cities, lakes, rivers, mountains

Distances: 1km, 5 miles, 100m… (converted to meters)

⚠️ Multilingual: the LLM does NOT translate. If the user types "Lac Léman", the SQL searches for "Lac Léman" in all name columns (name, name_fr, name_es, name_it).

⚠️ Target the logical views, not raw tables: templates must reference the fixed view names from STEP 1 (`places`, `divisions`, `rivers`, `lakes`…) and their flat columns (`name`, `name_fr`, `geometry`…) — never a specific backend's raw schema (raw Overture struct columns, OSM tags, etc.). This is what keeps a single generated SQL valid in both local mode and direct-Overture (S3) mode, and insulates the fine-tuned model from schema/version drift in the underlying data source.

Deliverable: parameterized SQL templates, written against the logical view layer.

### STEP 4 — Generate the training dataset
Method (inverting the problem):

Fill templates with real entities, sampled directly via live spatial joins on the logical views (no precomputed graph — see STEP 2)

Execute the SQL on DuckDB, against the logical views (STEP 1) — not raw backend tables — so the (question, SQL) pairs stay valid whether the end user runs them locally or directly against Overture on S3

Empty result → discard the pair

Non-empty result → keep only (question, SQL)

Generate multiple formulations (FR + EN)

⚠️ Critical nuance:

The geometric result is used only for validation

It is NEVER included in the dataset

The dataset contains only {question, sql}

Format:

json
{"question": "cities within 5km of Lake Sanabria", "sql": "SELECT ... WHERE name ILIKE '%Lake Sanabria%' OR name_es ILIKE '%Lago de Sanabria%'"}
{"question": "villes à moins de 5km du Lac de Sanabria", "sql": "SELECT ... WHERE name ILIKE '%Lac de Sanabria%' OR name_es ILIKE '%Lago de Sanabria%'"}
Deliverable: dataset.jsonl (~10k-70k validated pairs).

**Decisions on how STEP 4 samples and assembles the dataset:**

- **Sampling strategy**: join-driven, not random-then-discard. Start from a real entity and *derive* its context through the same spatial join the template needs (e.g. take a real bus stop, then find the real nearest city via `ST_Within`/`ST_DWithin`) — the pair is valid by construction. "Discard if empty" stays as a safety net, not the primary mechanism. Rejected: picking `(subtype, place)` combos purely at random, which has a low hit rate for narrow templates (many combos have zero real matches).
- **Geographic diversity**: stratify by country first — pick a random `divisions` row with `subtype = 'country'` (within the EU bbox), then sample an entity inside that country — rather than sampling directly from `infrastructures`, which mechanically over-represents whichever countries/cities have denser Overture data (Paris/Barcelona kept coming up in ad hoc testing precisely because of this).
- **Performance at generation scale**: the join-driven sampling above does the same kind of join against `water`/`division_areas` that the inference-time templates do, so it reuses the same bounding-box prefilter documented in `TEMPLATES.md` (`ST_Envelope` + `ST_Expand` + `ST_Intersects` before the exact spatial test) — for the same reason: without it, a real query against the full dataset OOM'd (19+ GiB).
  - Found while implementing `scripts/01_sample_entities.py`: the bbox prefilter alone isn't enough when the *reference* geometry itself is unbounded in practice — picking a random river segment for the "along"/left-right-bank samplers hit a case where a single `LINESTRING` spans a very long, unsplit stretch of river, so its own bbox (even before expansion) already covers a huge area, defeating the prefilter (a `left_right_bank` sampling run ran for 3+ minutes at 200% CPU before being killed). Fix: cap candidate reference segments to `ST_Length_Spheroid(geometry) < 50000` (50km) when picking one to sample from, so the bbox stays bounded regardless of how a given river was digitized. Also dropped `river` from the area-based-distance (`#8`) sampler's candidate classes for the same reason — its intent (`TEMPLATES.md` "around Lake Geneva") is bounded features like lakes, not open-ended linear ones.
  - Note this is a sampling-script-specific fix, not a template correction: the inference-time templates (`along`, `#8`, left/right bank) always match a *specific named* feature via `ILIKE '%{feature}%'` (one real river the user asked about, whatever its length), whereas the sampler was picking *any* random segment to build a training example from — that's what needed the extra length cap.
- **Multi-formulation generation (FR + EN)**: hand-written phrasing templates per relation type and language (e.g. proximity EN: "near", "close to", "in the vicinity of"; FR: "près de", "à proximité de", "pas loin de"), picked at random per generated pair. Rejected: paraphrasing via a separate LLM call — non-deterministic, and would break the $0 dataset-generation cost stated in "💻 Hardware and Costs" below.
- **Deduplication and balancing**: dedupe on the exact `(question, sql)` string pair (no fuzzy similarity needed, since formulations come from a fixed list rather than free-form generation). Balance by capping the max pairs per template (so no single easy-to-satisfy template dominates the dataset) and keeping FR/EN roughly 50/50 (naturally the case if each validated pair generates one formulation per language).

### STEP 5 — Fine-tune the model
~~Qwen 3.5 0.8B~~ (recommended) or Gemma 3 270M

⚠️ **Correction**: "Qwen 3.5 0.8B" isn't a real released model (no "Qwen 3.5" family, no official "0.8B" size exists) — see `FINETUNING.md` §2 for the verified real substitute chosen instead (`unsloth/Qwen3-0.6B-unsloth-bnb-4bit`).

LoRA + QLoRA 4-bit via Unsloth

2 epochs, rank 16-32, LR 2e-4

Quantize to GGUF Q8 (~800 MB)

Serve via llama-server

⚠️ Concept: fine-tuning anchors the model to your schema. It does not add geographic knowledge.

Deliverable: GGUF Q8 model.

### STEP 6 — Deploy the service
GGUF model + DuckDB + GeoParquet + Python pipeline + llama-server

Docker ~4 GB, runs on CPU

No persistent database

Deliverable: Dockerfile + scripts.

⚠️ Critical Points of Attention
Point	Detail
The LLM does not translate	It searches the term as-is in all name columns
No relations graph	Skipped (STEP 2) — entity pairs sampled live instead
Result is not in the dataset	Used only for validation
Multilingual	Generate questions in FR + EN at step 4
Performance	Flatten JSON columns at ingestion, index names
🔌 Local vs Overture: Two Connection Modes
Local database	Direct Overture (S3)
Data	Downloaded and imported	Read directly from S3
Schema	You define it	Fixed (Overture's)
Speed	Fast (local)	Depends on network
Storage	Disk space required	Almost none
Usage	Frequent queries	Occasional exploration
DuckDB reads GeoParquet directly from S3:

sql
LOAD spatial;
LOAD httpfs;
SET s3_region = 'us-west-2';

SELECT names.primary, geometry
FROM read_parquet('s3://overturemaps-us-west-2/release/.../theme=places/type=place/*')
WHERE names.primary ILIKE '%wawa%'
LIMIT 10;
📦 Model Distribution
You distribute a (model, schema) pair — inseparable.

text
📦 llaici-model/
├── model.gguf              # Fine-tuned model
├── schema.sql              # Exact schema (CONTRACT)
├── ingestion/              # Database build scripts
├── README.md               # Documentation
└── examples/               # Examples
The user must build their database with YOUR schema

If they respect the schema → your model works

If they change the schema → they must re-fine-tune

They can use OSM instead of Overture → as long as the schema is identical

## 💻 Hardware and Costs
On a consumer CUDA GPU
Step	Time	Cost
Dataset generation (70k pairs)	2-4 h	$0
Fine-tuning (Qwen 0.8B, QLoRA)	30 min - 1 h	$0
Total	~3-5 h	$0
Cloud is not worth it for this project: an A100 is ~3x more powerful, but for a 0.8B model, the real gain is ~30%. Keep your $15.

## 🔄 Recommended Execution Order
Start small: 1 source, 5 templates, 1000 pairs

Validate the full pipeline

Add progressively: templates, entities, languages, distances

Balance and deduplicate the dataset

Fine-tune and quantize

Package in Docker

Test on varied questions (FR + EN)

## 💡 Possible Fuzzy Matching Improvement
Replace Jaro-Winkler with a small BERT embedding:

Criterion	Jaro-Winkler	BERT embedding
Comparison	Characters	Meaning (vectors)
"Marrakech" vs "مراكش"	Low score ❌	High score ✅
Multilingual	❌	✅
Speed	Very fast	Slower
Size	0	~50-100 MB
→ More robust for translations/synonyms, at the cost of some CPU.

## 📍 `places` theme — not yet ingested, findings for later

Answering a question like "restaurants in Madrid" needs POI data (restaurants, shops, hotels...), which none of the 4 currently-ingested extracts (`division`, `division_area`, `infrastructure`, `water` — see `Makefile`) provide. That's Overture's **`places`** theme (`type=place`), still to be downloaded/wired into `00_init.sql`/`SCHEMA.md`. Explored via a small test download (bbox around Geneva, 16,030 rows, cleaned up afterward — not committed to the repo) to check the real schema before implementing:

- **Schema**: `id`, `geometry` (always a point), `names` (same `names.primary`/`names.common['fr']` shape as the other views — reusable as-is), `confidence` (0-1 score), `operating_status` (open/closed temporarily/permanently closed), plus commercial metadata (`brand`, `addresses`, `websites`, `socials`, `emails`, `phones`) likely out of scope for simple NL geocoding.

- **Category field: `categories` and `taxonomy` currently coexist and hold identical values.** The docs describe `categories.primary/alternate` as superseded by `taxonomy.primary/hierarchy/alternates`, but the actual downloaded release still ships both, in sync. Not an immediate conflict, but another instance of the schema-drift-between-releases risk already flagged for `names.common` — worth re-checking whichever field is used at ingestion time, in case a future release drops `categories` entirely.

- ⚠️ **Pitfall found, relevant to any future `places` template**: `taxonomy.primary` is the *most specific* category (e.g. `italian_restaurant`, `sushi_restaurant`), not a general one — a generic-sounding filter like `taxonomy.primary = 'restaurant'` only matched **457** of the 1,407 actual restaurants in the Geneva test extract, silently missing every typed cuisine (Italian, sushi, pizza, etc.). The correct filter checks the full category hierarchy instead: `list_contains(taxonomy.hierarchy, 'restaurant')`, which correctly returned all 1,407. Same trap that hit the STEP 3 templates before real-data testing (see `TEMPLATES.md`) — any generic-sounding category query on `places` must go through `hierarchy`, never `taxonomy.primary` alone. Note `fast_food_restaurant`'s hierarchy is `['food_and_drink', 'casual_eatery', 'fast_food_restaurant']` — it skips the `restaurant` node entirely, so a decision is needed on whether "restaurants" should include fast food.

- **Top-level categories found** (`taxonomy.hierarchy[1]`, Geneva test extract, ~16k places): `services_and_business` (3,886), `shopping` (2,443), `food_and_drink` (2,363), `health_care` (1,372), `lifestyle_services` (1,320), `community_and_government` (933), `education` (655), `travel_and_transportation` (598), `sports_and_recreation` (591), `arts_and_entertainment` (550), `cultural_and_historic` (287), `lodging` (165), `geographic_entities` (97). Only a subset of these (`restaurant`/`lodging`/`hotel`/`school`/`hospital`/`shopping_mall`/`grocery_store`) has a template so far — see `TEMPLATES.md` "9. Places by category" — plenty of room to extend later (bars, cafés, pharmacies, other shop types, education beyond schools...).

**Status update**: the `places` view is now *defined* in `00_init.sql` (alongside the other views, downloaded as part of `make download-overture`) and documented (`SCHEMA.md` §5), filtered to `restaurant`/`lodging`/`school`/`hospital`/`shopping_mall`/`grocery_store` for the same reason `infrastructures` is filtered — `places` is far denser than the other themes (a test extract was 24% generic `services_and_business` alone). `confidence`/`operating_status` were dropped from the view (not used by any template, kept things minimal) rather than wired up as quality filters. Template #9 (containment + proximity) has been sampled and validated against real data (`data/db/llaici.duckdb`), same as every other template.

## 🗺️ Alternatives to Overture
Source	Advantages	Challenges
OpenStreetMap	Detailed global coverage	Raw data, more ingestion work
Who's on First	Clean structure, stable IDs	No physical data
Natural Earth	Lightweight, clean, multilingual	Less detailed in urban areas
GeoNames	Good for names/translations	Few geometries (points)
Recommendation: Natural Earth + Who's on First to start.

## 🎯 In One Sentence
Build a dataset of (question, SQL) pairs starting from structured geography (templates + live-sampled entities), validate each pair by SQL execution, fine-tune a small LLM to generate spatial SQL, and deploy in a lightweight Docker container running on CPU.

## 📚 Resources
Overture Maps: open geo data, 40 languages in names.common

Natural Earth: 26 languages (name_fr, etc.)

DuckDB + spatial: embedded SQL engine, ST_* functions

GeoParquet: optimized columnar format

Unsloth: fast and VRAM-efficient LoRA fine-tuning

llama.cpp / llama-server: CPU inference with GGUF