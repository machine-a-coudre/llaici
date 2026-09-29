#!/usr/bin/env python3
"""STEP 4, step 4 — generate NL question formulations (FR + EN) for validated pairs.

Reads `data/samples/validated_samples.jsonl` (produced by `fill_and_validate.py`) and, for
each validated (template, params, sql) row, generates several natural-language
question formulations in French and English by filling hand-written phrase
templates — one relation type, one set of phrasings, per language (DESIGN.md STEP 4
"Decisions": hand-written templates, not LLM paraphrasing, to stay deterministic
and $0 cost).

Before writing, pairs are deduplicated on the exact (question, sql) string pair and,
if --max-per-template is given, capped per template so no single relation type
dominates the dataset (DESIGN.md STEP 4 "Decisions"). Output is the DESIGN.md STEP 4
deliverable shape: one {question, sql} pair per line, written to
`data/samples/dataset.jsonl`. The geometric result and every other field (template,
params, country) are dropped here — DESIGN.md: "The dataset contains only
{question, sql}".

Run:
    uvx --with duckdb python3 scripts/03_generate_questions.py
    uvx --with duckdb python3 scripts/03_generate_questions.py --in data/samples/validated_samples.jsonl --out data/samples/dataset.jsonl --max-per-template 5000
"""

import argparse
import json

# Human-readable (EN, FR) plural labels for the (subtype, class) categories kept in
# the `infrastructures` view (see scripts/00_init.sql). Fallback for anything missing:
# the raw "subtype class" string with underscores turned into spaces.
CATEGORY_LABELS = {
    ("transit", "parking"): ("parking areas", "parkings"),
    ("transit", "parking_space"): ("parking spaces", "places de parking"),
    ("bridge", "bridge"): ("bridges", "ponts"),
    ("transit", "bus_stop"): ("bus stops", "arrêts de bus"),
    ("transit", "bicycle_parking"): ("bicycle parking areas", "parkings à vélos"),
    ("transit", "railway_station"): ("railway stations", "gares"),
    ("transit", "subway_station"): ("subway stations", "stations de métro"),
    ("transit", "railway_halt"): ("railway halts", "haltes ferroviaires"),
    ("transit", "bus_station"): ("bus stations", "gares routières"),
    ("aerialway", "aerialway_station"): ("aerialway stations", "stations de télécabine"),
    ("pedestrian", "toilets"): ("public toilets", "toilettes publiques"),
    ("waste_management", "waste_basket"): ("waste baskets", "poubelles"),
    ("waste_management", "recycling"): ("recycling points", "points de recyclage"),
    ("waste_management", "waste_disposal"): ("waste disposal points", "points de collecte des déchets"),
    ("airport", "airport"): ("airports", "aéroports"),
    ("airport", "international_airport"): ("international airports", "aéroports internationaux"),
    ("airport", "regional_airport"): ("regional airports", "aéroports régionaux"),
    ("airport", "municipal_airport"): ("municipal airports", "aéroports municipaux"),
    ("airport", "military_airport"): ("military airports", "aéroports militaires"),
    ("airport", "private_airport"): ("private airports", "aéroports privés"),
    ("airport", "heliport"): ("heliports", "héliports"),
    ("airport", "helipad"): ("helipads", "hélisurfaces"),
    ("airport", "runway"): ("runways", "pistes d'atterrissage"),
    ("airport", "taxiway"): ("taxiways", "voies de circulation"),
    ("airport", "taxilane"): ("taxilanes", "taxilanes"),
    ("airport", "apron"): ("aprons", "aires de trafic"),
    ("airport", "airport_gate"): ("airport gates", "portes d'embarquement"),
    ("airport", "airstrip"): ("airstrips", "pistes sommaires"),
    ("airport", "stopway"): ("stopways", "prolongements d'arrêt"),
    ("communication", "communication_tower"): ("communication towers", "tours de communication"),
    ("communication", "mobile_phone_tower"): ("mobile phone towers", "antennes-relais"),
    ("communication", "communication_pole"): ("communication poles", "poteaux de communication"),
    ("communication", "communication_line"): ("communication lines", "lignes de communication"),
}

# Human-readable (EN, FR) plural labels for `places` categories (see
# scripts/init_places.sql, TEMPLATES.md "9. Places by category"). Keyed by a single
# `category` string, unlike CATEGORY_LABELS' (subtype, class) tuples, since `places`
# has no subtype/class split.
PLACE_CATEGORY_LABELS = {
    "restaurant": ("restaurants", "restaurants"),
    "lodging": ("places to stay", "hébergements"),
    "hotel": ("hotels", "hôtels"),
    "school": ("schools", "écoles"),
    "hospital": ("hospitals", "hôpitaux"),
    "shopping_mall": ("shopping malls", "centres commerciaux"),
    "grocery_store": ("grocery stores", "épiceries"),
}

DIRECTION_EN = {"north": "north", "south": "south", "east": "east", "west": "west"}
DIRECTION_FR = {"north": "au nord", "south": "au sud", "east": "à l'est", "west": "à l'ouest"}
# "de"/"du"/"de la" agreement with the place name's gender can't be determined from
# data alone — "de" is used uniformly, which reads slightly informally in French
# but is unambiguous and never grammatically wrong the way "du"/"de la" could be.

PHRASES = {
    "containment": {
        "en": ["{category} in {place}", "{category} located in {place}"],
        "fr": ["{category} à {place}", "{category} situés à {place}"],
    },
    "proximity": {
        "en": ["{category} near {place}", "{category} close to {place}", "{category} in the vicinity of {place}"],
        "fr": ["{category} près de {place}", "{category} à proximité de {place}", "{category} pas loin de {place}"],
    },
    "along": {
        "en": ["{category} along {feature}", "{category} along the {feature}"],
        "fr": ["{category} le long de {feature}"],
    },
    "center": {
        "en": ["{category} in the center of {place}", "{category} in downtown {place}"],
        "fr": ["{category} dans le centre de {place}", "{category} en centre-ville de {place}"],
    },
    "periphery": {
        "en": ["{category} on the outskirts of {place}", "{category} at the edge of {place}"],
        "fr": ["{category} en périphérie de {place}", "{category} en bordure de {place}"],
    },
    "direction": {
        "en": ["{category} {direction_en} of {place}"],
        "fr": ["{category} {direction_fr} de {place}"],
    },
    "direction_distance": {
        "en": ["{category} {distance} {direction_en} of {place}"],
        "fr": ["{category} à {distance} {direction_fr} de {place}"],
    },
    "area_distance": {
        "en": ["{category} within {distance} around {place}", "{category} within {distance} of {place}"],
        "fr": ["{category} à moins de {distance} autour de {place}", "{category} à moins de {distance} de {place}"],
    },
    "left_right_bank": {
        "en": ["{category} on the {side} bank of {feature}"],
        "fr": ["{category} rive {side_fr} de {feature}"],
    },
    "bordering": {
        # No {category}: the entity is always "cities" (divisions, subtype='locality')
        # for this template — see TEMPLATES.md "10. Bordering a place".
        "en": ["cities bordering {place}", "cities near the border of {place}", "cities close to the {place} border"],
        "fr": ["villes frontalières de {place}", "villes proches de la frontière de {place}", "villes à la frontière de {place}"],
    },
    "show_division": {
        # No {category}: this template resolves a place name to its geometry, it
        # doesn't search for entities within/near it — see TEMPLATES.md "11. Show a division".
        "en": ["show me {place}", "where is {place}", "show {place} on the map"],
        "fr": ["montre-moi {place}", "où se trouve {place}", "affiche {place} sur la carte"],
    },
    "city_direction": {
        # No {category}: the entity is always "cities" (divisions, subtype='locality')
        # for this template, like "bordering" — see TEMPLATES.md "12. Cities in a direction".
        "en": ["cities {direction_en} of {place}"],
        "fr": ["villes {direction_fr} de {place}"],
    },
    "city_direction_distance": {
        "en": ["cities {distance} {direction_en} of {place}"],
        "fr": ["villes à {distance} {direction_fr} de {place}"],
    },
}
# `places` templates reuse the same phrasing shapes as their `infrastructures`
# counterparts ("restaurants in X" / "restaurants near X" read the same as
# "bus stops in X" / "bus stops near X") — no new sentence structure needed.
PHRASES["places_containment"] = PHRASES["containment"]
PHRASES["places_proximity"] = PHRASES["proximity"]
SIDE_FR = {"left": "gauche", "right": "droite"}


def category_labels(params: dict) -> tuple[str, str]:
    if "category" in params:
        # `places` templates: a single category string (e.g. "restaurant"), not a
        # (subtype, class) pair.
        category = params["category"]
        if category in PLACE_CATEGORY_LABELS:
            return PLACE_CATEGORY_LABELS[category]
        fallback = category.replace("_", " ")
        return fallback, fallback
    key = (params.get("subtype"), params.get("class"))
    if key in CATEGORY_LABELS:
        return CATEGORY_LABELS[key]
    fallback = f"{params.get('subtype', '')} {params.get('class', '')}".replace("_", " ").strip()
    return fallback, fallback


def format_distance(meters: float) -> tuple[str, str]:
    """Returns (english, french) distance strings, e.g. (1500, 1500) -> ('1.5km', '1,5km')."""
    if meters >= 1000:
        km_str = f"{round(meters / 1000, 1):g}"
        return f"{km_str}km", f"{km_str.replace('.', ',')}km"
    m_str = f"{round(meters):g}"
    return f"{m_str}m", f"{m_str}m"


def build_questions(template: str, params: dict) -> list[str]:
    """Returns a flat list of NL question strings (EN + FR) for one validated pair."""
    category_en, category_fr = category_labels(params)
    fields_en = {"category": category_en}
    fields_fr = {"category": category_fr}

    place = params.get("place")
    feature = params.get("feature")
    if place is not None:
        fields_en["place"] = place
        fields_fr["place"] = place
    if feature is not None:
        fields_en["feature"] = feature
        fields_fr["feature"] = feature

    if "distance_m" in params:
        dist_en, dist_fr = format_distance(params["distance_m"])
        fields_en["distance"] = dist_en
        fields_fr["distance"] = dist_fr

    if "direction" in params:
        fields_en["direction_en"] = DIRECTION_EN[params["direction"]]
        fields_fr["direction_fr"] = DIRECTION_FR[params["direction"]]

    if "side" in params:
        fields_en["side"] = params["side"]
        fields_fr["side_fr"] = SIDE_FR[params["side"]]

    phrases = PHRASES[template]
    questions = [p.format(**fields_en) for p in phrases["en"]]
    questions += [p.format(**fields_fr) for p in phrases["fr"]]
    return questions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", default="data/samples/validated_samples.jsonl")
    parser.add_argument("--out", dest="out_path", default="data/samples/dataset.jsonl")
    parser.add_argument(
        "--max-per-template",
        type=int,
        default=None,
        help="cap pairs kept per template after dedup, so no single template dominates the dataset (default: no cap)",
    )
    args = parser.parse_args()

    # Collect (template, question, sql) before writing, so dedup/balance (DESIGN.md STEP 4
    # "Decisions") can run on the full set — both need `template`, which the final
    # {question, sql} shape doesn't carry.
    by_template: dict[str, list[tuple[str, str]]] = {}
    seen: set[tuple[str, str]] = set()
    duplicates = 0
    with open(args.in_path, encoding="utf-8") as fin:
        for line in fin:
            row = json.loads(line)
            for question in build_questions(row["template"], row["params"]):
                pair = (question, row["sql"])
                if pair in seen:
                    duplicates += 1
                    continue
                seen.add(pair)
                by_template.setdefault(row["template"], []).append(pair)

    capped = 0
    pairs_written = 0
    with open(args.out_path, "w", encoding="utf-8") as fout:
        for template, pairs in by_template.items():
            kept = pairs if args.max_per_template is None else pairs[: args.max_per_template]
            capped += len(pairs) - len(kept)
            for question, sql in kept:
                fout.write(json.dumps({"question": question, "sql": sql}, ensure_ascii=False) + "\n")
                pairs_written += 1

    print(f"# {pairs_written} (question, sql) pairs written to {args.out_path} "
          f"({duplicates} exact duplicates dropped, {capped} dropped by --max-per-template)")


if __name__ == "__main__":
    main()
