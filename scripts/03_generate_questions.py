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
import random
import unicodedata

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

# Grammatical gender of each country's FR name, keyed by ISO 3166-1 alpha-2 code
# (`place_country_code`, set by 01_sample_entities.py when the place *is* a country):
# "m" -> le Maroc, "f" -> la France, "pl" -> les Pays-Bas, None -> no article (Cuba,
# Malte, Israël...). "le"/"la" elide before a vowel (l'Espagne, l'Afghanistan), so
# there's no separate "l'" entry. Hand-written: Overture has no gender data. A code
# missing here gets no article, same as None. A few depend on which FR name Overture
# uses: CD/CG (République ... du Congo -> f), MM (Birmanie -> f, Myanmar -> m).
FR_COUNTRY_GENDER = {
    "AD": "f", "AE": "pl", "AF": "m", "AG": None, "AI": None, "AL": "f", "AM": "f",
    "AO": "m", "AQ": "m", "AR": "f", "AS": "pl", "AT": "f", "AU": "f", "AW": None,
    "AX": None, "AZ": "m", "BA": "f", "BB": "f", "BD": "m", "BE": "f", "BF": "m",
    "BG": "f", "BH": None, "BI": "m", "BJ": "m", "BL": None, "BM": "pl", "BN": "m",
    "BO": "f", "BQ": "pl", "BR": "m", "BS": "pl", "BT": "m", "BV": None, "BW": "m",
    "BY": "f", "BZ": "m", "CA": "m", "CC": "pl", "CD": "f", "CF": "f", "CG": "f",
    "CH": "f", "CI": "f", "CK": "pl", "CL": "m", "CM": "m", "CN": "f", "CO": "f",
    "CR": "m", "CU": None, "CV": "m", "CW": None, "CX": "f", "CY": None, "CZ": "f",
    "DE": "f", "DJ": None, "DK": "m", "DM": "f", "DO": "f", "DZ": "f", "EC": "m",
    "EE": "f", "EG": "f", "EH": "m", "ER": "f", "ES": "f", "ET": "f", "FI": "f",
    "FJ": "pl", "FK": "pl", "FM": "f", "FO": "pl", "FR": "f", "GA": "m", "GB": "m",
    "GD": "f", "GE": "f", "GF": "f", "GG": None, "GH": "m", "GI": None, "GL": "m",
    "GM": "f", "GN": "f", "GP": "f", "GQ": "f", "GR": "f", "GS": "f", "GT": "m",
    "GU": None, "GW": "f", "GY": "m", "HK": None, "HM": "pl", "HN": "m", "HR": "f",
    "HT": None, "HU": "f", "ID": "f", "IE": "f", "IL": None, "IM": "f", "IN": "f",
    "IO": "m", "IQ": "m", "IR": "m", "IS": "f", "IT": "f", "JE": None, "JM": "f",
    "JO": "f", "JP": "m", "KE": "m", "KG": "m", "KH": "m", "KI": None, "KM": "pl",
    "KN": None, "KP": "f", "KR": "f", "KW": "m", "KY": "pl", "KZ": "m", "LA": "m",
    "LB": "m", "LC": None, "LI": "m", "LK": "m", "LR": "m", "LS": "m", "LT": "f",
    "LU": "m", "LV": "f", "LY": "f", "MA": "m", "MC": None, "MD": "f", "ME": "m",
    "MF": None, "MG": None, "MH": "pl", "MK": "f", "ML": "m", "MM": "f", "MN": "f",
    "MO": None, "MP": "pl", "MQ": "f", "MR": "f", "MS": None, "MT": None, "MU": None,
    "MV": "pl", "MW": "m", "MX": "m", "MY": "f", "MZ": "m", "NA": "f", "NC": "f",
    "NE": "m", "NF": "f", "NG": "m", "NI": "m", "NL": "pl", "NO": "f", "NP": "m",
    "NR": None, "NU": None, "NZ": "f", "OM": None, "PA": "m", "PE": "m", "PF": "f",
    "PG": "f", "PH": "pl", "PK": "m", "PL": "f", "PM": None, "PN": None, "PR": None,
    "PS": "f", "PT": "m", "PW": "pl", "PY": "m", "QA": "m", "RE": None, "RO": "f",
    "RS": "f", "RU": "f", "RW": "m", "SA": "f", "SB": "pl", "SC": "pl", "SD": "m",
    "SE": "f", "SG": None, "SH": None, "SI": "f", "SJ": "m", "SK": "f", "SL": "f",
    "SM": None, "SN": "m", "SO": "f", "SR": "m", "SS": "m", "ST": None, "SV": "m",
    "SX": None, "SY": "f", "SZ": "m", "TC": "pl", "TD": "m", "TF": "pl", "TG": "m",
    "TH": "f", "TJ": "m", "TK": None, "TL": "m", "TM": "m", "TN": "f", "TO": None,
    "TR": "f", "TT": None, "TV": None, "TW": None, "TZ": "f", "UA": "f", "UG": "m",
    "UM": "pl", "US": "pl", "UY": "m", "UZ": "m", "VA": "m", "VC": None, "VE": "m",
    "VG": "pl", "VI": "pl", "VN": "m", "VU": "m", "WF": None, "WS": None, "XK": "m",
    "YE": "m", "YT": None, "ZA": "f", "ZM": "f", "ZW": "m",
}
# No "y": le Yémen. No "h" either: h muet vs aspiré can't be told from the spelling
# (l'Hérault / le Honduras), and "de Hambourg" is never wrong the way "d'Honduras" is.
FR_VOWELS = set("aeiouâàäéèêëîïôöûüœæ")
# de/à + le/les contract (du Maroc, aux Pays-Bas, au Havre); + la/l' don't.
FR_CONTRACTIONS = {("de", "le"): "du", ("de", "les"): "des", ("à", "le"): "au", ("à", "les"): "aux"}


def fr_article(place: str, params: dict) -> tuple[str, str]:
    """Splits the FR place into (article, rest), article lowercase, "" if none.

    Countries get theirs from FR_COUNTRY_GENDER, only when `place` is the FR exonym
    (not a kept local name like "España"). A city whose name starts with "Le "/"Les "
    ("Le Havre") has its article split off too, so it contracts ("au Havre") —
    "La"/"L'" never contract, so those names are left whole ("de La Rochelle")."""
    code = params.get("place_country_code")
    if code is not None:
        gender = FR_COUNTRY_GENDER.get(code)
        if gender is None or not params.get("place_is_exonym"):
            return "", place
        if gender == "pl":
            return "les", place
        if place[:1].lower() in FR_VOWELS:
            return "l'", place
        return ("le" if gender == "m" else "la"), place
    for article in ("Le", "Les"):
        if place.startswith(article + " "):
            return article.lower(), place[len(article) + 1:]
    return "", place


def fr_place(place: str, params: dict) -> str:
    """Bare FR place, with its article when it's a country ("l'Espagne", "Lyon")."""
    if params.get("place_country_code") is None:
        return place  # "montre-moi Le Havre": the city's own article stays capitalized
    article, rest = fr_article(place, params)
    if not article:
        return rest
    return f"{article}{rest}" if article == "l'" else f"{article} {rest}"


def fr_with_prep(prep: str, place: str, params: dict) -> str:
    """ "de"/"à" + FR place, contracted/elided: du Maroc, de la France, de l'Espagne,
    des Pays-Bas, d'Orléans, au Havre, à Lyon."""
    article, rest = fr_article(place, params)
    if (prep, article) in FR_CONTRACTIONS:
        return f"{FR_CONTRACTIONS[(prep, article)]} {rest}"
    if article:
        return f"{prep} {fr_place(place, params)}"
    if prep == "de" and place[:1].lower() in FR_VOWELS:
        return f"d'{place}"
    return f"{prep} {place}"


PHRASES = {
    "containment": {
        "en": ["{category} in {place}", "{category} located in {place}"],
        "fr": ["{category} {a_place}", "{category} situés {a_place}"],
    },
    "proximity": {
        "en": ["{category} near {place}", "{category} close to {place}", "{category} in the vicinity of {place}"],
        "fr": ["{category} près {de_place}", "{category} à proximité {de_place}", "{category} pas loin {de_place}"],
    },
    "along": {
        "en": ["{category} along {feature}", "{category} along the {feature}"],
        "fr": ["{category} le long de {feature}"],
    },
    "center": {
        "en": ["{category} in the center of {place}", "{category} in downtown {place}"],
        "fr": ["{category} dans le centre {de_place}", "{category} en centre-ville {de_place}"],
    },
    "periphery": {
        "en": ["{category} on the outskirts of {place}", "{category} at the edge of {place}"],
        "fr": ["{category} en périphérie {de_place}", "{category} en bordure {de_place}"],
    },
    "direction": {
        "en": ["{category} {direction_en} of {place}"],
        "fr": ["{category} {direction_fr} {de_place}"],
    },
    "direction_distance": {
        "en": ["{category} {distance} {direction_en} of {place}"],
        "fr": ["{category} à {distance} {direction_fr} {de_place}"],
    },
    "area_distance": {
        "en": ["{category} within {distance} around {place}", "{category} within {distance} of {place}"],
        "fr": ["{category} à moins de {distance} autour {de_place}", "{category} à moins de {distance} {de_place}"],
    },
    "left_right_bank": {
        "en": ["{category} on the {side} bank of {feature}"],
        "fr": ["{category} rive {side_fr} de {feature}"],
    },
    "bordering": {
        # No {category}: the entity is always "cities" (divisions, subtype='locality')
        # for this template — see TEMPLATES.md "10. Bordering a place".
        "en": ["cities bordering {place}", "cities near the border of {place}", "cities close to the {place} border"],
        "fr": ["villes frontalières {de_place}", "villes proches de la frontière {de_place}", "villes à la frontière {de_place}"],
    },
    "show_division": {
        # No {category}: this template resolves a place name to its geometry, it
        # doesn't search for entities within/near it — see TEMPLATES.md "11. Show a division".
        "en": ["show me {place}", "where is {place}", "show {place} on the map"],
        "fr": ["montre-moi {place}", "où {se_trouve} {place}", "affiche {place} sur la carte"],
    },
    "city_direction": {
        # No {category}: the entity is always "cities" (divisions, subtype='locality')
        # for this template, like "bordering" — see TEMPLATES.md "12. Cities in a direction".
        "en": ["cities {direction_en} of {place}"],
        "fr": ["villes {direction_fr} {de_place}"],
    },
    "city_direction_distance": {
        "en": ["cities {distance} {direction_en} of {place}"],
        "fr": ["villes à {distance} {direction_fr} {de_place}"],
    },
}
# `places` templates reuse the same phrasing shapes as their `infrastructures`
# counterparts ("restaurants in X" / "restaurants near X" read the same as
# "bus stops in X" / "bus stops near X") — no new sentence structure needed.
PHRASES["places_containment"] = PHRASES["containment"]
PHRASES["places_proximity"] = PHRASES["proximity"]
PHRASES["places_direction"] = PHRASES["direction"]
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


def build_questions(template: str, params: dict, lang: str) -> list[str]:
    """Returns the NL question strings in `lang` ("en" or "fr") for one validated pair.

    One language only: 02_fill_and_validate.py writes one row per (entity, language),
    whose `place`/`feature` is already the name a user of that language would type
    ("Espagne" vs "Spain"), and whose SQL matches on that same name."""
    category_en, category_fr = category_labels(params)
    fields_en = {"category": category_en}
    fields_fr = {"category": category_fr}

    place = params.get("place")
    feature = params.get("feature")
    if place is not None:
        fields_en["place"] = place
        fields_fr["place"] = fr_place(place, params)
        fields_fr["de_place"] = fr_with_prep("de", place, params)
        fields_fr["a_place"] = fr_with_prep("à", place, params)
        # "où se trouvent les Pays-Bas" — plural countries only; a city named
        # "Les Sables-d'Olonne" is one place, so it stays singular.
        plural = params.get("place_country_code") is not None and fr_article(place, params)[0] == "les"
        fields_fr["se_trouve"] = "se trouvent" if plural else "se trouve"
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

    fields = fields_en if lang == "en" else fields_fr
    return [p.format(**fields) for p in PHRASES[template][lang]]


def strip_accents(text: str) -> str:
    """"villes près de Séville" -> "villes pres de Seville" (also ñ -> n, ç -> c...)."""
    return "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))


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
    parser.add_argument(
        "--no-accent-ratio",
        type=float,
        default=0.3,
        help="share of accented questions that also get an accent-less copy, same SQL "
        "(users often type 'pres de Seville'; the SQL's strip_accents match finds 'Séville' anyway)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    # Collect (template, question, sql) before writing, so dedup/balance (DESIGN.md STEP 4
    # "Decisions") can run on the full set — both need `template`, which the final
    # {question, sql} shape doesn't carry.
    by_template: dict[str, list[tuple[str, str]]] = {}
    seen: set[tuple[str, str]] = set()
    duplicates = 0
    no_accent_variants = 0
    with open(args.in_path, encoding="utf-8") as fin:
        for line in fin:
            row = json.loads(line)
            for question in build_questions(row["template"], row["params"], row["lang"]):
                variants = [question]
                plain = strip_accents(question)
                if plain != question and rng.random() < args.no_accent_ratio:
                    variants.append(plain)
                    no_accent_variants += 1
                for variant in variants:
                    pair = (variant, row["sql"])
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

    print(f"# {no_accent_variants} accent-less question variants added (--no-accent-ratio {args.no_accent_ratio})")
    print(f"# {pairs_written} (question, sql) pairs written to {args.out_path} "
          f"({duplicates} exact duplicates dropped, {capped} dropped by --max-per-template)")


if __name__ == "__main__":
    main()
