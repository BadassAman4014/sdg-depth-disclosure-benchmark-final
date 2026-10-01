#!/usr/bin/env python3
"""
sdg_keyword_match.py — Match passages against SDG keyword regex patterns and write to DuckDB.

Adapted from AI-For-Sustainability v1-full-pipeline branch (base_filter.py + sdg_filter.py).

Usage:
    python scripts/sdg_keyword_match.py [--force]

Reads:  data/jsons/<Company>/<Year>/splits_semantic.json
        AI-For-Sustainability/kw_data/keywords_sdg.json
        AI-For-Sustainability/kw_data/keywords_sdg_de.json
Writes: data/dbs/sdg_hits.duckdb
"""

import argparse
import json
import re
import logging
from pathlib import Path
from typing import Any

import duckdb
from langdetect import detect
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────────
JSONS_DIR   = Path("data/jsons")
KW_EN_PATH  = Path("AI-For-Sustainability/kw_data/keywords_sdg.json")
KW_DE_PATH  = Path("AI-For-Sustainability/kw_data/keywords_sdg_de.json")
OUT_DB      = Path("data/dbs/sdg_hits.duckdb")
TABLE_NAME  = "sdg_hits"
SDG_CATS    = [f"sdg{i}" for i in range(1, 18)]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("data/sdg_keyword_match.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ── Keyword loading & compilation ─────────────────────────────────────────────

def load_keywords(path: Path) -> dict[str, list[str]]:
    """Load keyword JSON: {category: [pattern_str, ...]} with Mojibake auto-healing."""
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        content = f.read()
    for bad, good in [("├ñ", "ä"), ("├╢", "ö"), ("├╝", "ü"), ("├ƒ", "ß"), ("├ä", "Ä"), ("├Ц", "Ö"), ("├Ь", "Ü"),
                      ("Ã¤", "ä"), ("Ã¶", "ö"), ("Ã¼", "ü"), ("ÃŸ", "ß")]:
        content = content.replace(bad, good)
    return json.loads(content)


def _is_alnum(c: str) -> bool:
    return bool(re.match(r"[A-Za-z0-9]", c))


def _token_to_regex(token: str, star_is_wildcard: bool = True) -> str:
    """
    Build a regex pattern from a keyword token.
    Matches v1 base_filter.py logic:
    - trailing '*' -> '\\w*'
    - '*' elsewhere -> '.*'
    - spaces -> '\\s+'
    - word boundaries \\b added if ends are alphanumeric
    """
    raw = token.strip()
    if not raw:
        return ""

    pieces = []
    for i, ch in enumerate(raw):
        if star_is_wildcard and ch == "*":
            if i == len(raw) - 1:
                pieces.append(r"\w*")
            else:
                pieces.append(r".*")
        elif star_is_wildcard and ch == " ":
            pieces.append(r"\s+")
        else:
            pieces.append(re.escape(ch))

    pat_body = "".join(pieces)

    first_vis = next((c for c in raw if c != " "), raw[:1])
    last_vis = next((c for c in reversed(raw) if c != "*"), "")
    if _is_alnum(first_vis):
        pat_body = r"\b" + pat_body
    if _is_alnum(last_vis):
        pat_body = pat_body + r"\b"

    return pat_body


def compile_keywords_per_pattern(kw_raw: dict[str, list[str]]) -> dict[str, list[tuple[re.Pattern, str]]]:
    """
    Returns {category: [(compiled_regex, original_pattern_string), ...]}.
    Each pattern is compiled individually so we can report which ones matched.
    """
    compiled = {}
    for cat, patterns in kw_raw.items():
        cat_patterns = []
        for pat in patterns:
            regex_str = _token_to_regex(pat, star_is_wildcard=True)
            if not regex_str:
                continue
            try:
                compiled_re = re.compile(regex_str, re.IGNORECASE)
                cat_patterns.append((compiled_re, regex_str))
            except re.error as e:
                log.warning(f"Invalid regex for '{pat}' → '{regex_str}': {e}")
        compiled[cat] = cat_patterns
    return compiled


def compile_keywords_combined(kw_raw: dict[str, list[str]]) -> dict[str, re.Pattern]:
    """
    Returns {category: single_combined_regex} for fast screening.
    """
    compiled = {}
    for cat, patterns in kw_raw.items():
        regex_parts = []
        for pat in patterns:
            regex_str = _token_to_regex(pat, star_is_wildcard=True)
            if regex_str:
                regex_parts.append(f"(?:{regex_str})")
        if regex_parts:
            try:
                compiled[cat] = re.compile("|".join(regex_parts), re.IGNORECASE)
            except re.error:
                pass
    return compiled


# ── Language detection ─────────────────────────────────────────────────────────

def detect_language(passages: list[str], n: int = 5) -> str:
    """Detect language from first N passages."""
    sample = " ".join(passages[:n])
    if not sample.strip():
        return "unknown"
    try:
        lang = detect(sample)
        return "de" if lang == "de" else "en" if lang == "en" else lang
    except Exception:
        return "unknown"


# ── Matching ──────────────────────────────────────────────────────────────────

def find_hits(
    text: str,
    combined_en: dict[str, re.Pattern],
    combined_de: dict[str, re.Pattern],
    per_pattern_en: dict[str, list[tuple[re.Pattern, str]]],
    per_pattern_de: dict[str, list[tuple[re.Pattern, str]]],
    is_german: bool,
) -> dict[str, list[str]]:
    """
    Returns {category: [matched_regex_string, ...]} for categories with hits.
    Uses combined regex for fast screening, then per-pattern for detail.
    """
    # Choose which keyword set to use
    combined = {**combined_en}
    per_pattern = {**per_pattern_en}
    if is_german:
        for cat in SDG_CATS:
            if cat in combined_de:
                combined.setdefault(cat, combined_de[cat])
            if cat in per_pattern_de:
                per_pattern.setdefault(cat, [])
                per_pattern[cat] = per_pattern_en.get(cat, []) + per_pattern_de.get(cat, [])

    hits = {}
    for cat in SDG_CATS:
        # Fast screen with combined regex
        comb = combined.get(cat)
        if comb is None or not comb.search(text):
            continue

        # Detailed: find which patterns matched
        patterns = per_pattern.get(cat, [])
        matched = []
        for compiled_re, regex_str in patterns:
            if compiled_re.search(text):
                matched.append(regex_str)
        if matched:
            hits[cat] = matched

    return hits


# ── DuckDB writer ────────────────────────────────────────────────────────────

def _json_str(lst: list) -> str:
    return json.dumps(lst, ensure_ascii=False)


def _clean_company_for_id(company: str) -> str:
    return re.sub(r"[^a-z0-9]", "", company.lower())


def create_db(db_path: Path) -> duckdb.DuckDBPyConnection:
    """Create/open DuckDB with the sdg_hits table."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))

    hit_cols = ", ".join(f"hits_sdg{i} VARCHAR" for i in range(1, 18))
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            global_id VARCHAR,
            passage VARCHAR,
            company VARCHAR,
            year VARCHAR,
            language VARCHAR,
            {hit_cols}
        )
    """)
    return con


# ── Main pipeline ────────────────────────────────────────────────────────────

def process_one_file(
    company: str,
    year: str,
    json_path: Path,
    combined_en: dict,
    combined_de: dict,
    per_pattern_en: dict,
    per_pattern_de: dict,
) -> list[tuple]:
    """Process one splits_semantic.json file. Returns list of row tuples."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    passages = list(data.values())
    is_german = detect_language(passages) == "de"
    language = "de" if is_german else detect_language(passages)

    company_clean = _clean_company_for_id(company)
    rows = []

    for sid_str, passage in sorted(data.items(), key=lambda x: int(x[0])):
        text = (passage or "").strip()
        if not text:
            continue

        hits = find_hits(text, combined_en, combined_de, per_pattern_en, per_pattern_de, is_german)
        if not hits:
            continue

        global_id = f"{year}{company_clean}{sid_str}"
        hit_cols = tuple(_json_str(hits.get(f"sdg{i}", [])) for i in range(1, 18))
        row = (global_id, text, company, year, language) + hit_cols
        rows.append(row)

    return rows


def discover_jsons(jsons_dir: Path) -> list[tuple[str, str, Path]]:
    """Discover all (company, year, json_path) triples."""
    result = []
    for company_dir in sorted(jsons_dir.iterdir()):
        if not company_dir.is_dir():
            continue
        for year_dir in sorted(company_dir.iterdir()):
            if not year_dir.is_dir():
                continue
            json_path = year_dir / "splits_semantic.json"
            if json_path.exists():
                result.append((company_dir.name, year_dir.name, json_path))
    return result


def main():
    parser = argparse.ArgumentParser(description="Match passages against SDG keywords → DuckDB")
    parser.add_argument("--force", action="store_true", help="Drop and recreate table")
    args = parser.parse_args()

    # Ensure data dir exists
    Path("data/dbs").mkdir(parents=True, exist_ok=True)

    if args.force and OUT_DB.exists():
        OUT_DB.unlink()
        log.info("Removed existing DB (--force)")

    # Load & compile keywords
    log.info("Loading keywords...")
    kw_en = load_keywords(KW_EN_PATH)
    kw_de = load_keywords(KW_DE_PATH)

    combined_en = compile_keywords_combined(kw_en)
    combined_de = compile_keywords_combined(kw_de)
    per_pattern_en = compile_keywords_per_pattern(kw_en)
    per_pattern_de = compile_keywords_per_pattern(kw_de)

    log.info(f"Loaded {sum(len(v) for v in kw_en.values())} EN patterns, "
             f"{sum(len(v) for v in kw_de.values())} DE patterns across {len(SDG_CATS)} SDG categories")

    # Discover files
    files = discover_jsons(JSONS_DIR)
    log.info(f"Found {len(files)} JSON files to process")
    if not files:
        log.warning("No JSON files found! Run split_passages.py first.")
        return

    # Create DB
    con = create_db(OUT_DB)

    total_rows = 0
    errors = []

    for company, year, json_path in tqdm(files, desc="SDG matching", unit="file"):
        try:
            rows = process_one_file(
                company, year, json_path,
                combined_en, combined_de, per_pattern_en, per_pattern_de,
            )
            if rows:
                ncols = 5 + 17  # base + sdg cols
                placeholders = ", ".join(["?"] * ncols)
                con.executemany(
                    f"INSERT INTO {TABLE_NAME} VALUES ({placeholders})", rows
                )
                total_rows += len(rows)
        except Exception as e:
            errors.append(f"{company}/{year}: {e}")
            log.error(f"FAILED: {company}/{year} — {e}")

    con.close()

    log.info(f"\nDone. Files processed: {len(files)} | Rows inserted: {total_rows}")
    log.info(f"Database: {OUT_DB}")
    if errors:
        log.warning(f"\n{len(errors)} errors:")
        for e in errors[:20]:
            log.warning(f"  {e}")


if __name__ == "__main__":
    main()
