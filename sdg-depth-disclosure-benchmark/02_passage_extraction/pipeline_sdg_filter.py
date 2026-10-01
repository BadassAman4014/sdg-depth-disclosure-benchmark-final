#!/usr/bin/env python3
"""
pipeline_sdg_filter.py — Extract, sanitize, and export all SDG keyword-filtered passages.
Ponytail mode: concise, fast, stdlib + duckdb/pandas. Zero bloat.
Eliminates Caesar font shift, ligatures, Mojibake, mid-word spaces, and table debris AT SOURCE.
"""

from pathlib import Path
import re
import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "dbs" / "sdg_hits.duckdb"
if not DB_PATH.exists() and (ROOT.parent / "data" / "dbs" / "sdg_hits.duckdb").exists():
    DB_PATH = ROOT.parent / "data" / "dbs" / "sdg_hits.duckdb"

CSV_FALLBACK = ROOT / "data" / "ground_truth_full.csv"
if not CSV_FALLBACK.exists() and (ROOT.parent / "data" / "ground_truth_full.csv").exists():
    CSV_FALLBACK = ROOT.parent / "data" / "ground_truth_full.csv"

MOJIBAKE_MAP = {
    "├ñ": "ä", "├╢": "ö", "├╝": "ü", "├ƒ": "ß",
    "├ä": "Ä", "├Ц": "Ö", "├Ь": "Ü",
    "Ã¤": "ä", "Ã¶": "ö", "Ã¼": "ü", "ÃŸ": "ß",
    "Ã„": "Ä", "Ã–": "Ö", "Ãœ": "Ü"
}

CAESAR_WORDS_3 = {
    "WKH": "THE", "DQG": "AND", "IRU": "FOR", "DOO": "ALL", "WR": "TO",
    "RI": "OF", "LQ": "IN", "LV": "IS", "WKDW": "THAT", "WKLV": "THIS",
    "ZLWK": "WITH", "DV": "AS", "RQ": "ON", "DW": "AT", "IURP": "FROM",
    "KDYH": "HAVE", "EHHQ": "BEEN", "FRPSDQ\\": "COMPANY", "GLUHFWRUV": "DIRECTORS",
    "HPSOR\\HHV": "EMPLOYEES", "FRPSOLDQFH": "COMPLIANCE", "GLUHFWLYH": "DIRECTIVE"
}

def _decode_caesar_char_3(c: str) -> str:
    o = ord(c)
    if ord('A') <= o <= ord('Z'):
        return chr((o - ord('A') - 3) % 26 + ord('A'))
    elif ord('a') <= o <= ord('z'):
        return chr((o - ord('a') - 3) % 26 + ord('a'))
    elif c == '\\':
        return 'Y'
    elif c == '[':
        return 'X'
    elif c == ']':
        return 'Z'
    return c

def _decode_caesar_word_3(word: str) -> str:
    return "".join(_decode_caesar_char_3(c) for c in word.replace("ɝ", "ffi").replace("ȴ", "fi"))

def is_table_debris(text: str) -> bool:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return True
    unit_lines = sum(1 for l in lines if re.match(r'^(€k|€m|kEUR|TEUR|\$m|€|\$|%|in €k|in %)$', l, re.I))
    if unit_lines >= 3:
        return True
    words = re.findall(r'[A-Za-zÄÖÜäöüß]{2,}', text)
    numbers = re.findall(r'\b\d+([.,]\d+)?\b', text)
    if len(words) < 15 and len(numbers) > len(words):
        return True
    accounting_noise = ["Book value", "Cost of acquisition", "Measurement according to IAS", "Trade accounts receivable"]
    if sum(1 for term in accounting_noise if term.lower() in text.lower()) >= 2 and len(words) < 40:
        return True
    return False

def sanitize_passage(text: str) -> str:
    if not isinstance(text, str):
        return ""
    for bad, good in MOJIBAKE_MAP.items():
        if bad in text:
            text = text.replace(bad, good)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', ' ', text)
    text = text.replace("ɝ", "ffi").replace("ȴ", "fi").replace("\u00ad", "").replace("\u2009", " ")
    
    tokens = re.findall(r'[A-Za-z\\\[\]]+', text)
    caesar_hits = sum(1 for tok in tokens if tok.upper() in CAESAR_WORDS_3 or tok in CAESAR_WORDS_3)
    if caesar_hits >= 2 or (len(tokens) > 5 and caesar_hits / len(tokens) > 0.15):
        text = re.sub(r'[A-Za-z\\\[\]]+', lambda m: _decode_caesar_word_3(m.group(0)), text)
    else:
        for bad_w, good_w in CAESAR_WORDS_3.items():
            text = re.sub(r'(?<![A-Za-z])' + re.escape(bad_w) + r'(?![A-Za-z])', good_w, text, flags=re.IGNORECASE)

    text = re.sub(r'\b(sys)\s{2,}(tem)\b', r'\1\2', text, flags=re.I)
    text = re.sub(r'\b(ap)\s{2,}(proach)\b', r'\1\2', text, flags=re.I)
    text = re.sub(r'\b(compli)\s{2,}(ance)\b', r'\1\2', text, flags=re.I)
    text = re.sub(r'\b(busi)\s{2,}(ness)\b', r'\1\2', text, flags=re.I)
    text = re.sub(r'\b(require)\s{2,}(ments)\b', r'\1\2', text, flags=re.I)
    text = re.sub(r'\b(employ)\s{2,}(ees)\b', r'\1\2', text, flags=re.I)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def extract_all_passages():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "all_sdg_filtered_passages.csv"
    out_xlsx = OUT_DIR / "golden_dataset_1000.xlsx"
    out_summary_cat = OUT_DIR / "summary_by_sdg_category.csv"
    out_summary_comp = OUT_DIR / "summary_by_company.csv"

    # 1. Load from DuckDB if available, else CSV
    if DB_PATH.exists():
        print(f"Loading from {DB_PATH.name}...")
        con = duckdb.connect(str(DB_PATH), read_only=True)
        cols = ["global_id", "company", "year", "language", "passage"] + [f"hits_sdg{i}" for i in range(1, 18)]
        df = con.execute(f"SELECT {', '.join(cols)} FROM sdg_hits").fetchdf()
        con.close()

        records = []
        for _, r in df.iterrows():
            matched = [f"SDG {i}" for i in range(1, 18) if str(r[f"hits_sdg{i}"]) not in ("[]", "", "None", "{}")]
            if matched:
                clean_p = sanitize_passage(str(r["passage"]))
                if not is_table_debris(clean_p):
                    records.append({
                        "global_id": r["global_id"],
                        "company": r["company"],
                        "year": r["year"],
                        "language": r["language"],
                        "sdg_categories": "; ".join(matched),
                        "sdg_count": len(matched),
                        "passage": clean_p,
                    })
        res = pd.DataFrame(records)
    elif CSV_FALLBACK.exists():
        print(f"Loading from {CSV_FALLBACK.name}...")
        res = pd.read_csv(CSV_FALLBACK)
        res["passage"] = res["passage"].astype(str).apply(sanitize_passage)
        res = res[~res["passage"].apply(is_table_debris)].copy()
    else:
        raise FileNotFoundError("Neither DuckDB nor ground_truth_full.csv found.")

    # 2. Export full sanitized dataset
    res.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Exported {len(res):,} clean passages -> {out_csv.name}")

    # 3. Export stratified 1,000 sample Excel sheet (sanitizing PDF control characters)
    strat_sample = res.sample(n=min(1000, len(res)), random_state=42).copy()
    for c in strat_sample.select_dtypes(include="object").columns:
        strat_sample[c] = strat_sample[c].astype(str).str.replace(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]", "", regex=True)
    strat_sample.to_excel(out_xlsx, index=False, engine="openpyxl")
    print(f"Exported clean 1,000-sample Excel sheet -> {out_xlsx.name}")

    # 4. Export concise summary stats
    if "sdg_categories" in res.columns:
        cats = res["sdg_categories"].str.split("; ").explode().value_counts().reset_index()
        cats.columns = ["sdg_category", "passage_count"]
        cats.to_csv(out_summary_cat, index=False)
        print(f"Exported SDG summary -> {out_summary_cat.name}")

    comps = res["company"].value_counts().reset_index()
    comps.columns = ["company", "passage_count"]
    comps.to_csv(out_summary_comp, index=False)
    print(f"Exported Company summary -> {out_summary_comp.name}")


if __name__ == "__main__":
    extract_all_passages()
