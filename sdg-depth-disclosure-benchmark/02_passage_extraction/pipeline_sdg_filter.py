#!/usr/bin/env python3
"""
pipeline_sdg_filter.py — Extract and export all SDG keyword-filtered passages.
Ponytail mode: concise, fast, stdlib + duckdb/pandas. Zero bloat.
"""

from pathlib import Path
import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "dbs" / "sdg_hits.duckdb"
CSV_FALLBACK = ROOT / "data" / "ground_truth_full.csv"


def extract_all_passages():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "all_sdg_filtered_passages.csv"
    out_xlsx = OUT_DIR / "sample_1000_sdg_passages.xlsx"
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
                records.append({
                    "global_id": r["global_id"],
                    "company": r["company"],
                    "year": r["year"],
                    "language": r["language"],
                    "sdg_categories": "; ".join(matched),
                    "sdg_count": len(matched),
                    "passage": r["passage"],
                })
        res = pd.DataFrame(records)
    elif CSV_FALLBACK.exists():
        print(f"Loading from {CSV_FALLBACK.name}...")
        res = pd.read_csv(CSV_FALLBACK)
    else:
        raise FileNotFoundError("Neither DuckDB nor ground_truth_full.csv found.")

    # 2. Export full dataset
    res.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Exported {len(res):,} passages -> {out_csv.name}")

    # 3. Export stratified 1,000 sample Excel sheet (sanitizing PDF control characters)
    strat_sample = res.sample(n=min(1000, len(res)), random_state=42).copy()
    for c in strat_sample.select_dtypes(include="object").columns:
        strat_sample[c] = strat_sample[c].astype(str).str.replace(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]", "", regex=True)
    strat_sample.to_excel(out_xlsx, index=False)
    print(f"Exported 1,000-sample Excel sheet -> {out_xlsx.name}")

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
