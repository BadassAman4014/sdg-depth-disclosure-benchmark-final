#!/usr/bin/env python3
"""
verify_all.py — Comprehensive verification of all generated files across the pipeline.
"""

import json
import duckdb
import pandas as pd
from pathlib import Path

def main():
    print("=" * 70)
    print("  COMPREHENSIVE PIPELINE ARTIFACTS VERIFICATION")
    print("=" * 70)

    # 1. Text files
    print("\n[1] TEXT FILES (data/texts/)")
    text_files = list(Path("data/texts").glob("*/*/results.txt"))
    print(f"  - Total results.txt files: {len(text_files)}")
    total_text_bytes = sum(f.stat().st_size for f in text_files)
    print(f"  - Total extracted text size: {total_text_bytes / (1024*1024):.2f} MB")
    assert len(text_files) >= 1400, "Expected at least 1,400 text files"

    # 2. JSON files
    print("\n[2] JSON PASSAGE FILES (data/jsons/)")
    json_files = list(Path("data/jsons").glob("*/*/splits_semantic.json"))
    print(f"  - Total splits_semantic.json files: {len(json_files)}")
    sample_passages = 0
    for jf in json_files[:50]:
        with open(jf, encoding="utf-8") as f:
            sample_passages += len(json.load(f))
    print(f"  - Passages in first 50 sample files: {sample_passages:,} (~{sample_passages/50:.0f} per report)")
    assert len(json_files) >= 1400, "Expected at least 1,400 JSON files"

    # 3. DuckDB
    print("\n[3] DUCKDB DATABASE (data/dbs/sdg_hits.duckdb)")
    con = duckdb.connect("data/dbs/sdg_hits.duckdb", read_only=True)
    count = con.execute("SELECT count(*) FROM sdg_hits").fetchone()[0]
    cols = [col[0] for col in con.execute("DESCRIBE sdg_hits").fetchall()]
    print(f"  - Total rows in 'sdg_hits': {count:,}")
    print(f"  - Total columns: {len(cols)}")
    print(f"  - Column names: {', '.join(cols)}")
    
    # Check hit counts per SDG column
    print("  - Hit counts per SDG category in DuckDB:")
    for i in range(1, 18):
        cat_count = con.execute(f"SELECT count(*) FROM sdg_hits WHERE hits_sdg{i} != '[]'").fetchone()[0]
        print(f"    * hits_sdg{i:02d}: {cat_count:,} passages")
    con.close()
    assert count > 0, "Expected non-zero rows in DuckDB"

    # 4. Ground Truth Sample (1,000 rows)
    print("\n[4] GROUND TRUTH SAMPLE (data/ground_truth.xlsx & .csv)")
    df_sample = pd.read_excel("data/ground_truth.xlsx")
    df_sample_csv = pd.read_csv("data/ground_truth.csv")
    print(f"  - Excel dimensions: {df_sample.shape[0]:,} rows x {df_sample.shape[1]} columns")
    print(f"  - CSV dimensions: {df_sample_csv.shape[0]:,} rows x {df_sample_csv.shape[1]} columns")
    print(f"  - Column headers: {df_sample.columns.tolist()}")
    print(f"  - Null passages: {df_sample['passage'].isna().sum()} (must be 0)")
    print(f"  - Null keywords: {df_sample['keyword'].isna().sum()} (must be 0)")
    print(f"  - Blank ground_truth count: {df_sample['ground_truth'].isna().sum()} (must be {len(df_sample)})")
    assert df_sample.shape == (1000, 3), "Expected (1000, 3) for sample dataset"
    assert df_sample["passage"].isna().sum() == 0, "No passages should be null"
    assert df_sample["keyword"].isna().sum() == 0, "No keywords should be null"
    assert df_sample["ground_truth"].isna().sum() == 1000, "All ground_truth cells must be blank"

    # 5. Full Ground Truth (63,185 rows)
    print("\n[5] FULL DATASET (data/ground_truth_full.xlsx & .csv)")
    df_full = pd.read_excel("data/ground_truth_full.xlsx")
    df_full_csv = pd.read_csv("data/ground_truth_full.csv")
    print(f"  - Excel dimensions: {df_full.shape[0]:,} rows x {df_full.shape[1]} columns")
    print(f"  - CSV dimensions: {df_full_csv.shape[0]:,} rows x {df_full_csv.shape[1]} columns")
    print(f"  - Column headers: {df_full.columns.tolist()}")
    print(f"  - Unique companies: {df_full['company'].nunique()}")
    print(f"  - Unique years: {sorted(df_full['year'].unique().tolist())}")
    print(f"  - Unique SDG categories: {df_full['sdg_category'].nunique()}")
    assert len(df_full) == 63185, f"Expected 63,185 rows, got {len(df_full)}"
    assert df_full["sdg_category"].nunique() == 17, "Expected all 17 SDGs to be represented"

    # 6. Template Compatibility Check
    print("\n[6] TEMPLATE COMPATIBILITY CHECK")
    df_orig = pd.read_excel("Updated Ground Truths - 100 samples for evaluating models.xlsx")
    print(f"  - Original template columns: {df_orig.columns.tolist()}")
    print(f"  - Generated sample columns:  {df_sample.columns.tolist()}")
    match = df_orig.columns.tolist() == df_sample.columns.tolist()
    print(f"  - Exact schema match: {'PASSED (100% Identical)' if match else 'FAILED'}")
    assert match, "Schema must match original template"

    # Sample rows display
    print("\n[7] SPOT-CHECK: FIRST 2 SAMPLE ROWS (for Human Evaluation)")
    for i, row in df_sample.head(2).iterrows():
        print(f"\n--- Sample #{i+1} ---")
        print(f"Keyword Matched: {row['keyword']}")
        print(f"Passage Preview: {row['passage'][:250]}...")

    print("\n" + "=" * 70)
    print("  ALL VERIFICATION CHECKS PASSED SUCCESSFULLY (6/6)")
    print("=" * 70)

if __name__ == "__main__":
    main()
