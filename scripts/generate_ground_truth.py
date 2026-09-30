#!/usr/bin/env python3
"""
generate_ground_truth.py — Generate a ground truth Excel file from SDG DuckDB hits.

Produces two files:
  1. data/ground_truth.xlsx       — Stratified sample for human labeling (blank ground_truth column)
  2. data/ground_truth_full.xlsx  — ALL matched passages (for reference)

Usage:
    python scripts/generate_ground_truth.py [--sample N] [--seed SEED]

Reads:  data/dbs/sdg_hits.duckdb
"""

import argparse
import json
import logging
import random
import re
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

# ── Config ─────────────────────────────────────────────────────────────────────
DB_PATH    = Path("data/dbs/sdg_hits.duckdb")
OUT_SAMPLE = Path("data/ground_truth.xlsx")
OUT_FULL   = Path("data/ground_truth_full.xlsx")
TABLE_NAME = "sdg_hits"

SDG_MAP = {f"hits_sdg{i}": f"SDG {i}" for i in range(1, 18)}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def clean_for_excel(text: Any) -> Any:
    """Remove illegal control characters that crash openpyxl."""
    if not isinstance(text, str):
        return text
    # Remove XML-incompatible control chars except \t, \n, \r
    return re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]", "", text)


def extract_passage_keyword_pairs(db_path: Path) -> pd.DataFrame:
    """
    Read all rows from sdg_hits, explode into (passage, keyword) pairs.
    Each row in the output = one (passage, keyword_regex) pair.
    """
    con = duckdb.connect(str(db_path), read_only=True)

    df = con.execute(f"SELECT * FROM {TABLE_NAME}").fetchdf()
    con.close()

    log.info(f"Loaded {len(df)} passages from DuckDB")

    records = []
    for _, row in df.iterrows():
        clean_passage = clean_for_excel(row["passage"])
        for col, sdg_label in SDG_MAP.items():
            val = row.get(col, "[]")
            try:
                patterns = json.loads(val) if isinstance(val, str) else val
            except (json.JSONDecodeError, TypeError):
                continue

            if isinstance(patterns, list):
                for pat in patterns:
                    if isinstance(pat, str) and pat.strip():
                        records.append({
                            "global_id": row["global_id"],
                            "passage": clean_passage,
                            "keyword": clean_for_excel(pat),
                            "sdg_category": sdg_label,
                            "company": row["company"],
                            "year": row["year"],
                            "language": row["language"],
                            "ground_truth": "",  # blank for human labeling
                        })
            elif isinstance(patterns, dict):
                for pat in patterns.keys():
                    if isinstance(pat, str) and pat.strip():
                        records.append({
                            "global_id": row["global_id"],
                            "passage": clean_passage,
                            "keyword": clean_for_excel(pat),
                            "sdg_category": sdg_label,
                            "company": row["company"],
                            "year": row["year"],
                            "language": row["language"],
                            "ground_truth": "",
                        })

    result = pd.DataFrame(records)
    log.info(f"Exploded into {len(result)} (passage, keyword) pairs")
    return result


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """
    Stratified sample across SDG categories, companies, and years.
    Ensures each SDG category is represented proportionally.
    """
    rng = random.Random(seed)

    if len(df) <= n:
        log.info(f"Dataset ({len(df)}) is smaller than requested sample ({n}), returning all")
        return df.copy()

    # Group by SDG category for proportional sampling
    groups = df.groupby("sdg_category")
    per_group = max(1, n // len(groups))
    remainder = n - per_group * len(groups)

    sampled_indices = set()
    group_sizes = []

    for cat, group_df in groups:
        available = group_df.index.tolist()
        take = min(len(available), per_group)
        chosen = rng.sample(available, take)
        sampled_indices.update(chosen)
        group_sizes.append((cat, len(available), take))

    # Fill remainder from largest undersampled groups
    remaining_pool = [i for i in df.index if i not in sampled_indices]
    if remainder > 0 and remaining_pool:
        extra = rng.sample(remaining_pool, min(remainder, len(remaining_pool)))
        sampled_indices.update(extra)

    result = df.loc[sorted(sampled_indices)].copy()

    log.info(f"Stratified sample: {len(result)} rows from {len(groups)} SDG categories")
    for cat, total, taken in sorted(group_sizes):
        log.info(f"  {cat}: {taken}/{total}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Generate ground truth Excel from SDG DuckDB")
    parser.add_argument("--sample", type=int, default=1000,
                        help="Number of rows for the stratified sample (default: 1000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    if not DB_PATH.exists():
        log.error(f"Database not found: {DB_PATH}. Run sdg_keyword_match.py first.")
        return

    # Extract all (passage, keyword) pairs
    full_df = extract_passage_keyword_pairs(DB_PATH)

    if full_df.empty:
        log.warning("No passages found in database!")
        return

    # Column order matching the example file
    cols_sample = ["passage", "keyword", "ground_truth"]
    cols_full   = ["global_id", "passage", "keyword", "sdg_category",
                   "company", "year", "language", "ground_truth"]

    # Write full export
    full_df[cols_full].to_excel(str(OUT_FULL), index=False, engine="openpyxl")
    full_csv = OUT_FULL.with_suffix(".csv")
    full_df[cols_full].to_csv(str(full_csv), index=False, encoding="utf-8-sig")
    log.info(f"Full export: {OUT_FULL} & {full_csv} ({len(full_df)} rows)")

    # Write stratified sample
    sample_df = stratified_sample(full_df, args.sample, args.seed)
    sample_df[cols_sample].to_excel(str(OUT_SAMPLE), index=False, engine="openpyxl")
    sample_csv = OUT_SAMPLE.with_suffix(".csv")
    sample_df[cols_sample].to_csv(str(sample_csv), index=False, encoding="utf-8-sig")
    log.info(f"Stratified sample: {OUT_SAMPLE} & {sample_csv} ({len(sample_df)} rows)")

    # Print summary stats
    log.info("\n─── Summary ───")
    log.info(f"Total passages in DB: {full_df['global_id'].nunique()}")
    log.info(f"Total (passage,keyword) pairs: {len(full_df)}")
    log.info(f"Companies: {full_df['company'].nunique()}")
    log.info(f"Years: {sorted(full_df['year'].unique())}")
    log.info(f"SDG categories with hits: {sorted(full_df['sdg_category'].unique())}")
    log.info(f"\nPer-SDG hit counts:")
    for cat in sorted(full_df["sdg_category"].unique()):
        count = len(full_df[full_df["sdg_category"] == cat])
        log.info(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
