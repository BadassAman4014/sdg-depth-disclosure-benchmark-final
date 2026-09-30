#!/usr/bin/env python3
"""
create_golden_dataset.py — Create the Golden Dataset for Human Evaluation & AI Reliability Testing.

Grounded in:
  1. De Kok (2025) - "ChatGPT for Textual Analysis? How to Use Generative LLMs in Accounting Research" (Management Science)
  2. Shah et al. (2020) - "A Comparative Analysis of Logistic Regression, Random Forest and KNN Models for Text Classification"

Generates:
  - data/golden_dataset_500.xlsx & .csv   (Standard N=500 academic evaluation dataset)
  - data/golden_dataset_1000.xlsx & .csv  (Extended N=1000 dataset for high statistical power)

Schema includes:
  - sample_id: Unique integer identifier (1..N)
  - global_id: Report passage identifier
  - company: Firm name
  - year: Report year (2014..2024)
  - sdg_category: SDG category (SDG 1..17)
  - keyword: Matched keyword regex pattern
  - passage: Text passage to evaluate
  - coder_1: Label from first human rater (sym / sub)
  - coder_2: Label from second human rater (sym / sub)
  - human_consensus: Final agreed human ground truth (sym / sub)
  - chatgpt_prediction: AI model classification (sym / sub)
  - notes: Optional rationale/comments from coders
"""

import json
import logging
import random
import re
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

DB_PATH = Path("data/dbs/sdg_hits.duckdb")
TABLE_NAME = "sdg_hits"
SDG_MAP = {f"hits_sdg{i}": f"SDG {i}" for i in range(1, 18)}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def clean_for_excel(text: Any) -> Any:
    if not isinstance(text, str):
        return text
    return re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]", "", text)


def load_and_explode_pairs(db_path: Path) -> pd.DataFrame:
    con = duckdb.connect(str(db_path), read_only=True)
    df = con.execute(f"SELECT * FROM {TABLE_NAME}").fetchdf()
    con.close()

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
                            "company": row["company"],
                            "year": row["year"],
                            "language": row["language"],
                            "sdg_category": sdg_label,
                            "keyword": clean_for_excel(pat),
                            "passage": clean_passage,
                        })
            elif isinstance(patterns, dict):
                for pat in patterns.keys():
                    if isinstance(pat, str) and pat.strip():
                        records.append({
                            "global_id": row["global_id"],
                            "company": row["company"],
                            "year": row["year"],
                            "language": row["language"],
                            "sdg_category": sdg_label,
                            "keyword": clean_for_excel(pat),
                            "passage": clean_passage,
                        })

    result = pd.DataFrame(records)
    log.info(f"Loaded {len(result):,} total (passage, keyword) pairs")
    return result


def create_stratified_sample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    
    # Stratify by SDG category
    groups = df.groupby("sdg_category")
    per_group = max(1, n // len(groups))
    remainder = n - per_group * len(groups)

    sampled_indices = set()
    for cat, group_df in sorted(groups, key=lambda x: x[0]):
        available = group_df.index.tolist()
        take = min(len(available), per_group)
        sampled_indices.update(rng.sample(available, take))

    # Fill remainder from remaining pool
    remaining = [i for i in df.index if i not in sampled_indices]
    if remainder > 0 and remaining:
        sampled_indices.update(rng.sample(remaining, min(remainder, len(remaining))))

    sample = df.loc[sorted(sampled_indices)].copy()
    
    # Add human coding and model columns
    sample.insert(0, "sample_id", range(1, len(sample) + 1))
    sample["coder_1"] = ""
    sample["coder_2"] = ""
    sample["human_consensus"] = ""
    sample["chatgpt_prediction"] = ""
    sample["notes"] = ""

    ordered_cols = [
        "sample_id",
        "passage",
        "keyword",
        "sdg_category",
        "company",
        "year",
        "language",
        "coder_1",
        "coder_2",
        "human_consensus",
        "chatgpt_prediction",
        "notes",
        "global_id",
    ]
    return sample[ordered_cols]


def main():
    log.info("Extracting data for Golden Dataset creation...")
    full_df = load_and_explode_pairs(DB_PATH)

    # 1. Generate N=500 dataset (De Kok 2025 standard)
    sample_500 = create_stratified_sample(full_df, n=500, seed=42)
    sample_500.to_excel("data/golden_dataset_500.xlsx", index=False, engine="openpyxl")
    sample_500.to_csv("data/golden_dataset_500.csv", index=False, encoding="utf-8-sig")
    log.info(f"Created data/golden_dataset_500.xlsx & .csv ({len(sample_500)} rows)")

    # 2. Generate N=1000 dataset (Extended statistical power)
    sample_1000 = create_stratified_sample(full_df, n=1000, seed=42)
    sample_1000.to_excel("data/golden_dataset_1000.xlsx", index=False, engine="openpyxl")
    sample_1000.to_csv("data/golden_dataset_1000.csv", index=False, encoding="utf-8-sig")
    log.info(f"Created data/golden_dataset_1000.xlsx & .csv ({len(sample_1000)} rows)")

    # Also output a pure human labeling template (simplified 3-column view)
    human_view_500 = sample_500[["sample_id", "passage", "keyword", "coder_1", "coder_2", "human_consensus", "notes"]]
    human_view_500.to_excel("data/human_coding_sheet_500.xlsx", index=False, engine="openpyxl")
    log.info("Created data/human_coding_sheet_500.xlsx (dedicated sheet for human annotators)")


if __name__ == "__main__":
    main()
