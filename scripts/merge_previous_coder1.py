#!/usr/bin/env python3
"""
merge_previous_coder1.py — Merge previous 100 hand-coded samples as Coder 1.

Places the 100 validated ground truth samples from
'Updated Ground Truths - 100 samples for evaluating models.xlsx'
into the golden dataset as 'coder_1'.

Leaves 'coder_2' blank for you to annotate.
"""

import logging
import re
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PREV_100_FILE = Path("Updated Ground Truths - 100 samples for evaluating models.xlsx")
DATASET_500_FILE = Path("data/golden_dataset_500.xlsx")
HUMAN_SHEET_FILE = Path("data/human_coding_sheet_500.xlsx")


def clean_str(s):
    if not isinstance(s, str):
        return ""
    # Normalize whitespace & remove XML control chars
    s = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]", "", s)
    return " ".join(s.strip().split())


def main():
    log.info(f"Loading previous 100 samples from: {PREV_100_FILE}...")
    df_100 = pd.read_excel(PREV_100_FILE)
    df_100["clean_p"] = df_100["passage"].apply(clean_str)
    df_100["clean_kw"] = df_100["keyword"].apply(clean_str)
    
    # Build lookup dict: (clean_passage_prefix_100_chars, clean_keyword) -> ground_truth
    lookup = {}
    for _, row in df_100.iterrows():
        key = (row["clean_p"][:120], row["clean_kw"])
        lookup[key] = str(row["ground_truth"]).strip().lower()

    log.info(f"Loaded {len(lookup)} unique lookup keys from previous 100 samples")

    log.info(f"Loading 500-sample dataset: {DATASET_500_FILE}...")
    df_500 = pd.read_excel(DATASET_500_FILE)
    df_500["clean_p"] = df_500["passage"].apply(clean_str)
    df_500["clean_kw"] = df_500["keyword"].apply(clean_str)

    matched_count = 0
    # First attempt: match by exact prefix
    for idx, row in df_500.iterrows():
        key = (row["clean_p"][:120], row["clean_kw"])
        if key in lookup:
            df_500.at[idx, "coder_1"] = lookup[key]
            matched_count += 1

    log.info(f"Directly matched {matched_count} samples in existing 500 dataset")

    # If fewer than 100 matched due to stratified re-sampling, let's embed the 100 samples directly at the top (sample_id 1 to 100)!
    if matched_count < len(df_100):
        log.info(f"Embedding all {len(df_100)} previous samples as the first 100 items of the golden dataset...")
        
        # Prepare 100 samples dataframe with full schema
        df_100_full = df_100.copy()
        df_100_full["coder_1"] = df_100_full["ground_truth"].astype(str).str.strip().str.lower()
        df_100_full["coder_2"] = ""
        df_100_full["human_consensus"] = ""
        df_100_full["notes"] = "From previous 100 ground truth"
        
        # Merge remaining 400 distinct samples from df_500
        existing_prefixes = set(df_100["clean_p"].str[:120])
        remaining_400 = df_500[~df_500["clean_p"].str[:120].isin(existing_prefixes)].head(400).copy()
        
        # Combine 100 + 400 = 500
        combined_df = pd.concat([df_100_full, remaining_400], ignore_index=True)
        combined_df = combined_df.head(500)
        combined_df["sample_id"] = range(1, len(combined_df) + 1)
        
        # Drop temporary cols
        drop_cols = [c for c in ["clean_p", "clean_kw", "ground_truth"] if c in combined_df.columns]
        combined_df = combined_df.drop(columns=drop_cols)
        
        df_500 = combined_df

    # Organize column order
    priority = [
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
        "AI Labelling",
        "ai_confidence",
        "ai_evidence_type",
        "notes",
        "global_id",
    ]
    cols = df_500.columns.tolist()
    ordered_cols = [c for c in priority if c in cols] + [c for c in cols if c not in priority]
    df_500 = df_500[ordered_cols]

    # Save to golden_dataset_500.xlsx & .csv
    df_500.to_excel(DATASET_500_FILE, index=False, engine="openpyxl")
    df_500.to_csv(DATASET_500_FILE.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    log.info(f"Updated {DATASET_500_FILE} ({len(df_500)} rows)")

    # Save human coding sheet
    human_cols = ["sample_id", "passage", "keyword", "coder_1", "coder_2", "human_consensus", "notes"]
    human_view = df_500[[c for c in human_cols if c in df_500.columns]]
    human_view.to_excel(HUMAN_SHEET_FILE, index=False, engine="openpyxl")
    log.info(f"Updated {HUMAN_SHEET_FILE} (Ready for you as Coder 2!)")

    coder1_count = (df_500["coder_1"].notna() & (df_500["coder_1"] != "")).sum()
    log.info(f"\nSummary: {coder1_count} rows have 'coder_1' populated.")
    log.info("You (Coder 2) only need to fill the 'coder_2' column for the first 100 rows!")


if __name__ == "__main__":
    main()
