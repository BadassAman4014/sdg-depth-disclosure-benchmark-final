#!/usr/bin/env python3
"""
merge_and_evaluate_coders.py — Merge Person A and Person B's sheets and evaluate reliability.

Reads:
  - data/coding_sheet_person_A.xlsx
  - data/coding_sheet_person_B.xlsx
  - data/golden_dataset_500.xlsx

Outputs:
  - Inter-Coder Reliability (Cohen's Kappa κ, % agreement)
  - Identifies disagreements for consensus resolution
  - Runs 5-Fold Cross-Validated Logistic Regression, KNN, Random Forest, and AI reliability benchmark
"""

import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import cohen_kappa_score

from run_logistic_regression_pipeline import train_and_evaluate_models, print_top_features, predict_full_corpus

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SHEET_A = Path("data/coding_sheet_person_A.xlsx")
SHEET_B = Path("data/coding_sheet_person_B.xlsx")
GOLDEN_500 = Path("data/golden_dataset_500.xlsx")


def main():
    parser = argparse.ArgumentParser(description="Merge Person A & B sheets and run reliability evaluation")
    parser.add_argument("--sheet-a", default="data/coding_sheet_person_A.xlsx", help="Person A's sheet")
    parser.add_argument("--sheet-b", default="data/coding_sheet_person_B.xlsx", help="Person B's sheet")
    parser.add_argument("--predict-full", action="store_true", help="Scale Logistic Regression to full 63k corpus")
    args = parser.parse_args()

    df_a = pd.read_excel(args.sheet_a)
    df_b = pd.read_excel(args.sheet_b)
    df_main = pd.read_excel(GOLDEN_500)

    # Extract label columns
    col_a = [c for c in df_a.columns if "label" in c.lower() or "coder" in c.lower()][0]
    col_b = [c for c in df_b.columns if "label" in c.lower() or "coder" in c.lower()][0]

    labels_a = df_a[col_a].astype(str).str.strip().str.lower()
    labels_b = df_b[col_b].astype(str).str.strip().str.lower()

    # Populate main dataset
    df_main["coder_1"] = labels_a.replace({"nan": "", "none": ""})
    df_main["coder_2"] = labels_b.replace({"nan": "", "none": ""})

    # Find overlapping coded items
    mask_both = df_main["coder_1"].isin(["sym", "sub"]) & df_main["coder_2"].isin(["sym", "sub"])
    overlap_count = mask_both.sum()

    print("\n" + "=" * 78)
    print("  1. DUAL-CODING & INTER-CODER RELIABILITY REPORT")
    print("=" * 78)

    if overlap_count >= 5:
        c1 = df_main.loc[mask_both, "coder_1"]
        c2 = df_main.loc[mask_both, "coder_2"]
        kappa = cohen_kappa_score(c1, c2)
        pct = (c1 == c2).mean() * 100
        disagreements = (c1 != c2).sum()

        print(f"  Overlapping Items Coded:  {overlap_count} samples")
        print(f"  Cohen's Kappa (κ):        {kappa:.4f}  (>=0.80 = Strong Academic Agreement)")
        print(f"  Raw Agreement:            {pct:.2f}%")
        print(f"  Disagreements to Resolve: {disagreements} items")

        # Auto-fill consensus where both agree
        consensus = []
        for _, row in df_main.iterrows():
            a = row["coder_1"]
            b = row["coder_2"]
            if a in ["sym", "sub"] and b in ["sym", "sub"]:
                consensus.append(a if a == b else "")
            elif a in ["sym", "sub"]:
                consensus.append(a)
            elif b in ["sym", "sub"]:
                consensus.append(b)
            else:
                consensus.append("")
        df_main["human_consensus"] = consensus

    else:
        print(f"  Overlapping Items Coded:  {overlap_count} (Need at least 5 for Cohen's Kappa)")

    # Save merged dataset
    df_main.to_excel(GOLDEN_500, index=False, engine="openpyxl")
    df_main.to_csv(GOLDEN_500.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    print(f"\n  Updated merged dataset: {GOLDEN_500}")

    # Run ML and AI evaluation
    valid_mask = df_main["human_consensus"].isin(["sym", "sub"])
    valid_df = df_main[valid_mask].copy()
    valid_df["target_label"] = valid_df["human_consensus"]

    if len(valid_df) >= 10:
        print("\n" + "=" * 78)
        print("  2. LOGISTIC REGRESSION & AI RELIABILITY BENCHMARK")
        print("=" * 78)
        results, vectorizer, log_reg = train_and_evaluate_models(valid_df)

        table_rows = []
        for model_name, m in results.items():
            table_rows.append({
                "Model": model_name,
                "Accuracy": f"{m['Accuracy (%)']}%",
                "Precision": f"{m['Precision (sub) (%)']}%",
                "Recall": f"{m['Recall (sub) (%)']}%",
                "F1-Score": f"{m['F1 (sub) (%)']}%",
                "Macro F1": f"{m['Macro F1 (%)']}%",
                "Type I Err": f"{m['Type I Error (%)']}%",
                "Type II Err": f"{m['Type II Error (%)']}%",
            })
        print(pd.DataFrame(table_rows).to_string(index=False))
        print("=" * 78)

        print_top_features(vectorizer, log_reg)

        if args.predict_full:
            predict_full_corpus(vectorizer, log_reg)
    else:
        print(f"\n  [NOTE] {len(valid_df)} agreed consensus labels found. (Need >=10 for ML cross-validation).")


if __name__ == "__main__":
    main()
