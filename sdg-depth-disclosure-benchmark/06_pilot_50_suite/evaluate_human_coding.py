#!/usr/bin/env python3
"""
evaluate_human_coding.py — Benchmarks YOUR Manual Human Labels against Model Predictions.

Usage:
  python evaluate_human_coding.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, cohen_kappa_score, confusion_matrix
from scipy.stats import pearsonr

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

SINGLE_HUMAN_PATH = DATA_DIR / "coding_sheet_50_samples.xlsx"
CODER_A_PATH = DATA_DIR / "coding_sheet_person_A_50.xlsx"
CODER_B_PATH = DATA_DIR / "coding_sheet_person_B_50.xlsx"
MODEL_PRED_PATH = BASE_DIR / "output_predictions_50.xlsx"


def load_human_labels():
    """Finds and loads whichever human coding sheet has filled labels."""
    # Check single human sheet
    if SINGLE_HUMAN_PATH.exists():
        df = pd.read_excel(SINGLE_HUMAN_PATH)
        for col in ["human_score_0_to_5", "human_score", "score", "score_0_to_5"]:
            if col in df.columns:
                valid = df[df[col].notna() & (df[col] != "")]
                if len(valid) > 0:
                    print(f"Found {len(valid)} labeled rows in: {SINGLE_HUMAN_PATH.name}")
                    return df, col, "Your Human Labels"

    # Check Coder A
    if CODER_A_PATH.exists():
        df = pd.read_excel(CODER_A_PATH)
        for col in ["person_A_score_0_to_5", "person_a_score", "score"]:
            if col in df.columns:
                valid = df[df[col].notna() & (df[col] != "")]
                if len(valid) > 0:
                    print(f"Found {len(valid)} labeled rows in: {CODER_A_PATH.name}")
                    return df, col, "Person A Labels"

    return None, None, None


def main():
    print("=" * 80)
    print("  HUMAN VS. MODEL EVALUATION BENCHMARK (0 to 5 Depth Scale)")
    print("=" * 80)

    if not MODEL_PRED_PATH.exists():
        print(f"Error: Model predictions not found at {MODEL_PRED_PATH.name}. Run run_pipeline_50.py first.")
        sys.exit(1)

    df_human, score_col, coder_name = load_human_labels()

    if df_human is None:
        print("\n[!] No human labels detected yet!")
        print("  Please open one of the coding sheets in 'Pipeline Test/data/':")
        print(f"    * {SINGLE_HUMAN_PATH.name} (Recommended)")
        print(f"    * {CODER_A_PATH.name}")
        print("\n  Enter your scores (0, 1, 2, 3, 4, or 5) into the highlighted yellow column,")
        print("  save the Excel file, and run this script again to see your benchmark scores!")
        print("=" * 80 + "\n")
        return

    df_model = pd.read_excel(MODEL_PRED_PATH)

    # Filter to only rows the human has labeled so far
    mask = df_human[score_col].notna() & (df_human[score_col].astype(str).str.strip() != "")
    labeled_count = mask.sum()

    if labeled_count == 0:
        print("\n[!] You have not entered any numeric scores yet. Please label some rows and re-run.")
        return

    sub_human = df_human[mask].copy()
    sub_model = df_model.loc[sub_human.index].copy()

    try:
        y_human = sub_human[score_col].astype(float).values
    except ValueError:
        print(f"Error: Some values in '{score_col}' are not numbers. Please only enter numbers from 0 to 5.")
        sys.exit(1)

    y_pred_cont = sub_model["continuous_score (0.00-5.00)"].values
    y_pred_int = sub_model["izhar_level (0-5)"].values

    # Compute Metrics
    rmse = np.sqrt(mean_squared_error(y_human, y_pred_cont))
    mae = mean_absolute_error(y_human, y_pred_cont)
    
    exact_match = (y_human.astype(int) == y_pred_int).mean() * 100
    within_one = (np.abs(y_human.astype(int) - y_pred_int) <= 1).mean() * 100

    kappa_w = cohen_kappa_score(y_human.astype(int), y_pred_int, weights="quadratic")

    # Binary comparison
    bin_human = np.array(["sub" if s >= 2 else "sym" for s in y_human])
    bin_model = sub_model["binary_label"].values
    bin_acc = (bin_human == bin_model).mean() * 100
    bin_kappa = cohen_kappa_score(bin_human, bin_model)

    print(f"\nEvaluated on {labeled_count} human-labeled passages ({coder_name}):")
    print("-" * 80)
    print(f"1. Continuous Depth Metrics (0.00 to 5.00 Scale):")
    if labeled_count >= 3:
        corr, p_val = pearsonr(y_human, y_pred_cont)
        print(f"   • Pearson Correlation (r):       {corr:.4f} (p = {p_val:.2e})")
    print(f"   • Root Mean Squared Error (RMSE): ±{rmse:.3f} points")
    print(f"   • Mean Absolute Error (MAE):      ±{mae:.3f} points")

    print(f"\n2. Ordinal Level Agreement (Scores 0, 1, 2, 3, 4, 5):")
    print(f"   • Exact Integer Match:            {exact_match:.1f}% ({int(exact_match*labeled_count/100)} / {labeled_count})")
    print(f"   • Within +/- 1 Level Agreement:   {within_one:.1f}% ({int(within_one*labeled_count/100)} / {labeled_count})")
    print(f"   • Quadratic Weighted Kappa (κ_w): {kappa_w:.4f}")

    print(f"\n3. High-Level Binary Agreement (Symbolic vs. Substantive):")
    print(f"   • Binary Accuracy:                {bin_acc:.1f}%")
    print(f"   • Binary Cohen's κ:               {bin_kappa:.4f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
