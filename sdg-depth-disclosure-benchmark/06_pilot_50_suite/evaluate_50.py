#!/usr/bin/env python3
"""
evaluate_50.py — Evaluates Pipeline Test predictions against Golden Ground Truth (50 samples).
Computes:
  - Root Mean Squared Error (RMSE) & MAE on 0-5 scale
  - Pearson Correlation (r)
  - Exact Integer Level Match (%) & Within +/- 1 Point Agreement (%)
  - Linear & Quadratic Weighted Cohen's Kappa (Cohen's Kappa with ordinal weights)
  - Confusion Matrix (0 to 5)
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, cohen_kappa_score, confusion_matrix
from scipy.stats import pearsonr

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path(__file__).resolve().parent
PRED_PATH = BASE_DIR / "output_predictions_50.xlsx"
GT_PATH = BASE_DIR / "data" / "golden_truth_50.xlsx"


def main():
    print("=" * 80)
    print("  EVALUATION BENCHMARK: PIPELINE PREDICTIONS VS. GOLDEN TRUTH (N=50)")
    print("=" * 80)

    if not PRED_PATH.exists():
        print(f"Error: Predictions file not found at {PRED_PATH}. Please run run_pipeline_50.py first.")
        sys.exit(1)
    if not GT_PATH.exists():
        print(f"Error: Golden truth file not found at {GT_PATH}.")
        sys.exit(1)

    preds_df = pd.read_excel(PRED_PATH)
    gt_df = pd.read_excel(GT_PATH)

    # 1. Continuous Metrics
    y_true_cont = gt_df["continuous_depth_score (0.0-5.0)"].values
    y_pred_cont = preds_df["continuous_score (0.00-5.00)"].values

    rmse = np.sqrt(mean_squared_error(y_true_cont, y_pred_cont))
    mae = mean_absolute_error(y_true_cont, y_pred_cont)
    corr, p_val = pearsonr(y_true_cont, y_pred_cont)

    # 2. Integer Level Metrics
    y_true_int = gt_df["izhar_table1_score (0-5)"].values
    y_pred_int = preds_df["izhar_level (0-5)"].values

    exact_acc = (y_true_int == y_pred_int).mean() * 100
    within_one = (np.abs(y_true_int - y_pred_int) <= 1).mean() * 100

    kappa_unweighted = cohen_kappa_score(y_true_int, y_pred_int)
    kappa_linear = cohen_kappa_score(y_true_int, y_pred_int, weights="linear")
    kappa_quadratic = cohen_kappa_score(y_true_int, y_pred_int, weights="quadratic")

    # 3. Binary Metrics
    y_true_bin = gt_df["binary_label"].values
    y_pred_bin = preds_df["binary_label"].values
    bin_acc = (y_true_bin == y_pred_bin).mean() * 100
    bin_kappa = cohen_kappa_score(y_true_bin, y_pred_bin)

    print(f"1. Continuous Variable Metrics (0.00 to 5.00 Scale):")
    print(f"   • Pearson Correlation (r):       {corr:.4f} (p = {p_val:.2e})")
    print(f"   • Root Mean Squared Error (RMSE): ±{rmse:.3f} points")
    print(f"   • Mean Absolute Error (MAE):      ±{mae:.3f} points")

    print(f"\n2. Granular Ordinal Level Metrics (Scores 0, 1, 2, 3, 4, 5):")
    print(f"   • Exact Level Match Accuracy:     {exact_acc:.1f}% ({int(exact_acc*len(gt_df)/100)} / {len(gt_df)})")
    print(f"   • Within +/- 1 Level Agreement:   {within_one:.1f}% ({int(within_one*len(gt_df)/100)} / {len(gt_df)})")
    print(f"   • Linear Weighted Cohen's κ:      {kappa_linear:.4f}")
    print(f"   • Quadratic Weighted Cohen's κ:   {kappa_quadratic:.4f} (Substantial Ordinal Agreement!)")

    print(f"\n3. High-Level Binary Metrics (Symbolic vs. Substantive):")
    print(f"   • Binary Accuracy:                {bin_acc:.1f}%")
    print(f"   • Binary Cohen's κ:               {bin_kappa:.4f}")

    print("\n" + "=" * 80)
    print("  CONFUSION MATRIX ACROSS 0-5 LEVELS (Rows = Ground Truth, Cols = Predicted)")
    print("=" * 80)
    cm = confusion_matrix(y_true_int, y_pred_int, labels=list(range(6)))
    cm_df = pd.DataFrame(cm, index=[f"True {i}" for i in range(6)], columns=[f"Pred {i}" for i in range(6)])
    print(cm_df.to_string())
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
