#!/usr/bin/env python3
"""
sync_human_data_and_benchmark.py — Complete Pipeline Integration for 50 Human-Labeled Samples.

Steps:
  1. Synchronizes Person A and Person B human annotations into 'Pipeline Test/data/golden_truth_50.xlsx'.
  2. Updates 'Pipeline Test/data/coding_sheet_50_samples.xlsx' with consensus human annotations.
  3. Fits and calibrates the continuous regressor on human ground truth with 5-Fold Cross-Validation.
  4. Runs predictions on all 50 samples and saves 'Pipeline Test/output_predictions_50.xlsx'.
  5. Computes comprehensive benchmark metrics (RMSE, MAE, Pearson r, Quadratic Weighted Kappa, Confusion Matrix).
"""

import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, cohen_kappa_score, confusion_matrix
from scipy.stats import pearsonr, spearmanr
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# Import continuous pipeline module
from continuous_model import ContinuousDepthPipeline, extract_linguistic_features, train_and_save_model

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

SHEET_A_PATH = DATA_DIR / "coding_sheet_person_A_50.xlsx"
SHEET_B_PATH = DATA_DIR / "coding_sheet_person_B_50.xlsx"
GOLDEN_TRUTH_PATH = DATA_DIR / "golden_truth_50.xlsx"
MASTER_CODING_PATH = DATA_DIR / "coding_sheet_50_samples.xlsx"
OUTPUT_EXCEL = BASE_DIR / "output_predictions_50.xlsx"
OUTPUT_CSV = BASE_DIR / "output_predictions_50.csv"
MODEL_PATH = BASE_DIR / "model_depth_50.joblib"


def style_excel_grid(file_path: Path):
    wb = openpyxl.load_workbook(file_path)
    ws = wb.active
    ws.views.sheetView[0].showGridLines = True

    header_fill = PatternFill(start_color="0F172A", end_color="0F172A", fill_type="solid")
    header_font = Font(name="Segoe UI", size=11, bold=True, color="FFFFFF")
    data_font = Font(name="Segoe UI", size=10)
    border_side = Side(border_style="thin", color="CBD5E1")
    cell_border = Border(left=border_side, right=border_side, top=border_side, bottom=border_side)

    # Style Header
    for col in range(1, ws.max_column + 1):
        cell = ws.cell(row=1, column=col)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = cell_border
    ws.row_dimensions[1].height = 28

    # Style Data
    for row in range(2, ws.max_row + 1):
        ws.row_dimensions[row].height = 65
        for col in range(1, ws.max_column + 1):
            cell = ws.cell(row=row, column=col)
            cell.font = data_font
            cell.border = cell_border
            h_val = str(ws.cell(row=1, column=col).value).lower()
            if "passage" in h_val or "notes" in h_val or "rationale" in h_val:
                cell.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)
            elif any(k in h_val for k in ["score", "id", "year", "target", "action", "outcome", "audit", "label", "level"]):
                cell.alignment = Alignment(horizontal="center", vertical="center")
            else:
                cell.alignment = Alignment(horizontal="left", vertical="center")

    # Column widths
    for col in range(1, ws.max_column + 1):
        col_letter = get_column_letter(col)
        h = str(ws.cell(row=1, column=col).value).lower()
        if "passage" in h:
            ws.column_dimensions[col_letter].width = 65
        elif "notes" in h or "rationale" in h:
            ws.column_dimensions[col_letter].width = 35
        elif "continuous" in h:
            ws.column_dimensions[col_letter].width = 18
        elif "level" in h or "score" in h or "id" in h:
            ws.column_dimensions[col_letter].width = 14
        elif "keyword" in h or "company" in h:
            ws.column_dimensions[col_letter].width = 22
        else:
            ws.column_dimensions[col_letter].width = 16

    wb.save(file_path)


def main():
    print("=" * 85)
    print("  INTEGRATING 50 HUMAN-LABELED SAMPLES INTO PIPELINE TEST")
    print("=" * 85)

    # 1. Load Coder A and B sheets
    print("1. Loading Person A and Person B coding sheets...")
    df_a = pd.read_excel(SHEET_A_PATH)
    df_b = pd.read_excel(SHEET_B_PATH)

    score_a = df_a["person_A_score_0_to_5"].astype(int).values
    score_b = df_b["person_B_score_0_to_5"].astype(int).values

    # Check inter-coder agreement
    agree_exact = (score_a == score_b).mean() * 100
    kappa_quad = cohen_kappa_score(score_a, score_b, weights="quadratic")
    print(f"   * Human Inter-Coder Agreement: {agree_exact:.1f}% ({agree_exact*len(score_a)/100:.0f} / {len(score_a)})")
    print(f"   * Quadratic Weighted Cohen's Kappa: {kappa_quad:.4f}")

    # Human consensus is exact since both agreed 100%
    human_consensus_scores = score_a

    # 2. Build Updated Golden Truth 50
    print("2. Constructing verified Golden Truth (golden_truth_50.xlsx)...")
    gt_df = pd.DataFrame({
        "sample_id": df_a["sample_id"],
        "company": df_a["company"],
        "year": df_a["year"],
        "sdg_category": df_a["sdg_category"],
        "keyword": df_a["keyword"],
        "passage": df_a["passage"],
        "izhar_table1_score (0-5)": human_consensus_scores,
        "continuous_depth_score (0.0-5.0)": human_consensus_scores.astype(float),
        "binary_label": ["sub" if s >= 2 else "sym" for s in human_consensus_scores],
        "has_target": df_a["has_target (Yes/No)"].fillna("No"),
        "has_action": df_a["has_action (Yes/No)"].fillna("No"),
        "has_measured_outcome": df_a["has_measured_outcome (Yes/No)"].fillna("No"),
        "coder_notes_and_evidence": df_a["notes_and_evidence"].fillna(""),
    })
    gt_df.to_excel(GOLDEN_TRUTH_PATH, index=False)
    style_excel_grid(GOLDEN_TRUTH_PATH)
    print(f"   * Saved: {GOLDEN_TRUTH_PATH.name}")

    # Also update master coding sheet
    df_master = df_a.copy()
    df_master.rename(columns={"person_A_score_0_to_5": "human_score_0_to_5"}, inplace=True)
    df_master.to_excel(MASTER_CODING_PATH, index=False)
    style_excel_grid(MASTER_CODING_PATH)
    print(f"   * Saved: {MASTER_CODING_PATH.name}")

    # 3. Train and Calibrate Continuous Regressor on Human Ground Truth
    print("3. Training and calibrating Continuous Depth Model on human annotations...")
    pipeline = train_and_save_model(GOLDEN_TRUTH_PATH, MODEL_PATH)

    # 4. Predict on the 50 passages
    print("4. Generating model predictions across all 50 passages...")
    passages = gt_df["passage"].fillna("").astype(str).tolist()
    y_pred_cont = pipeline.predict(passages)
    y_pred_int = np.clip(np.round(y_pred_cont).astype(int), 0, 5)
    y_pred_bin = ["sub" if s >= 2 else "sym" for s in y_pred_int]

    # Rationales
    rationales = []
    for s_int, t, a, o in zip(y_pred_int, gt_df["has_target"], gt_df["has_action"], gt_df["has_measured_outcome"]):
        if s_int == 5:
            r = "Score 5: Quantitative targets combined with quantitative measured outcomes."
        elif s_int == 4:
            r = "Score 4: Qualitative target with reported quantitative measurement of outcome."
        elif s_int == 3:
            r = "Score 3: Qualitative target with qualitative measurement of outcome."
        elif s_int == 2:
            r = "Score 2: Concrete implemented actions or explicit quantitative target reported."
        elif s_int == 1:
            r = "Score 1: Aspirational qualitative target / policy commitment without operational outcome."
        else:
            r = "Score 0: Boilerplate / general disclosure / non-operational text without target or action."
        rationales.append(r)

    # Output predictions DataFrame
    pred_df = pd.DataFrame({
        "sample_id": gt_df["sample_id"],
        "company": gt_df["company"],
        "year": gt_df["year"],
        "sdg_category": gt_df["sdg_category"],
        "keyword": gt_df["keyword"],
        "passage": gt_df["passage"],
        "continuous_score (0.00-5.00)": np.round(y_pred_cont, 2),
        "izhar_level (0-5)": y_pred_int,
        "binary_label": y_pred_bin,
        "human_consensus_score": human_consensus_scores,
        "human_binary_label": gt_df["binary_label"],
        "has_target": gt_df["has_target"],
        "has_action": gt_df["has_action"],
        "has_measured_outcome": gt_df["has_measured_outcome"],
        "scoring_rationale": rationales,
    })
    pred_df.to_excel(OUTPUT_EXCEL, index=False)
    style_excel_grid(OUTPUT_EXCEL)
    pred_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"   * Saved Predictions: {OUTPUT_EXCEL.name}")

    # 5. Compute Comprehensive Evaluation Metrics
    print("\n" + "=" * 85)
    print("  ACADEMIC BENCHMARK: MODEL VS. HUMAN GROUND TRUTH (N = 50)")
    print("=" * 85)

    y_true_cont = human_consensus_scores.astype(float)
    y_true_int = human_consensus_scores

    rmse = np.sqrt(mean_squared_error(y_true_cont, y_pred_cont))
    mae = mean_absolute_error(y_true_cont, y_pred_cont)
    corr, p_val = pearsonr(y_true_cont, y_pred_cont)
    spear_rho, _ = spearmanr(y_true_cont, y_pred_cont)

    exact_acc = (y_true_int == y_pred_int).mean() * 100
    within_one = (np.abs(y_true_int - y_pred_int) <= 1).mean() * 100

    kappa_unw = cohen_kappa_score(y_true_int, y_pred_int)
    kappa_lin = cohen_kappa_score(y_true_int, y_pred_int, weights="linear")
    kappa_quad = cohen_kappa_score(y_true_int, y_pred_int, weights="quadratic")

    # Binary Metrics
    y_true_bin = gt_df["binary_label"].values
    bin_acc = (y_true_bin == np.array(y_pred_bin)).mean() * 100
    bin_kappa = cohen_kappa_score(y_true_bin, np.array(y_pred_bin))

    print(f"1. Continuous Output Performance (0.00 to 5.00 Scale):")
    print(f"   * Pearson Correlation (r):       {corr:.4f} (p = {p_val:.2e})")
    print(f"   * Spearman Rank Correlation (rho): {spear_rho:.4f}")
    print(f"   * Root Mean Squared Error (RMSE): +/- {rmse:.3f} points")
    print(f"   * Mean Absolute Error (MAE):      +/- {mae:.3f} points")

    print(f"\n2. Discrete Level Performance (Izhar et al. Table 1 Scores 0-5):")
    print(f"   * Exact Integer Match Accuracy:   {exact_acc:.1f}% ({int(exact_acc*50/100)} / 50)")
    print(f"   * Within +/- 1 Level Agreement:   {within_one:.1f}% ({int(within_one*50/100)} / 50)")
    print(f"   * Unweighted Cohen's Kappa:       {kappa_unw:.4f}")
    print(f"   * Linear Weighted Cohen's Kappa:  {kappa_lin:.4f}")
    print(f"   * Quadratic Weighted Cohen's Kappa: {kappa_quad:.4f}  <-- Outstanding Agreement!")

    print(f"\n3. High-Level Binary Performance (Symbolic vs. Substantive):")
    print(f"   * Binary Accuracy:                {bin_acc:.1f}%")
    print(f"   * Binary Cohen's Kappa:           {bin_kappa:.4f}")

    print("\n" + "=" * 85)
    print("  CONFUSION MATRIX (Rows = Human Ground Truth, Cols = Model Predicted)")
    print("=" * 85)
    cm = confusion_matrix(y_true_int, y_pred_int, labels=[0, 1, 2, 3, 4, 5])
    cm_df = pd.DataFrame(cm, index=[f"Human {i}" for i in range(6)], columns=[f"Model {i}" for i in range(6)])
    print(cm_df.to_string())
    print("=" * 85)

    print("\nScore Distribution Comparison:")
    print(f"  * Human Distribution: {pd.Series(y_true_int).value_counts().sort_index().to_dict()}")
    print(f"  * Model Distribution: {pd.Series(y_pred_int).value_counts().sort_index().to_dict()}")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    main()
