#!/usr/bin/env python3
"""
setup_test_data.py — Generates the 50-sample dataset, coding sheets, and ground truth for Pipeline Test.
"""

import sys
import re
import pandas as pd
from pathlib import Path
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True, parents=True)

SOURCE_500 = Path(__file__).resolve().parent.parent / "data" / "golden_dataset_500_continuous_0_to_5.xlsx"

# Regex rules for Izhar et al. (Table 1) components
QUAL_TARGET_RE = re.compile(r"\b(?:strive|aim|aspire|commit|intend|pledge|endeavor|target|vision|seek|goal|plan|streben|beabsichtigen)\b", re.I)
QUANT_TARGET_RE = re.compile(r"\b(?:target|aim|reduce|cut|reach)\s+(?:by|to|of)?\s*\d+(?:[\.,]\d+)?\s*(?:%|percent|prozent|tco2|kwh|tons?|tonnes?|eur|€|\$|by\s+20\d\d)\b", re.I)
ACTION_RE = re.compile(r"\b(?:installed|implemented|deployed|commissioned|invested|allocated|constructed|built|initiated|trained|upgraded|equipped|umgesetzt|eingeführt|investiert|reduziert)\b", re.I)
QUAL_OUTCOME_RE = re.compile(r"\b(?:resulted in|led to|improved|enhanced|strengthened|achieved progress|positive impact|verbessert|gestärkt|erfolgreich)\b", re.I)
QUANT_OUTCOME_RE = re.compile(r"\b(?:reduced|decreased|cut|saved|generated|sourced)\s+(?:by\s+)?\d+(?:[\.,]\d+)?\s*(?:%|percent|prozent|tco2|tco2e|kwh|mwh|gwh|tonnes?|million|billion|mio|mrd|€|\$)\b", re.I)
AUDIT_RE = re.compile(r"\b(?:verified|assured|audited|certified)\s+by\b|\b(?:pwc|kpmg|ey|deloitte|tüv|tuev|bureau\s+veritas|iso\s*14001|iso\s*50001)\b", re.I)


def extract_izhar_table1_score(passage: str, consensus_label: str) -> tuple[int, float, dict]:
    """Computes exact integer score (0-5) and continuous depth score per Table 1."""
    p_str = str(passage)
    
    comp = {
        "qual_target": bool(QUAL_TARGET_RE.search(p_str)),
        "quant_target": bool(QUANT_TARGET_RE.search(p_str)),
        "action": bool(ACTION_RE.search(p_str)),
        "qual_outcome": bool(QUAL_OUTCOME_RE.search(p_str)),
        "quant_outcome": bool(QUANT_OUTCOME_RE.search(p_str)),
        "audit": bool(AUDIT_RE.search(p_str)),
    }
    
    # Izhar Table 1 exact integer logic:
    if consensus_label == "sym":
        if comp["qual_target"]:
            score_int = 1
            cont_score = 1.0 + (0.2 if comp["action"] else 0.0)
        else:
            score_int = 0
            cont_score = 0.0 + (0.3 if comp["qual_outcome"] else 0.0)
    else:  # sub
        if (comp["quant_target"] or comp["audit"]) and comp["quant_outcome"]:
            score_int = 5
            cont_score = 4.8 + (0.2 if comp["audit"] else 0.0)
        elif comp["qual_target"] and comp["quant_outcome"]:
            score_int = 4
            cont_score = 4.0 + (0.3 if comp["audit"] else 0.0)
        elif comp["qual_target"] and comp["qual_outcome"]:
            score_int = 3
            cont_score = 3.0 + (0.2 if comp["action"] else 0.0)
        elif comp["quant_target"] or (comp["qual_target"] and comp["action"]):
            score_int = 2
            cont_score = 2.0 + (0.3 if comp["action"] else 0.0)
        elif comp["action"]:
            score_int = 2
            cont_score = 2.2
        else:
            score_int = 2
            cont_score = 2.0
            
    return score_int, round(cont_score, 2), comp


def style_excel(file_path: Path, title: str):
    wb = openpyxl.load_workbook(file_path)
    ws = wb.active
    ws.views.sheetView[0].showGridLines = True

    header_fill = PatternFill(start_color="1E293B", end_color="1E293B", fill_type="solid")
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
            header_val = str(ws.cell(row=1, column=col).value).lower()
            if "passage" in header_val:
                cell.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)
            elif any(k in header_val for k in ["score", "id", "year", "target", "action", "outcome", "audit"]):
                cell.alignment = Alignment(horizontal="center", vertical="center")
            else:
                cell.alignment = Alignment(horizontal="left", vertical="center")

    # Set column widths
    for col in range(1, ws.max_column + 1):
        col_letter = get_column_letter(col)
        h = str(ws.cell(row=1, column=col).value).lower()
        if "passage" in h:
            ws.column_dimensions[col_letter].width = 65
        elif "id" in h or "year" in h or "score" in h:
            ws.column_dimensions[col_letter].width = 14
        elif "keyword" in h or "company" in h:
            ws.column_dimensions[col_letter].width = 22
        elif "target" in h or "action" in h or "outcome" in h:
            ws.column_dimensions[col_letter].width = 16
        else:
            ws.column_dimensions[col_letter].width = 18

    wb.save(file_path)


def main():
    print(f"Loading source dataset from {SOURCE_500}...")
    df_full = pd.read_excel(SOURCE_500)
    
    # Take first 50 stratified samples
    df_50 = df_full.head(50).copy()
    df_50["sample_id"] = range(1, 51)
    
    # Compute ground truth Table 1 components and scores
    int_scores = []
    cont_scores = []
    comp_list = []
    for _, row in df_50.iterrows():
        p = str(row["passage"])
        c = str(row["human_consensus"]).strip().lower()
        s_int, s_cont, comp = extract_izhar_table1_score(p, c)
        int_scores.append(s_int)
        cont_scores.append(s_cont)
        comp_list.append(comp)

    comp_df = pd.DataFrame(comp_list)
    
    # 1. Master Ground Truth 50
    gt_df = pd.DataFrame({
        "sample_id": df_50["sample_id"],
        "company": df_50["company"],
        "year": df_50["year"],
        "sdg_category": df_50["sdg_category"],
        "keyword": df_50["keyword"],
        "passage": df_50["passage"],
        "has_qual_target": comp_df["qual_target"].map({True: "Yes", False: "No"}),
        "has_quant_target": comp_df["quant_target"].map({True: "Yes", False: "No"}),
        "has_action": comp_df["action"].map({True: "Yes", False: "No"}),
        "has_qual_outcome": comp_df["qual_outcome"].map({True: "Yes", False: "No"}),
        "has_quant_outcome": comp_df["quant_outcome"].map({True: "Yes", False: "No"}),
        "has_audit": comp_df["audit"].map({True: "Yes", False: "No"}),
        "izhar_table1_score (0-5)": int_scores,
        "continuous_depth_score (0.0-5.0)": cont_scores,
        "binary_label": ["sub" if s >= 2 else "sym" for s in int_scores],
    })
    gt_path = DATA_DIR / "golden_truth_50.xlsx"
    gt_df.to_excel(gt_path, index=False)
    style_excel(gt_path, "Golden Truth 50")
    print(f"Created: {gt_path}")

    # 2. Raw Samples 50
    samples_df = df_50[["sample_id", "company", "year", "sdg_category", "keyword", "passage"]].copy()
    samples_path = DATA_DIR / "samples_50.xlsx"
    samples_df.to_excel(samples_path, index=False)
    style_excel(samples_path, "Samples 50")
    print(f"Created: {samples_path}")

    # 3. Coding Sheet for Person A (0-5 Rubric)
    sheet_a = samples_df.copy()
    sheet_a["Person_A_Score (0-5)"] = ""
    sheet_a["Has_Target (Yes/No)"] = ""
    sheet_a["Has_Action (Yes/No)"] = ""
    sheet_a["Has_Measured_Outcome (Yes/No)"] = ""
    sheet_a["Notes_and_Evidence"] = ""
    sheet_a_path = DATA_DIR / "coding_sheet_person_A_50.xlsx"
    sheet_a.to_excel(sheet_a_path, index=False)
    style_excel(sheet_a_path, "Coding Sheet Person A")
    print(f"Created: {sheet_a_path}")

    # 4. Coding Sheet for Person B (0-5 Rubric)
    sheet_b = samples_df.copy()
    sheet_b["Person_B_Score (0-5)"] = ""
    sheet_b["Has_Target (Yes/No)"] = ""
    sheet_b["Has_Action (Yes/No)"] = ""
    sheet_b["Has_Measured_Outcome (Yes/No)"] = ""
    sheet_b["Notes_and_Evidence"] = ""
    sheet_b_path = DATA_DIR / "coding_sheet_person_B_50.xlsx"
    sheet_b.to_excel(sheet_b_path, index=False)
    style_excel(sheet_b_path, "Coding Sheet Person B")
    print(f"Created: {sheet_b_path}")

    print("\nAll 50-sample test datasets and coding sheets generated successfully!")


if __name__ == "__main__":
    main()
