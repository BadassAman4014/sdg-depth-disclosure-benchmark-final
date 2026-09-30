#!/usr/bin/env python3
"""
setup_two_coders.py — Create separate coding sheets for Person A and Person B.

Creates:
  1. data/coding_sheet_person_A.xlsx
  2. data/coding_sheet_person_B.xlsx
"""

import pandas as pd
from pathlib import Path

DATASET_FILE = Path("data/golden_dataset_500.xlsx")

def main():
    df = pd.read_excel(DATASET_FILE)
    
    # Base columns to show annotators
    base_cols = ["sample_id", "passage", "keyword", "sdg_category", "company", "year"]
    
    # Person A sheet
    df_a = df[base_cols].copy()
    df_a["my_label_A (sym or sub)"] = ""
    df_a["notes"] = ""
    df_a.to_excel("data/coding_sheet_person_A.xlsx", index=False, engine="openpyxl")
    
    # Person B sheet
    df_b = df[base_cols].copy()
    df_b["my_label_B (sym or sub)"] = ""
    df_b["notes"] = ""
    df_b.to_excel("data/coding_sheet_person_B.xlsx", index=False, engine="openpyxl")
    
    print("Created data/coding_sheet_person_A.xlsx (for Person A)")
    print("Created data/coding_sheet_person_B.xlsx (for Person B)")

if __name__ == "__main__":
    main()
