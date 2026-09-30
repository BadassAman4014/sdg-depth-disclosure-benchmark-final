#!/usr/bin/env python3
"""
predict_depth.py — Run inference on new corporate passages and assign 0-5 depth scores.

Outputs:
  - Continuous depth score in [0.00, 5.00]
  - Discrete integer level in {0, 1, 2, 3, 4, 5} (Izhar et al. 2026, Table 1)
  - High-level binary classification ('sym' vs 'sub')
  - Automated natural language rationale explaining the score
"""

import sys
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

from continuous_model import ContinuousDepthPipeline

sys.stdout.reconfigure(encoding='utf-8')


def build_rationale(score_int: int) -> str:
    if score_int == 5:
        return "Score 5: Quantitative targets combined with quantitative measured outcomes."
    elif score_int == 4:
        return "Score 4: Qualitative target with reported quantitative measurement of outcome."
    elif score_int == 3:
        return "Score 3: Qualitative target with qualitative measurement of outcome."
    elif score_int == 2:
        return "Score 2: Concrete implemented actions or explicit quantitative target reported."
    elif score_int == 1:
        return "Score 1: Aspirational qualitative target / policy commitment without operational outcome."
    else:
        return "Score 0: Boilerplate / general disclosure / non-operational text without target or action."


def predict_passages(model_path: Path, input_path: Path, output_excel: Path):
    print("=" * 80)
    print("  PREDICTING CONTINUOUS & ORDINAL SDG DEPTH (Izhar et al. 2026)")
    print("=" * 80)
    
    pipeline = joblib.load(model_path)
    
    if input_path.suffix == ".csv":
        df = pd.read_csv(input_path)
    else:
        df = pd.read_excel(input_path)
        
    passages = df["passage"].fillna("").astype(str).tolist()
    print(f"Loaded {len(passages)} passages from {input_path}")
    
    preds_cont = pipeline.predict(passages)
    preds_int = np.clip(np.round(preds_cont).astype(int), 0, 5)
    preds_bin = ["sub" if s >= 2 else "sym" for s in preds_int]
    rationales = [build_rationale(s) for s in preds_int]
    
    out_df = df.copy()
    out_df["continuous_score (0.00-5.00)"] = np.round(preds_cont, 2)
    out_df["izhar_level (0-5)"] = preds_int
    out_df["binary_label"] = preds_bin
    out_df["scoring_rationale"] = rationales
    
    output_excel.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_excel(output_excel, index=False)
    
    csv_out = output_excel.with_suffix(".csv")
    out_df.to_csv(csv_out, index=False, encoding="utf-8-sig")
    
    print(f"Predictions successfully exported:")
    print(f"  * Excel: {output_excel}")
    print(f"  * CSV:   {csv_out}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict depth scores for corporate passages")
    parser.add_argument("--model", default="06_pilot_50_suite/model_depth_50.joblib", help="Trained model path")
    parser.add_argument("--input", default="06_pilot_50_suite/data/samples_50.xlsx", help="Input Excel or CSV with 'passage' column")
    parser.add_argument("--out", default="data/predictions_output.xlsx", help="Output file path")
    args = parser.parse_args()
    
    predict_passages(Path(args.model), Path(args.input), Path(args.out))
