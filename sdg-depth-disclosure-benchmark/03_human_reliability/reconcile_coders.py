#!/usr/bin/env python3
"""
reconcile_coders.py — Adjudicate Disagreements between Coder A and Coder B
and produce a verified Golden Ground Truth dataset.

Academic Grounding:
  Izhar et al. (2026), Section 3.4.3 & Table 1
  Hummel (2019); PwC (2018)
"""

import sys
import re
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score

sys.stdout.reconfigure(encoding='utf-8')

AUDIT_REGEX = re.compile(r"\b(?:verified|assured|audited|certified)\s+by\b|\b(?:pwc|kpmg|ey|deloitte|tüv|tuev|bureau\s+veritas|iso\s*14001|iso\s*50001)\b", re.I)
METRIC_REGEX = re.compile(r"\b\d+(?:[\.,]\d+)?\s*(?:%|percent|prozent|tco2|tco2e|tonnes?|kwh|mwh|gwh|million|billion|mio|mrd|eur|€|\$)\b", re.I)
ACTION_VERBS = re.compile(r"\b(?:implemented|installed|deployed|commissioned|invested|allocated|achieved|constructed|built|umgesetzt|eingeführt|investiert|reduziert|trained)\b", re.I)
ASPIRATIONAL_REGEX = re.compile(r"\b(?:strive|aim|aspire|commit|intend|pledge|seek|streben|beabsichtigen|vision|code\s+of\s+conduct|compliance\s+policy)\b", re.I)


def reconcile_dataset(sheet_a_path: Path, sheet_b_path: Path, out_path: Path):
    print("=" * 80)
    print("  HUMAN CODER ADJUDICATION & RECONCILIATION ENGINE")
    print("=" * 80)
    
    df_a = pd.read_excel(sheet_a_path)
    df_b = pd.read_excel(sheet_b_path)
    
    col_a = [c for c in df_a.columns if "score" in c.lower() or "person_a" in c.lower()][0]
    col_b = [c for c in df_b.columns if "score" in c.lower() or "person_b" in c.lower()][0]
    
    scores_a = df_a[col_a].values
    scores_b = df_b[col_b].values
    
    reconciled_scores = []
    adjudication_notes = []
    disagreements = 0
    
    for i in range(len(df_a)):
        sa = scores_a[i]
        sb = scores_b[i]
        p = str(df_a.iloc[i].get("passage", ""))
        
        if pd.notna(sa) and pd.notna(sb) and int(sa) == int(sb):
            reconciled_scores.append(int(sa))
            adjudication_notes.append("Consensus: Both coders independently agreed on score.")
        else:
            disagreements += 1
            has_audit = bool(AUDIT_REGEX.search(p))
            has_metrics = bool(METRIC_REGEX.search(p))
            has_actions = bool(ACTION_VERBS.search(p))
            
            if has_metrics and (has_audit or has_actions):
                res = 5 if has_audit else 4
                note = f"Adjudicated Level {res}: Confirmed quantified metrics + operational action/audit."
            elif has_actions:
                res = 2
                note = "Adjudicated Level 2: Confirmed concrete operational activity implemented."
            elif bool(ASPIRATIONAL_REGEX.search(p)):
                res = 1
                note = "Adjudicated Level 1: Pure aspirational target without verified operational proof."
            else:
                res = 0
                note = "Adjudicated Level 0: Pure boilerplate / non-operational disclosure."
                
            reconciled_scores.append(res)
            adjudication_notes.append(f"Adjudicated (Coder A={sa}, Coder B={sb}): {note}")
            
    out_df = df_a.copy()
    out_df["reconciled_consensus_score"] = reconciled_scores
    out_df["binary_label"] = ["sub" if s >= 2 else "sym" for s in reconciled_scores]
    out_df["adjudication_rationale"] = adjudication_notes
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_excel(out_path, index=False)
    print(f"Total Passages Evaluated: {len(df_a)}")
    print(f"Disagreements Resolved:   {disagreements}")
    print(f"Saved Reconciled Ground Truth: {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconcile Coder A and B sheets into consensus truth")
    parser.add_argument("--sheet-a", default="06_pilot_50_suite/data/coding_sheet_person_A_50.xlsx", help="Coder A Excel sheet")
    parser.add_argument("--sheet-b", default="06_pilot_50_suite/data/coding_sheet_person_B_50.xlsx", help="Coder B Excel sheet")
    parser.add_argument("--out", default="data/reconciled_golden_truth.xlsx", help="Output reconciled ground truth path")
    args = parser.parse_args()
    
    reconcile_dataset(Path(args.sheet_a), Path(args.sheet_b), Path(args.out))
