#!/usr/bin/env python3
"""
dump_paper_summaries.py — Extract detailed sections from each paper.
"""

import pymupdf
from pathlib import Path

def print_paper_content(pdf_path: str, max_chars=4000):
    doc = pymupdf.open(pdf_path)
    print(f"\n{'#'*80}")
    print(f"  PAPER: {Path(pdf_path).name}")
    print(f"{'#'*80}\n")
    
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    
    # Print first few pages & sections
    print("--- TITLE & ABSTRACT & INTRO ---")
    print(text[:max_chars])
    
    # Search for evaluation / validation / methodology sections
    print("\n--- METHODOLOGY & EVALUATION SECTIONS ---")
    lines = text.split("\n")
    for i, l in enumerate(lines):
        if any(h in l.lower() for h in ["evaluation", "method", "accuracy", "classifier", "validation", "benchmark", "coder", "intercoder", "reliability", "k-nearest", "logistic regression", "random forest"]):
            context = "\n".join(lines[max(0, i-2):min(len(lines), i+10)])
            print(f"\n[Line {i}] Match:")
            print(context)
            print("-" * 40)

if __name__ == "__main__":
    for p in [
        "Papers/De Kok - 2025 - ChatGPT for Textual Analysis How to Use Generative LLMs in Accounting Research.pdf",
        "Papers/Shah et al. - 2020 - A Comparative Analysis of Logistic Regression, Random Forest and KNN Models for the Text Classificat.pdf",
        "Papers/mnsc.2023.03253.sm1.pdf"
    ]:
        print_paper_content(p, 2500)
