#!/usr/bin/env python3
"""
read_papers.py — Extract and summarize the methodology from the three papers in the Papers folder.
"""

import pymupdf
from pathlib import Path

def inspect_paper(pdf_path: str):
    doc = pymupdf.open(pdf_path)
    print(f"\n{'='*80}")
    print(f"FILE: {Path(pdf_path).name}")
    print(f"Total Pages: {len(doc)}")
    print(f"{'='*80}\n")
    
    full_text = ""
    for i, page in enumerate(doc):
        full_text += f"\n--- Page {i+1} ---\n" + page.get_text()
    
    return full_text

if __name__ == "__main__":
    papers = [
        "Papers/De Kok - 2025 - ChatGPT for Textual Analysis How to Use Generative LLMs in Accounting Research.pdf",
        "Papers/Shah et al. - 2020 - A Comparative Analysis of Logistic Regression, Random Forest and KNN Models for the Text Classificat.pdf",
        "Papers/mnsc.2023.03253.sm1.pdf",
    ]
    
    for p in papers:
        text = inspect_paper(p)
        # Search for key sections
        keywords = ["validation", "reliability", "ground truth", "sample", "logistic", "kappa", "alpha", "inter-coder", "human", "golden", "benchmarking"]
        print(f"Summary of key sections for {Path(p).name}:")
        for kw in keywords:
            count = text.lower().count(kw)
            print(f"  - Mention of '{kw}': {count} times")
