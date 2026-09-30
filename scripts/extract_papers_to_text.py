#!/usr/bin/env python3
"""
extract_papers_to_text.py — Extract the full text of all papers into markdown files.
"""

import pymupdf
from pathlib import Path

def main():
    out_dir = Path("data/paper_summaries")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    papers = {
        "de_kok_2025.md": "Papers/De Kok - 2025 - ChatGPT for Textual Analysis How to Use Generative LLMs in Accounting Research.pdf",
        "shah_et_al_2020.md": "Papers/Shah et al. - 2020 - A Comparative Analysis of Logistic Regression, Random Forest and KNN Models for the Text Classificat.pdf",
        "mnsc_supplement.md": "Papers/mnsc.2023.03253.sm1.pdf"
    }
    
    for out_name, pdf_path in papers.items():
        doc = pymupdf.open(pdf_path)
        out_file = out_dir / out_name
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(f"# Paper: {Path(pdf_path).name}\n\n")
            f.write(f"Total Pages: {len(doc)}\n\n")
            for i, page in enumerate(doc):
                f.write(f"## Page {i+1}\n\n")
                f.write(page.get_text() + "\n\n")
        print(f"Extracted {pdf_path} -> {out_file}")

if __name__ == "__main__":
    main()
