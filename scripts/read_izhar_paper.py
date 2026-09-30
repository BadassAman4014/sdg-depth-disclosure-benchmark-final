import fitz

doc = fitz.open("Papers/Izhar et al. - 2026 - Exploring firm-level SDG engagement in an emerging economy through content analysis.pdf")
with open("data/izhar_extracted_text.txt", "w", encoding="utf-8") as f:
    for i, page in enumerate(doc):
        text = page.get_text()
        if "Depth of SDG engagement" in text or "Table 1" in text or "Scoring criteria" in text:
            f.write(f"\n--- PAGE {i+1} ---\n")
            f.write(text)

print("Saved extracted text to data/izhar_extracted_text.txt")
