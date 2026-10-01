#!/usr/bin/env python3
"""
extract_text.py — Extract text from all downloaded annual report PDFs.

Usage:
    python scripts/extract_text.py [--workers N] [--force]

Reads:  reports/<Company>/<Year>/<*.pdf>
Writes: data/texts/<Company>/<Year>/results.txt
"""

import argparse
import os
import sys
import logging
from pathlib import Path
import re
import pymupdf
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────────
REPORTS_DIR = Path("reports")
OUTPUT_DIR  = Path("data/texts")
MIN_TEXT_LEN = 500  # skip PDFs that yield less than this many chars (likely scanned/corrupt)

CAESAR_WORDS_3 = {
    "WKH": "THE", "DQG": "AND", "IRU": "FOR", "DOO": "ALL", "WR": "TO",
    "RI": "OF", "LQ": "IN", "LV": "IS", "WKDW": "THAT", "WKLV": "THIS",
    "ZLWK": "WITH", "DV": "AS", "RQ": "ON", "DW": "AT", "IURP": "FROM",
    "KDYH": "HAVE", "EHHQ": "BEEN", "FRPSDQ\\": "COMPANY", "GLUHFWRUV": "DIRECTORS",
    "HPSOR\\HHV": "EMPLOYEES", "FRPSOLDQFH": "COMPLIANCE", "GLUHFWLYH": "DIRECTIVE"
}

def _decode_caesar_char_3(c: str) -> str:
    o = ord(c)
    if ord('A') <= o <= ord('Z'):
        return chr((o - ord('A') - 3) % 26 + ord('A'))
    elif ord('a') <= o <= ord('z'):
        return chr((o - ord('a') - 3) % 26 + ord('a'))
    elif c == '\\':
        return 'Y'
    elif c == '[':
        return 'X'
    elif c == ']':
        return 'Z'
    return c

def _decode_caesar_word_3(word: str) -> str:
    return "".join(_decode_caesar_char_3(c) for c in word.replace("ɝ", "ffi").replace("ȴ", "fi"))

def sanitize_page_text(raw: str) -> str:
    # 1. Clean control chars, null bytes, and common ligatures
    t = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', ' ', raw)
    t = t.replace("ɝ", "ffi").replace("ȴ", "fi").replace("\xad", "")
    
    # 2. Check for +3 Caesar shift (fonts lacking /ToUnicode CMaps)
    tokens = re.findall(r'[A-Za-z\\\[\]]+', t)
    caesar_hits = sum(1 for tok in tokens if tok.upper() in CAESAR_WORDS_3 or tok in CAESAR_WORDS_3)
    if caesar_hits >= 2 or (len(tokens) > 5 and caesar_hits / len(tokens) > 0.15):
        t = re.sub(r'[A-Za-z\\\[\]]+', lambda m: _decode_caesar_word_3(m.group(0)), t)
    else:
        for bad_w, good_w in CAESAR_WORDS_3.items():
            t = re.sub(r'(?<![A-Za-z])' + re.escape(bad_w) + r'(?![A-Za-z])', good_w, t, flags=re.IGNORECASE)
            
    # 3. Heal mid-word justification gaps
    t = re.sub(r'\b(sys)\s{2,}(tem)\b', r'\1\2', t, flags=re.I)
    t = re.sub(r'\b(ap)\s{2,}(proach)\b', r'\1\2', t, flags=re.I)
    t = re.sub(r'\b(compli)\s{2,}(ance)\b', r'\1\2', t, flags=re.I)
    t = re.sub(r'\b(busi)\s{2,}(ness)\b', r'\1\2', t, flags=re.I)
    t = re.sub(r'\b(require)\s{2,}(ments)\b', r'\1\2', t, flags=re.I)
    t = re.sub(r'\b(employ)\s{2,}(ees)\b', r'\1\2', t, flags=re.I)
    return t

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract and sanitize text from a PDF using PyMuPDF."""
    text_blocks = []
    with pymupdf.open(str(pdf_path)) as doc:
        for page in doc:
            page_text = page.get_text("text")
            if page_text:
                text_blocks.append(sanitize_page_text(page_text))
    return "\n".join(text_blocks).strip()


def process_one_worker(item: tuple) -> dict:
    pdf_path_str, out_dir_str, force = item
    pdf_path = Path(pdf_path_str)
    out_dir = Path(out_dir_str)
    out_file = out_dir / "results.txt"

    if out_file.exists() and not force:
        return {"path": str(pdf_path), "status": "skipped", "chars": 0}

    try:
        text = extract_text_from_pdf(pdf_path)
        if len(text) < MIN_TEXT_LEN:
            return {
                "path": str(pdf_path),
                "status": "too_short",
                "chars": len(text),
            }

        out_dir.mkdir(parents=True, exist_ok=True)
        out_file.write_text(text, encoding="utf-8")
        return {"path": str(pdf_path), "status": "ok", "chars": len(text)}

    except Exception as e:
        return {"path": str(pdf_path), "status": "error", "chars": 0, "error": str(e)}


def discover_pdfs(reports_dir: Path) -> list[tuple[Path, Path]]:
    """Discover all (pdf_path, output_dir) pairs."""
    pairs = []
    for company_dir in sorted(reports_dir.iterdir()):
        if not company_dir.is_dir():
            continue
        for year_dir in sorted(company_dir.iterdir()):
            if not year_dir.is_dir():
                continue
            for pdf in sorted(year_dir.glob("*.pdf")):
                out_dir = OUTPUT_DIR / company_dir.name / year_dir.name
                pairs.append((pdf, out_dir))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Extract text from annual report PDFs")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--force", action="store_true", help="Re-extract even if output exists")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pairs = discover_pdfs(REPORTS_DIR)
    log.info(f"Found {len(pairs)} PDFs to process")

    stats = {"ok": 0, "skipped": 0, "too_short": 0, "error": 0}
    errors = []

    work_items = [(str(p), str(o), args.force) for p, o in pairs]

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_one_worker, item): item for item in work_items}
        with tqdm(total=len(futures), desc="Extracting text", unit="pdf") as pbar:
            for fut in as_completed(futures):
                try:
                    result = fut.result()
                    stats[result["status"]] += 1
                    if result["status"] == "error":
                        errors.append(result)
                        log.error(f"FAILED: {result['path']} — {result.get('error', '?')}")
                except Exception as ex:
                    stats["error"] += 1
                    log.error(f"Process error: {ex}")
                pbar.update(1)

    log.info(
        f"\nDone. OK={stats['ok']}  Skipped={stats['skipped']}  "
        f"TooShort={stats['too_short']}  Errors={stats['error']}"
    )
    if errors:
        log.warning(f"\n{len(errors)} errors:")
        for e in errors:
            log.warning(f"  {e['path']}: {e.get('error', '?')}")


if __name__ == "__main__":
    main()
