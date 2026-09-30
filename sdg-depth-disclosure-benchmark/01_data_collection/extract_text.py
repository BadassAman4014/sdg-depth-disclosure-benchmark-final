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
import pymupdf
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────────
REPORTS_DIR = Path("reports")
OUTPUT_DIR  = Path("data/texts")
MIN_TEXT_LEN = 500  # skip PDFs that yield less than this many chars (likely scanned/corrupt)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("data/extract_text.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract all text from a PDF using PyMuPDF."""
    text = ""
    with pymupdf.open(str(pdf_path)) as doc:
        for page in doc:
            text += "\n" + page.get_text()
    return text.strip()


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
