#!/usr/bin/env python3
"""
split_passages.py — Split extracted text files into clean semantic/sentence passages.

Uses NLTK sentence tokenization + text cleaning + greedy 512-token packing (tiktoken),
matching the NLTKSplitter from AI-For-Sustainability/src/preprocessing/splitter.py.

Usage:
    python scripts/split_passages.py [--workers N] [--force]

Reads:  data/texts/<Company>/<Year>/results.txt
Writes: data/jsons/<Company>/<Year>/splits_semantic.json
"""

import argparse
import json
import os
import re
import sys
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import nltk
import tiktoken
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────────
TEXTS_DIR  = Path("data/texts")
OUTPUT_DIR = Path("data/jsons")
MAX_CHUNK_TOKENS = 512
MIN_CHUNK_TOKENS = 10

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("data/split_passages.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ── Tokenizer ──────────────────────────────────────────────────────────────────

_tokenizer = None

def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = tiktoken.get_encoding("cl100k_base")
    return _tokenizer


def count_tokens(text: str) -> int:
    enc = get_tokenizer()
    return len(enc.encode(text))


def hard_cap_by_tokens(text: str, max_tokens: int) -> list[str]:
    enc = get_tokenizer()
    ids = enc.encode(text)
    return [enc.decode(ids[i:i+max_tokens]) for i in range(0, len(ids), max_tokens)]


def pack_sentences_greedy_strict(sentences: list[str], max_tokens: int) -> list[str]:
    """Greedily pack sentences into chunks <= max_tokens."""
    enc = get_tokenizer()
    chunks, cur_ids = [], []
    for s in sentences:
        s_ids = enc.encode(s)
        if len(s_ids) > max_tokens:
            if cur_ids:
                chunks.append(enc.decode(cur_ids))
                cur_ids = []
            chunks.extend(hard_cap_by_tokens(s, max_tokens))
            continue
        if len(cur_ids) + len(s_ids) <= max_tokens:
            cur_ids.extend(s_ids)
        else:
            if cur_ids:
                chunks.append(enc.decode(cur_ids))
            cur_ids = list(s_ids)
    if cur_ids:
        chunks.append(enc.decode(cur_ids))
    return chunks


# ── Text cleaning (from AI-For-Sustainability) ────────────────────────────────

def clean_pdf_text(raw_text: str) -> str:
    """Clean common PDF artifacts."""
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"Page\s+\d+", "", text)       # drop "Page 12"
    text = re.sub(r"-\n", "", text)              # de-hyphenate
    text = re.sub(r"\n([a-z])", r" \1", text)    # join broken lines mid-sentence
    text = re.sub(r"\n\s*\n+", "\n\n", text)     # collapse extra blank lines
    text = re.sub(r"[ \t]+", " ", text)          # collapse spaces
    text = text.replace("\u00ad", "").replace("\u2009", "").strip()
    return text


def split_text_into_passages(raw_text: str) -> list[str]:
    """Clean text, tokenize sentences, and pack into <=512 token passages."""
    cleaned = clean_pdf_text(raw_text)
    try:
        sentences = [s.strip() for s in nltk.sent_tokenize(cleaned) if s.strip()]
    except Exception:
        # Fallback simple regex split
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', cleaned) if s.strip()]

    if not sentences:
        return []

    chunks = pack_sentences_greedy_strict(sentences, MAX_CHUNK_TOKENS)

    # Filter out tiny noise chunks (<10 tokens)
    valid_chunks = [c for c in chunks if count_tokens(c) >= MIN_CHUNK_TOKENS]
    return valid_chunks


# ── Worker function ────────────────────────────────────────────────────────────

def process_one_file(item: tuple) -> dict:
    txt_path_str, out_path_str, force = item
    txt_path = Path(txt_path_str)
    out_path = Path(out_path_str)
    out_file = out_path / "splits_semantic.json"

    if out_file.exists() and not force:
        return {"path": str(txt_path), "status": "skipped", "passages": 0}

    try:
        text = txt_path.read_text(encoding="utf-8", errors="replace")
        if len(text.strip()) < 100:
            return {"path": str(txt_path), "status": "too_short", "passages": 0}

        passages = split_text_into_passages(text)
        result = {str(i): p for i, p in enumerate(passages)}

        out_path.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        return {"path": str(txt_path), "status": "ok", "passages": len(result)}

    except Exception as e:
        return {"path": str(txt_path), "status": "error", "passages": 0, "error": str(e)}


def discover_texts(texts_dir: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for company_dir in sorted(texts_dir.iterdir()):
        if not company_dir.is_dir():
            continue
        for year_dir in sorted(company_dir.iterdir()):
            if not year_dir.is_dir():
                continue
            txt = year_dir / "results.txt"
            if txt.exists():
                out_dir = OUTPUT_DIR / company_dir.name / year_dir.name
                pairs.append((txt, out_dir))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Split text files into passages")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--force", action="store_true", help="Re-split even if output exists")
    args = parser.parse_args()

    # Ensure punkt is downloaded
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pairs = discover_texts(TEXTS_DIR)
    log.info(f"Found {len(pairs)} text files to split")

    if not pairs:
        log.warning("No text files found! Run extract_text.py first.")
        return

    stats = {"ok": 0, "skipped": 0, "too_short": 0, "error": 0}
    total_passages = 0
    errors = []

    work_items = [(str(t), str(o), args.force) for t, o in pairs]

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_one_file, item): item for item in work_items}
        with tqdm(total=len(futures), desc="Splitting passages", unit="file") as pbar:
            for fut in as_completed(futures):
                try:
                    result = fut.result()
                    stats[result["status"]] += 1
                    total_passages += result["passages"]
                    if result["status"] == "error":
                        errors.append(result)
                        log.error(f"FAILED: {result['path']} — {result.get('error', '?')}")
                except Exception as ex:
                    stats["error"] += 1
                    log.error(f"Worker exception: {ex}")
                pbar.update(1)

    log.info(
        f"\nDone. OK={stats['ok']}  Skipped={stats['skipped']}  "
        f"TooShort={stats['too_short']}  Errors={stats['error']}  "
        f"TotalPassages={total_passages}"
    )


if __name__ == "__main__":
    main()
