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


# ── Text cleaning & Gibberish Elimination ─────────────────────────────────────

MOJIBAKE_MAP = {
    "├ñ": "ä", "├╢": "ö", "├╝": "ü", "├ƒ": "ß",
    "├ä": "Ä", "├Ц": "Ö", "├Ь": "Ü",
    "Ã¤": "ä", "Ã¶": "ö", "Ã¼": "ü", "ÃŸ": "ß",
    "Ã„": "Ä", "Ã–": "Ö", "Ãœ": "Ü"
}

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

def is_table_debris(text: str) -> bool:
    """Filter out financial tables with no narrative sentences."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return True
    unit_lines = sum(1 for l in lines if re.match(r'^(€k|€m|kEUR|TEUR|\$m|€|\$|%|in €k|in %)$', l, re.I))
    if unit_lines >= 3:
        return True
    words = re.findall(r'[A-Za-zÄÖÜäöüß]{2,}', text)
    numbers = re.findall(r'\b\d+([.,]\d+)?\b', text)
    if len(words) < 15 and len(numbers) > len(words):
        return True
    accounting_noise = ["Book value", "Cost of acquisition", "Measurement according to IAS", "Trade accounts receivable"]
    if sum(1 for term in accounting_noise if term.lower() in text.lower()) >= 2 and len(words) < 40:
        return True
    return False

def clean_pdf_text(raw_text: str) -> str:
    """Clean common PDF artifacts, Caesar shifts, and Mojibake."""
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    
    # Mojibake
    for bad, good in MOJIBAKE_MAP.items():
        if bad in text:
            text = text.replace(bad, good)
            
    # Control chars & ligatures
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', ' ', text)
    text = text.replace("ɝ", "ffi").replace("ȴ", "fi").replace("\u00ad", "").replace("\u2009", " ")
    
    # Caesar shift (+3)
    tokens = re.findall(r'[A-Za-z\\\[\]]+', text)
    caesar_hits = sum(1 for tok in tokens if tok.upper() in CAESAR_WORDS_3 or tok in CAESAR_WORDS_3)
    if caesar_hits >= 2 or (len(tokens) > 5 and caesar_hits / len(tokens) > 0.15):
        text = re.sub(r'[A-Za-z\\\[\]]+', lambda m: _decode_caesar_word_3(m.group(0)), text)
    else:
        for bad_w, good_w in CAESAR_WORDS_3.items():
            text = re.sub(r'(?<![A-Za-z])' + re.escape(bad_w) + r'(?![A-Za-z])', good_w, text, flags=re.IGNORECASE)

    # De-hyphenate and line structure
    text = re.sub(r"Page\s+\d+", "", text)
    text = re.sub(r"-\n", "", text)
    text = re.sub(r"\n([a-z])", r" \1", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    
    # Heal mid-word splits
    text = re.sub(r'\b(sys)\s{2,}(tem)\b', r'\1\2', text, flags=re.I)
    text = re.sub(r'\b(ap)\s{2,}(proach)\b', r'\1\2', text, flags=re.I)
    text = re.sub(r'\b(compli)\s{2,}(ance)\b', r'\1\2', text, flags=re.I)
    text = re.sub(r'\b(busi)\s{2,}(ness)\b', r'\1\2', text, flags=re.I)
    text = re.sub(r'\b(require)\s{2,}(ments)\b', r'\1\2', text, flags=re.I)
    text = re.sub(r'\b(employ)\s{2,}(ees)\b', r'\1\2', text, flags=re.I)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def split_text_into_passages(raw_text: str) -> list[str]:
    """Clean text, tokenize sentences, and pack into <=512 token passages, filtering table noise."""
    cleaned = clean_pdf_text(raw_text)
    try:
        sentences = [s.strip() for s in nltk.sent_tokenize(cleaned) if s.strip()]
    except Exception:
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', cleaned) if s.strip()]

    if not sentences:
        return []

    chunks = pack_sentences_greedy_strict(sentences, MAX_CHUNK_TOKENS)

    # Filter out tiny noise chunks (<10 tokens) and non-narrative accounting table debris
    valid_chunks = [c for c in chunks if count_tokens(c) >= MIN_CHUNK_TOKENS and not is_table_debris(c)]
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
