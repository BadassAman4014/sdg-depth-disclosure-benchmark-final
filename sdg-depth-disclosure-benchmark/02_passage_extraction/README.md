# 🔍 Task 02: Passage Extraction & Multilingual SDG Matching

This module handles the segmentation of full-length corporate reports into semantically coherent passages (up to 512 tokens) and filters for SDG engagement across all 17 UN Sustainable Development Goals using expert-compiled multilingual regex patterns (German & English).

---

## 📌 Architectural Overview

```
data/texts/<Company>/<Year>/results.txt
               │
               ▼
     split_passages.py  ──> NLTK sentence splitting + Tiktoken (512-token packing)
               │
               ▼
data/jsons/<Company>/<Year>/splits_semantic.json
               │
               ▼
   sdg_keyword_match.py ──> Regex matchers (kw_data/keywords_sdg[_de].json)
               │
               ▼
data/dbs/sdg_hits.duckdb (or CSV/Parquet export)
```

---

## 🚀 Key Scripts & Assets

| Script / Asset | Purpose | Inputs | Outputs |
|---|---|---|---|
| [`split_passages.py`](split_passages.py) | Sentence-aware greedy token chunker utilizing `tiktoken` (`cl100k_base`). Avoids splitting mid-sentence. | `data/texts/**/*.txt` | `data/jsons/**/splits_semantic.json` |
| [`sdg_keyword_match.py`](sdg_keyword_match.py) | High-speed regex scanner evaluating 17 SDG categories in German and English, recording match positions and frequencies. | `splits_semantic.json`, `kw_data/*.json` | `data/dbs/sdg_hits.duckdb` |
| `kw_data/keywords_sdg.json` | Curated English keyword patterns compiled from academic literature and UN metadata. | Static Config | Regex Patterns |
| `kw_data/keywords_sdg_de.json` | Curated German keyword patterns (e.g., *Treibhausgasemissionen*, *Kreislaufwirtschaft*, *Menschenrechte*). | Static Config | Regex Patterns |

---

## 🛠️ Step-by-Step Execution Guide

### 1. Split Text into Semantic Passages
```bash
# Tokenize and chunk all extracted company texts into 512-token passages
python 02_passage_extraction/split_passages.py --workers 8

# Force re-chunking
python 02_passage_extraction/split_passages.py --workers 8 --force
```
* **Algorithm:** Iterates through paragraphs, applies NLTK sentence tokenization, and greedily aggregates sentences up to a maximum limit of 512 tokens.

### 2. Match Passages Against SDG Keywords
```bash
# Scan all passages and store matched hits in DuckDB
python 02_passage_extraction/sdg_keyword_match.py

# Force overwrite of existing database
python 02_passage_extraction/sdg_keyword_match.py --force
```
* Identifies language (via `langdetect`), selects the corresponding English or German regex vocabulary, and records the exact matched keyword, SDG goal (SDG 1 to SDG 17), and snippet context.

---

## 📋 Output Schema (`sdg_hits.duckdb`)

```sql
CREATE TABLE sdg_hits (
    passage_id    VARCHAR PRIMARY KEY,
    company_name  VARCHAR,
    year          INTEGER,
    sdg_category  VARCHAR,  -- e.g., 'sdg13'
    keyword       VARCHAR,  -- e.g., 'carbon neutrality'
    passage_text  VARCHAR,
    token_count   INTEGER,
    match_count   INTEGER
);
```
