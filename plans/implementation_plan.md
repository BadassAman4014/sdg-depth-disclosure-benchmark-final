# Extract SDG Paragraphs & Build Ground Truth Excel

## Background

The user has **1,405 annual report PDFs** (2014–2024) for 151 German companies already downloaded to `reports/<Company>/<Year>/<Company>_Annual_Report_<Year>.pdf`. The goal is to:

1. Extract text from these PDFs
2. Split text into semantic passages
3. Match passages against SDG keyword regex patterns (from the AI-For-Sustainability repo)
4. Output matched passages + keywords into a **ground truth Excel file** for human labeling

The [AI-For-Sustainability](file:///c:/Users/amanr/Documents/GitHub/Reliability%20test%20German/AI-For-Sustainability) repo has a complete pipeline for this on the `v1-full-pipeline` branch: PDF → text → semantic splits → DuckDB with regex hits. However, the current `main` branch starts from pre-computed DuckDB files and focuses on GPT classification.

We need to **adapt the v1 preprocessing pipeline** to work with our downloaded reports, running in the `escp` conda environment.

## User Review Required

> [!IMPORTANT]
> The existing ground truth example ([Updated Ground Truths - 100 samples for evaluating models.xlsx](file:///c:/Users/amanr/Documents/GitHub/Reliability%20test%20German/Updated%20Ground%20Truths%20-%20100%20samples%20for%20evaluating%20models.xlsx)) has 3 columns: `passage`, `keyword`, `ground_truth` (values: `sym`/`sub`). The new ground truth file will follow this same schema, with `ground_truth` left blank for human labeling.

> [!WARNING]
> The v1 pipeline uses `PyMuPDF (fitz)` for PDF text extraction and `langchain-experimental` + `sentence-transformers` for semantic chunking. These are heavy dependencies that need to be installed in the `escp` environment. The semantic chunking step uses an embedding model (`paraphrase-multilingual-MiniLM-L12-v2`) which will be downloaded on first run (~120MB).

## Open Questions

> [!IMPORTANT]
> **Q1: Processing scope** — Do you want ALL 1,405 reports processed, or start with a smaller subset (e.g., a few companies) to validate the pipeline first?

> [!IMPORTANT]
> **Q2: Sampling strategy for ground truth** — The ground truth Excel should contain a representative sample for humans to label. Options:
> - **Random sample** (e.g., 500–1000 passages randomly sampled from all matched passages)
> - **Stratified sample** (ensuring coverage across companies, years, and SDG categories)
> - **Full export** (ALL matched passages — could be 100k+ rows; likely too many for manual labeling)

> [!IMPORTANT]
> **Q3: Language filter** — The reports are for German companies but may contain English text. Should we:
> - Match both English AND German SDG keywords (the repo has both `keywords_sdg.json` and `keywords_sdg_de.json`)
> - Match English keywords only?

## Proposed Changes

### Phase 1 — Environment Setup

Install all required dependencies in the `escp` conda environment:

```
PyMuPDF (fitz), langchain-experimental, langchain-huggingface,
sentence-transformers, tiktoken, nltk, duckdb, tqdm, langdetect
```

Also extract the v1 keyword files (`kw_data/keywords_sdg.json`, `kw_data/keywords_sdg_de.json`) from the `v1-full-pipeline` branch.

---

### Phase 2 — PDF Text Extraction

#### [NEW] `scripts/extract_text.py`

- Reads all PDFs from `reports/<Company>/<Year>/*.pdf`
- Uses PyMuPDF (`fitz`) to extract raw text (same as v1's [pdf2text.py](file:///c:/Users/amanr/Documents/GitHub/Reliability%20test%20German/AI-For-Sustainability/src/preprocessing/pdf2text.py))
- Saves plain text files to `data/texts/<Company>/<Year>/results.txt`
- Multi-threaded for speed
- Tracks progress and skips already-extracted files

---

### Phase 3 — Semantic Text Splitting

#### [NEW] `scripts/split_passages.py`

- Reads extracted text files from `data/texts/<Company>/<Year>/results.txt`
- Splits each into semantic passages using the same approach as v1's [splitter.py](file:///c:/Users/amanr/Documents/GitHub/Reliability%20test%20German/AI-For-Sustainability/src/preprocessing/splitter.py):
  - Primary: `SemanticChunker` from langchain-experimental with `paraphrase-multilingual-MiniLM-L12-v2`
  - Passages capped at 512 tokens
  - Min chunk size: 10 tokens
- Outputs JSON files to `data/jsons/<Company>/<Year>/splits_semantic.json` (dict of `{sentence_id: passage_text}`)

---

### Phase 4 — SDG Keyword Matching & DuckDB

#### [NEW] `scripts/sdg_keyword_match.py`

- Adapts the v1 [sdg_filter.py](file:///c:/Users/amanr/Documents/GitHub/Reliability%20test%20German/AI-For-Sustainability/src/filtering/sdg_filter.py) + [base_filter.py](file:///c:/Users/amanr/Documents/GitHub/Reliability%20test%20German/AI-For-Sustainability/src/filtering/base_filter.py)
- Scans all `data/jsons/<Company>/<Year>/splits_semantic.json` for SDG keyword regex matches
- Uses both English and German keyword dictionaries
- Writes matched passages to `data/dbs/sdg_hits.duckdb` (same schema as the repo expects)
- Schema: `global_id, passage, company, year, language, hits_sdg1..hits_sdg17`

---

### Phase 5 — Ground Truth Excel Generation

#### [NEW] `scripts/generate_ground_truth.py`

- Reads from `data/dbs/sdg_hits.duckdb`
- For each passage with hits: extracts each (passage, keyword_pattern) pair
- Applies a sampling strategy (stratified by company/year/SDG category)
- Outputs to `data/ground_truth.xlsx` with columns:
  - `passage` — the text passage
  - `keyword` — the matched regex pattern
  - `ground_truth` — left **blank** for human labeling
  - Additional metadata columns: `global_id`, `company`, `year`, `language`, `sdg_category`
- Also generates a companion `data/ground_truth_full.xlsx` with ALL matched passages (not just the sample)

---

### Phase 6 — Pipeline Orchestrator

#### [NEW] `scripts/run_sdg_pipeline.py`

A master script that chains all steps in order:
1. Extract text from PDFs
2. Split text into passages
3. Run SDG keyword matching
4. Generate ground truth Excel

Can be run end-to-end or step-by-step via CLI flags.

## Verification Plan

### Automated Tests
- Verify PDF text extraction produces valid text files for a sample of reports
- Verify passage splitter produces properly sized passages (10–512 tokens)
- Verify DuckDB schema matches the AI-For-Sustainability repo's expected format
- Verify ground truth Excel matches the example format (passage, keyword, ground_truth columns)

### Manual Verification
- Spot-check extracted passages for quality
- Compare a few entries with the existing 100-sample ground truth to ensure consistency
- User reviews the output ground truth Excel before proceeding with human labeling
