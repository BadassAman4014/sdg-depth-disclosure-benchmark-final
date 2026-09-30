# 📥 Task 01: Corporate Report Collection & Text Ingestion

This module manages the automated discovery, downloading, verification, and text extraction of corporate annual and sustainability reports for German listed firms across the DAX, MDAX, and SDAX indices (covering fiscal years 2014 through 2024).

---

## 📌 Architectural Overview

```
GRI score.xlsx / companies.csv
          │
          ├──> extract_companies.py ──> Generates data/companies.json & data/companies.csv
          │
          ├──> download_reports.py  ──> Multi-Engine Query Search (DDG / Bing / Yahoo)
          │                                  │
          │                                  └──> downloads: reports/<Company>/<Year>/*.pdf
          │
          ├──> audit_reports.py     ──> Verifies download integrity & builds coverage_matrix.csv
          │
          └──> extract_text.py      ──> Multiprocess PyMuPDF extraction
                                             │
                                             └──> outputs: data/texts/<Company>/<Year>/results.txt
```

---

## 🚀 Key Scripts & Functions

| Script | Purpose | Key Inputs | Primary Outputs |
|---|---|---|---|
| [`extract_companies.py`](download_reports.py) | Parses German company names, ISINs, and tickers into standardized filesystem slugs. | `GRI score.xlsx` | `data/companies.json`, `data/companies.csv` |
| [`download_reports.py`](download_reports.py) | Automated multi-engine PDF scraper with fallback rotation (DuckDuckGo, Bing, Yahoo) and exponential backoff. | `data/companies.json` | `reports/<Company>/<Year>/report.pdf`, `reports_manifest.csv` |
| [`audit_reports.py`](audit_reports.py) | Generates a coverage matrix tracking report acquisition across 10+ fiscal years. | `reports/reports_manifest.csv` | `reports/coverage_matrix.csv` |
| [`extract_text.py`](extract_text.py) | High-speed multi-core PDF text extractor utilizing PyMuPDF with layout cleaning and length validation. | `reports/**/*.pdf` | `data/texts/<Company>/<Year>/results.txt` |

---

## 🛠️ Step-by-Step Execution Guide

### 1. Extract and Verify Target Companies
```bash
python 01_data_collection/extract_companies.py
```
* Generates sanitized folder names, stripping illegal filesystem characters and standardizing legal forms (e.g., `AG`, `SE`, `GmbH`).

### 2. Run Multi-Engine PDF Downloader
```bash
# Run downloading across all companies for years 2014-2024
python 01_data_collection/download_reports.py

# Optional: Run with specific batch size or debug logging
python 01_data_collection/download_reports.py --limit 10
```
* **Anti-Blocking Strategy:** The crawler rotates between DuckDuckGo HTML, Bing search, and Yahoo search APIs, enforcing custom User-Agent headers, polite request intervals, and document MIME-type checks before saving.

### 3. Audit Download Coverage
```bash
python 01_data_collection/audit_reports.py
```
* Produces a terminal summary and exports `coverage_matrix.csv` displaying which fiscal years have been successfully acquired for every firm.

### 4. Extract Clean Text from PDFs
```bash
# Parallel extraction using 8 CPU cores
python 01_data_collection/extract_text.py --workers 8

# Force re-extraction of previously processed files
python 01_data_collection/extract_text.py --workers 8 --force
```
* Skips corrupted files or empty scans (threshold: `MIN_TEXT_LEN = 500` characters).
* Preserves sentence boundaries and paragraphs for downstream sliding-window chunking.

---

## 📋 Data Formats

### `reports_manifest.csv`
```csv
company_id,company_name,year,pdf_url,file_size_mb,download_time,status,file_path
1,Adidas_AG,2022,https://...,14.25,2026-09-01 10:15:20,SUCCESS,reports/Adidas_AG/2022/report.pdf
```
