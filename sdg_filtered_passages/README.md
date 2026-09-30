# 📊 SDG-Filtered Corporate Report Passages

This directory contains the complete corpus of sustainability and annual report passages filtered by UN Sustainable Development Goal (SDG) keywords across German listed corporations (DAX, MDAX, SDAX) from fiscal years **2014 through 2024**.

---

## 📁 Directory Contents

| File | Format | Description |
|---|:---:|---|
| **`all_sdg_filtered_passages.csv`** | CSV (UTF-8 BOM) | **Master Dataset:** All ~30,434 corporate passages that matched one or more of the 17 UN SDGs. |
| **`sample_1000_sdg_passages.xlsx`** | Excel (`.xlsx`) | **Supervisor Review Sheet:** A stratified sample of 1,000 representative passages optimized for opening directly in Microsoft Excel without lag. |
| **`summary_by_sdg_category.csv`** | CSV | Frequency of passages matching each goal from **SDG 1** (*No Poverty*) to **SDG 17** (*Partnerships*). |
| **`summary_by_company.csv`** | CSV | Breakdown of SDG-relevant passages by corporate reporting entity. |
| **`pipeline_sdg_filter.py`** | Python | Minimalist, self-contained pipeline script to re-export or update these files. |

---

## 🔍 Extraction & Filtering Methodology

1. **Document Ingestion:** Annual and sustainability PDFs from 151 German companies were ingested and text-extracted using PyMuPDF.
2. **Semantic Passage Chunking:** Raw text was split into coherent passages using NLTK sentence tokenization packed up to **512 tokens** (`cl100k_base`), preventing arbitrary mid-sentence cuts.
3. **Multilingual Regex Keyword Matching:** Each passage was evaluated against **3,128 English** and **3,090 German** curated regex patterns covering all 17 SDGs (e.g., *decarbonization*, *Erneuerbare Energien*, *Lieferkettensorgfaltspflicht*, *anti-corruption*).
4. **Output Generation:** Passages containing valid SDG pattern hits were recorded and indexed.

---

## 📋 Data Dictionary (`all_sdg_filtered_passages.csv`)

| Column Name | Type | Description | Example |
|---|---|---|---|
| `global_id` | String | Unique passage reference ID | `2022AdidasAG142` |
| `company` | String | Reporting firm name | `Adidas_AG` |
| `year` | Integer | Fiscal reporting year | `2022` |
| `language` | String | Detected language code (`en` or `de`) | `en` |
| `sdg_categories` | String | Semicolon-delimited list of matched SDGs | `SDG 12; SDG 13` |
| `sdg_count` | Integer | Total distinct SDGs addressed in the excerpt | `2` |
| `passage` | String | Full 512-token disclosure excerpt | `"We defined a clear roadmap to achieve our emission reduction targets..."` |

---

## ⚡ How to Run or Re-Export

To regenerate the files at any time using your Conda environment:

```bash
python sdg_filtered_passages/pipeline_sdg_filter.py
```
