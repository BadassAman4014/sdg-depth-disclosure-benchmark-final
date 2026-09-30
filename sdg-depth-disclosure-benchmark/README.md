# 🌍 Corporate SDG Disclosure Depth Scoring & Machine Learning Benchmarks
### Empirical Study of 151 German Listed Companies (DAX, MDAX, SDAX) Across Fiscal Years 2014–2024

[![Python 3.11](https://img.shields.io/badge/python-3.11.15-blue.svg)](https://www.python.org/downloads/)
[![Git LFS](https://img.shields.io/badge/Git%20LFS-Enabled-informational.svg)](https://git-lfs.github.com/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn%20%7C%20PyTorch%20%7C%20Transformers-orange.svg)](https://scikit-learn.org/)
[![Database](https://img.shields.io/badge/Database-DuckDB-yellow.svg)](https://duckdb.org/)
[![Academic Reference](https://img.shields.io/badge/Depth%20Rubric-Izhar%20et%20al.%20(2026)-success.svg)](https://doi.org/10.1016/j.jclepro)
[![Benchmark Paper](https://img.shields.io/badge/ML%20Benchmark-Shah%20et%20al.%20(2020)-blueviolet.svg)](https://doi.org/10.1007/s42979-020-0012-3)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## 📑 Table of Contents
1. [Executive Summary & Research Motivation](#1-executive-summary--research-motivation)
2. [Academic Foundations & Theoretical Framework](#2-academic-foundations--theoretical-framework)
3. [Full Corpus Ingestion & Data Pipeline Funnel](#3-full-corpus-ingestion--data-pipeline-funnel)
4. [Data Assets & Dedicated Supervisor Folder (`sdg_filtered_passages/`)](#4-data-assets--dedicated-supervisor-folder-sdg_filtered_passages)
5. [Human Ground Truth & Inter-Coder Reliability Protocol (N = 50)](#5-human-ground-truth--inter-coder-reliability-protocol-n--50)
6. [Continuous 0–5 Depth Scoring Engine (*Izhar et al., 2026*)](#6-continuous-05-depth-scoring-engine-izhar-et-al-2026)
7. [Machine Learning Benchmarks & Hyperparameter Tuning (*Shah et al., 2020*)](#7-machine-learning-benchmarks--hyperparameter-tuning-shah-et-al-2020)
8. [Repository Architecture & Modular File Map](#8-repository-architecture--modular-file-map)
9. [Environment Setup & Reproduction Manual](#9-environment-setup--reproduction-manual)
10. [Git LFS Configuration & GitHub Push Walkthrough](#10-git-lfs-configuration--github-push-walkthrough)
11. [Technical Gotchas & Robust Engineering Solutions](#11-technical-gotchas--robust-engineering-solutions)
12. [Academic Citations & BibTeX Reference](#12-academic-citations--bibtex-reference)

---

## 1. Executive Summary & Research Motivation

### The Corporate Sustainability Dilemma: Symbolic vs. Substantive Disclosure
Corporate non-financial reporting under the United Nations Sustainable Development Goals (SDGs) has grown exponentially. However, empirical accounting literature highlights a profound methodological challenge: **the Greenwashing / Symbolic Disclosure Gap**.
1. **The Greenwashing Trap:** Corporations routinely fill glossy annual and sustainability reports with high-level pledges, mission statements, and aspirational imagery without committing capital, executing operational actions, or subjecting progress to external audit verification.
2. **Failure of Naive Keyword Counting:** Traditional sustainability research employs dictionary-based keyword counters (e.g., counting mentions of *"climate"*, *"water"*, or *"diversity"*). This method fails fundamentally because it assigns identical weight to an empty marketing aspiration (*"we strive to protect the climate"* $\rightarrow$ Level 1) and an audited, capital-intensive operational achievement (*"reduced Scope 1 GHG emissions by 24.5% verified under limited assurance by PwC"* $\rightarrow$ Level 5).
3. **The Coarse Binary Discard:** Treating corporate disclosures as purely binary (*"mentions SDG"* vs. *"does not mention SDG"*) discards the rich, multi-tiered spectrum of corporate accountability.

### The Solution Developed in this Repository
This repository provides an enterprise-scale, scientifically validated computational framework that:
* **Ingests and processes 1,401 full-length corporate reports** (2014–2024) across **151 German listed corporations** from the DAX 40, MDAX 50, and SDAX 70.
* **Segments reports into 453,404 semantically coherent 512-token passages** using sentence-boundary preservation.
* **Filters 30,434 unique SDG passages** using 6,218 expert-compiled multilingual regex patterns (German and English) across all 17 UN SDGs.
* **Establishes a human ground truth baseline** with an independent dual-coder protocol achieving **98.0% agreement** and a Quadratic Weighted Cohen's Kappa of **$\kappa_w = 0.9896$**.
* **Implements the first automated continuous ($0.00$ to $5.00$) and ordinal ($0$ to $5$) depth scoring engine** grounded in *Izhar et al. (2026)* Table 1, achieving **$r = 0.9584$** correlation with human consensus and **100% within $\pm 1$ level agreement**.
* **Benchmarks and fine-tunes classical and neural machine learning architectures** grounded in *Shah et al. (2020)*, where fine-tuned Random Forest achieved **86.0% accuracy** and an **83.0% Macro F1-score**.

### Critical Distinction from `kushal-10/AI-For-Sustainability`
* **`kushal-10/AI-For-Sustainability`:** Focused exclusively on API prompt engineering experiments (Zero-Shot, Few-Shot, Chain-of-Thought, Tree-of-Thought with GPT-4o via OpenAI Batch API).
* **This Repository:** Implements the complete end-to-end data acquisition and machine learning pipeline that makes systematic research possible: multi-engine report scraping, OCR/text extraction, 512-token sliding chunking, DuckDB indexing, dual-coder blinding/adjudication, continuous depth regression, classical ML baselines, and reproducible Git LFS datasets.

---

## 2. Academic Foundations & Theoretical Framework

The computational architecture is directly grounded in five foundational peer-reviewed studies:

### 1. Depth Scoring Rubric: *Izhar et al. (2026)*
* **Citation:** Izhar, M., et al. (2026). *Exploring firm-level SDG engagement in an emerging economy through content analysis.* Journal of Cleaner Production.
* **Theoretical Construct:** Table 1 and Section 3.4.3 define disclosure depth through the intersection of three operational dimensions: **Targets**, **Actions**, and **Outcomes**.

#### The Table 1 Scoring Matrix
| Score Level | Theoretical Description | Qualitative Target | Quantitative Target | Implemented Actions | Qualitative Outcome | Quantitative / Audited Outcome | Construct Class |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **0** | **No SDG Information / Boilerplate** | ❌ | ❌ | ❌ | ❌ | ❌ | **Symbolic (`sym`)** |
| **1** | **Qualitative Target Only (Aspiration)** | ✔️ | ❌ | ❌ | ❌ | ❌ | **Symbolic (`sym`)** |
| **2** | **Qual Target + Actions** OR **Quant Target** | ✔️ *(or ❌)* | *(or ✔️)* ❌ | ✔️ *(or ❌)* | ❌ | ❌ | **Substantive (`sub`)** |
| **3** | **Qual Target + Qualitative Outcome** | ✔️ | ❌ | *(often ✔️)* | ✔️ | ❌ | **Substantive (`sub`)** |
| **4** | **Qual Target + Quantitative Outcome** | ✔️ | ❌ | *(often ✔️)* | ❌ | ✔️ | **Substantive (`sub`)** |
| **5** | **Quant Target + Measured / Audited Outcome** | ❌ | ✔️ | *(often ✔️)* | ✔️ | ✔️ | **Substantive (`sub`)** |

* **Binary Distinction:** Levels 0–1 represent **Symbolic (`sym`)** reporting (cheap talk, aspirational claims), while Levels 2–5 represent **Substantive (`sub`)** reporting (verifiable actions, capital allocation, measurable outputs).

### 2. Multi-Model Text Classification Benchmark: *Shah et al. (2020)*
* **Citation:** Shah, K., Patel, H., Sanghvi, D., & Shah, M. (2020). *A Comparative Analysis of Logistic Regression, Random Forest and KNN Models for Text Classification.* Augmented Human Research, 5(1), 12.
* **Empirical Focus:** Systematically contrasts linear classifiers (Logistic Regression), ensemble decision tree bagging (Random Forest), and instance-based learners (KNN) in high-dimensional text vector spaces. We expand this benchmark by incorporating multilingual dense sentence embeddings (*MiniLM-L12-v2*) and Glassbox Explainable Boosting Machines (EBM).

### 3. Complementary Standards in Accounting & NLP
* **Hummel, K. & Schlick, C. (2016) / PwC (2018):** Demonstrates that third-party independent assurance (Big 4 accounting firms, TÜV, ISO certifications) is the primary empirical indicator separating credible non-financial disclosure from greenwashing.
* **Landis, J. R., & Koch, G. G. (1977):** Establishes the statistical threshold for inter-annotator reliability: $\kappa > 0.80$ denotes *Almost Perfect Agreement*.
* **Krippendorff, K. (2004):** *Content Analysis: An Introduction to Its Methodology.* Defines the gold-standard protocol for dual-coder blinding, dispute adjudication, and reproducible annotation.

---

## 3. Full Corpus Ingestion & Data Pipeline Funnel

### 1. Corporate Universe (151 Listed German Equities)
The study analyzes the full index composition of the German stock exchange across three market tiers:
* **DAX 40 (Blue Chips):** *adidas, Airbus, Allianz, BASF, Bayer, Beiersdorf, BMW, Brenntag, Continental, Covestro, Daimler Truck, Deutsche Bank, Deutsche Börse, Deutsche Post (DHL), Deutsche Telekom, E.ON, Fresenius, Fresenius Medical Care, Hannover Rück, Heidelberg Materials, Henkel, Infineon, Mercedes-Benz Group, Merck KGaA, MTU Aero Engines, Munich Re, Porsche AG, Porsche SE, Qiagen, RWE, SAP, Sartorius, Siemens AG, Siemens Energy, Siemens Healthineers, Symrise, Volkswagen, Vonovia, Zalando.*
* **MDAX 50 (Mid-Cap Leaders):** *Aixtron, Aurubis, Bechtle, Bilfinger, Carl Zeiss Meditec, Commerzbank, CTS Eventim, Delivery Hero, Dürr, Evonik Industries, Freenet, GEA Group, Gerresheimer, Hapag-Lloyd, Heidelberger Druckmaschinen, HelloFresh, Hochtief, Hugo Boss, Jenoptik, K+S, KION Group, Krones, Lanxess, LEG Immobilien, Nemetschek, Nordex, Puma, Rational, Rheinmetall, SAF-Holland, Salzgitter, Scout24, Siltronic, SMA Solar, Software AG, Stabilus, Ströer, TAG Immobilien, TeamViewer, Telefonica Deutschland, Thyssenkrupp, TUI, United Internet, Vossloh, Wacker Chemie.*
* **SDAX 70 (Small-Cap Growth):** *1&1 AG, Adesso, Amadeus Fire, Atoss Software, AUTO1 Group, Basler, BayWa, Boruss. Dortmund, Cancom, Ceconomy, Cewe Stiftung, Cherry, CompuGroup Medical, CropEnergies, Dermapharm, Deutz, Douglas, Dr. Hönle, Eckert & Ziegler, ElringKlinger, Energiekontor, Fielmann, GFT Technologies, Hamborner REIT, HENSOLDT, Hornbach Holding, Hypoport, Indus Holding, Jost Werke, Klöckner & Co, Koenig & Bauer, Kontron, KWS Saat, MorphoSys, Mutares, Nagarro, Norma Group, PATRIZIA, PNE, ProSiebenSat.1, PVA TePla, SGL Carbon, Sixt, Sto, STRATEC, Süss MicroTec, Synlab, Takkt, Verbio, Vitesco Technologies, Wacker Neuson.*

### 2. The Data Ingestion Funnel
```
========================================================================================
1. RAW ANNUAL & SUSTAINABILITY REPORTS
   1,654 PDF Files Downloaded (~20+ GB on disk)
   ├── Tracked in: reports/reports_manifest.csv
   ├── Years: 2014 to 2024 (11 Fiscal Years)
   └── Resilient multi-engine scraper with anti-bot rotation (DuckDuckGo, Bing, Yahoo)
========================================================================================
                                      │
                                      ▼
========================================================================================
2. TEXT EXTRACTION & NORMALIZATION
   1,401 Reports Successfully Parsed
   ├── Multiprocess PyMuPDF text stream extraction (data/extract_text.log: OK=1401)
   └── Validated minimum length threshold (MIN_TEXT_LEN = 500 chars)
========================================================================================
                                      │
                                      ▼
========================================================================================
3. PASSAGE CHUNKING & BOUNDARY PRESERVATION
   453,404 Raw Text Chunks Produced
   ├── NLTK sentence boundary tokenization + Tiktoken cl100k_base
   ├── Greedy packing up to 512 tokens with 50-token sliding context overlap
   └── Logged in: data/split_passages.log (TotalPassages=453404)
========================================================================================
                                      │
                                      ▼
========================================================================================
4. MULTILINGUAL SDG REGEX PATTERN MATCHING
   30,434 Unique SDG Passages Isolated (~74 MB)
   ├── Scanned against 3,128 English + 3,090 German expert regex patterns
   ├── Evaluates all 17 UN Sustainable Development Goals
   ├── Pre-indexed in DuckDB high-performance analytics database (sdg_hits.duckdb)
   └── 1,242,827 Keyword Match Pairs Exploded (~158 MB)
========================================================================================
```

### 3. Deep-Dive: Why 30,434 Passages from 453,404 Chunks?
Corporate annual reports average 250 to 450 pages each. A thorough examination of corporate reporting structure reveals:
* **70%–90% of total report volume** is comprised of mandatory statutory financial statements: IFRS/HGB balance sheets, income statements, statements of comprehensive income, cash flow notes, segment accounting, pension liability disclosures, board remuneration tables, and auditor sign-off forms. These sections do not discuss sustainability initiatives.
* **Passage hit rate:** Across the 453,404 total chunks, exactly **30,434 chunks (~6.71%)** contained verified SDG keywords.
* **Keyword explosion:** Because corporate sustainability narratives frequently address interconnected targets simultaneously (e.g., an energy-efficiency initiative matching keywords for SDG 7 *Clean Energy*, SDG 9 *Industry & Innovation*, and SDG 13 *Climate Action*), exploding passages by their individual matched keywords yields **1,242,827 keyword match pairs**.

---

## 4. Data Assets & Dedicated Supervisor Folder (`sdg_filtered_passages/`)

For executive review, thesis committees, and supervisory oversight, all filtered data assets are centralized in the `sdg_filtered_passages/` directory:

```
sdg_filtered_passages/
├── README.md                            <-- Data dictionary and overview for supervisors
├── pipeline_sdg_filter.py               <-- Concise, standalone DuckDB extraction pipeline
├── all_sdg_filtered_passages.csv        <-- 30,434 unique passages (~74 MB, UTF-8)
├── ground_truth_full.csv                <-- 1,242,827 keyword-matched rows (~158 MB, Git LFS)
├── ground_truth_full.xlsx               <-- Complete formatted Excel workbook (~28.4 MB)
├── golden_dataset_1000.xlsx             <-- 1,000 stratified samples for instant Excel inspection
├── golden_dataset_500_continuous_0_to_5.xlsx <-- 500 samples scored on the continuous scale
└── summary_by_sdg_category.csv          <-- Passage and keyword volume per SDG (1 to 17)
```

### Comprehensive Data Dictionary
| Column Name | Data Type | Description | Example Content |
|---|---|---|---|
| `passage_id` | Integer | Unique identifier across the 453,404 chunk corpus | `210492` |
| `company` | String | Standardized corporate entity name | `Siemens AG` |
| `index` | String | German equity market index classification | `DAX 40` |
| `fiscal_year` | Integer | Fiscal reporting year (2014 to 2024) | `2023` |
| `report_type` | String | Document category | `sustainability_report` |
| `sdg_number` | Integer | Associated UN Sustainable Development Goal | `13` |
| `keyword_matched`| String | Specific regex keyword triggering match | `emissions reduction` |
| `passage_text` | String | Full 512-token narrative chunk text | *"By 2030, we target a 90% reduction..."* |
| `word_count` | Integer | Total word count in passage chunk | `384` |
| `char_count` | Integer | Total character length of chunk | `2491` |

### SDG Distribution Breakdown
Passage density is naturally concentrated in goals directly relevant to industrial manufacturing, energy, and corporate governance:
* **Top Disclosed Goals:** SDG 13 (*Climate Action*, ~24.1%), SDG 8 (*Decent Work & Economic Growth*, ~18.5%), SDG 12 (*Responsible Consumption & Production*, ~15.2%), SDG 9 (*Industry, Innovation & Infrastructure*, ~12.8%), SDG 7 (*Affordable & Clean Energy*, ~9.4%).
* **Lower Frequency Goals:** SDG 1 (*No Poverty*, ~1.1%), SDG 2 (*Zero Hunger*, ~0.8%), SDG 14 (*Life Below Water*, ~1.4%).

---

## 5. Human Ground Truth & Inter-Coder Reliability Protocol (N = 50)

To validate the machine learning architecture and establish a defensible ground truth baseline, a rigorous scientific annotation protocol was executed in accordance with *Krippendorff (2004)* and *Landis & Koch (1977)*:

```
                        Stratified Candidate Passages (N = 50)
                                      │
            ┌─────────────────────────┴─────────────────────────┐
            ▼                                                   ▼
   Coder 1 (Person A)                                  Coder 2 (Person B)
   Blinded Annotation Sheet                           Blinded Annotation Sheet
   (Forest Green Theme)                                (Burgundy Wine Theme)
            │                                                   │
            └─────────────────────────┬─────────────────────────┘
                                      ▼
                        Inter-Coder Agreement Engine
                       (analyze_coder_agreement.py)
                                      │
                         ┌────────────┴────────────┐
                         ▼                         ▼
               49 Identical Scores       1 Divergent Score (ID 5)
                (98.0% Agreement)            Adjudication Review
                         │                         │
                         └────────────┬────────────┘
                                      ▼
                           Consensus Ground Truth
                          (reconcile_coders.py)
```

### Statistical Agreement Metrics
* **Exact Percentage Agreement:** **`98.0%`** (49 out of 50 passages assigned the exact same integer level).
* **Within $\pm 1$ Level Agreement:** **`100.0%`** (50 out of 50 passages within 1 score point).
* **Unweighted Cohen's Kappa ($\kappa$):** **`0.9686`** (Far exceeding the 0.80 benchmark for near-perfect agreement).
* **Linear Weighted Kappa ($\kappa_{lin}$):** **`0.9805`**.
* **Quadratic Weighted Kappa ($\kappa_{quad}$):** **`0.9896`** (The academic standard for ordered categorical scales).
* **Pearson Correlation ($r$):** **`0.9943`** ($p = 1.14 \times 10^{-47}$).
* **Spearman Rank Correlation ($\rho$):** **`0.9940`**.

### Adjudication of the Single Divergent Sample
* **Sample ID 5 (1&1 AG):** 
  * *Person A Assigned:* **Score 2** (focusing on concrete HR training actions implemented under `#MYBEST`).
  * *Person B Assigned:* **Score 1** (viewing `#MYBEST` as an internal aspirational policy framework).
  * *Consensus Resolution:* Adjudicated as **Score 2** because the text specifically reported implemented training programs, workshops, and quality management modules active in fiscal year 2024.

---

## 6. Continuous 0–5 Depth Scoring Engine (*Izhar et al., 2026*)

### 1. Hybrid Engineering Architecture
To model the complex semantics of non-financial accounting text, the scoring engine combines **dense neural representations** with **theory-guided linguistic features**:

```
Input Passage (German or English)
   │
   ├──> 1. Dense Semantic Encoder: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
   │       └── Yields: 384-dimensional dense context vector
   │
   └──> 2. Theory-Guided Linguistic Feature Extractor (Izhar et al. 2026 Table 1 Criteria)
           ├── has_audit: Big 4 (PwC, KPMG, EY, Deloitte), TÜV, ISO 14001, ISO 50001
           ├── metric_count: %, tCO2e, tonnes, MWh, GWh, million EUR, kWh
           ├── action_count: implemented, constructed, installed, commissioned, trained
           ├── asp_count: aim to, strive, aspire, commit, pledge, vision
           ├── qual_outcome: improved, enhanced, strengthened, expanded
           └── evidence_density: (2.0*metrics + 1.5*actions + 3.0*audits) / (words + 1)
                   │
                   ▼
   Stacked Hybrid Representation Vector [384 + 6 = 390 dimensions]
                   │
                   ▼
   Calibrated Penalized Ridge Regressor (L2 alpha = 1.5)
                   │
                   ▼
   Continuous Depth Score [0.00 to 5.00] ──> Binned to Table 1 Level [0, 1, 2, 3, 4, 5]
```

### 2. Empirical Validation Against Human Ground Truth (N = 50)
* **Pearson Correlation ($r$):** **`0.9584`** ($p = 8.77 \times 10^{-28}$) — Extremely high linear tracking.
* **Spearman Rank Correlation ($\rho$):** **`0.9051`** — Monotonic ordering fidelity.
* **Root Mean Squared Error (RMSE):** **`±0.453`** on a 6-point scale.
* **Mean Absolute Error (MAE):** **`±0.339`** points (average deviation ~1/3 of a score level).
* **Exact Integer Match Accuracy:** **`80.0%`** (40 / 50 exact matches).
* **Within $\pm 1$ Level Agreement:** **`100.0%`** (50 / 50 passages within 1 level).
* **Quadratic Weighted Cohen's Kappa ($\kappa_w$):** **`0.9072`** (Almost Perfect Agreement).
* **Binary Accuracy (`sym` vs. `sub`):** **`88.0%`** ($\kappa = 0.6907$).

### 3. Confusion Matrix
```
                Model 0   Model 1   Model 2   Model 3   Model 4   Model 5
Human Level 0     25         1         0         0         0         0
Human Level 1      0         8         1         0         0         0
Human Level 2      0         5         7         0         0         0
Human Level 3      0         0         0         0         0         0
Human Level 4      0         0         0         0         0         0
Human Level 5      0         0         0         0         3         0
```
*Key Diagnostic:* 100% of errors occur on adjacent off-diagonals (e.g., Level 2 vs Level 1). There are zero extreme false positives (Level 0 vs Level 5).

---

## 7. Machine Learning Benchmarks & Hyperparameter Tuning (*Shah et al., 2020*)

To establish academic baselines, four classifier paradigms were benchmarked using **5-Fold Stratified Cross-Validation** on human ground truth:

### Comprehensive Model Benchmark Table
| Model Architecture | Hyperparameter Configuration | Accuracy | Precision (`sub`) | Recall (`sub`) | F1-Score (`sub`) | Macro F1 | Type I (FP) Greenwash | Type II (FN) Missed Act |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1. Logistic Regression** (Default) | $C=1.0$, $L_2$, lbfgs | 74.0% | 58.33% | 46.67% | 51.85% | 67.02% | 14.29% | 53.33% |
| **1. Logistic Regression** *(Fine-Tuned)* | $C=5.0$, $L_1$, liblinear, balanced | **80.0%** | **69.23%** | **60.00%** | **64.29%** | **75.20%** | 11.43% | 40.00% |
| **2. Random Forest** (Default) | $n=100$, depth=6 | 80.0% | 69.23% | 60.00% | 64.29% | 75.20% | 11.43% | 40.00% |
| **2. Random Forest** *(Fine-Tuned [BEST])* | $n=200$, depth=4, feat=0.3, balanced | **86.0%** | **78.57%** | **73.33%** | **75.86%** | **83.00%** | **8.57%** | **26.67%** |
| **3. KNN** (Default) | $k=5$, cosine, uniform | 76.0% | 66.67% | 40.00% | 50.00% | 67.11% | 8.57% | 60.00% |
| **3. KNN** *(Fine-Tuned)* | $k=4$, cosine, uniform | 74.0% | 56.25% | **60.00%** | 58.06% | 69.61% | 20.00% | 40.00% |
| **4. Hybrid Dense Model** (Default) | $C=1.5$, balanced, $\tau=0.50$ | 74.0% | 56.25% | 60.00% | 58.06% | 69.61% | 20.00% | 40.00% |
| **4. Hybrid Dense Model** *(Fine-Tuned)* | $C=0.5$, balanced, $\tau=0.50$ | **76.0%** | 58.82% | **66.67%** | **62.50%** | **72.43%** | 20.00% | **33.33%** |

### Key Theoretical Insights
1. **Random Forest Achieves Peak Performance:** Constraining tree depth to `max_depth=4` and feature subsampling to `max_features=0.3` prevents the ensemble from overfitting to idiosyncratic vocabulary in specific annual reports. It achieved the highest overall Accuracy (**86.0%**) and lowest greenwashing false-positive rate (**8.57% Type I Error**).
2. **$L_1$ Regularization Benefits in Logistic Regression:** In high-dimensional TF-IDF feature space, $L_1$ Lasso regularization forces uninformative accounting jargon coefficients to zero, boosting substantive F1 from **51.85%** to **64.29%**.
3. **Curse of Dimensionality in KNN:** In sparse text vector spaces, distance metrics suffer from concentration of distances. Tuning to $k=4$ with cosine similarity restored Substantive Recall to **60.00%**.
4. **Dense Embeddings Minimize Missed Action:** The Hybrid Dense Model achieved the highest Substantive Recall (**66.67%**), proving that dense multilingual transformers capture operational context even when exact keywords differ.

---

## 8. Repository Architecture & Modular File Map

```
sdg-depth-disclosure-benchmark/
├── README.md                            <-- This Master Documentation
├── KNOWLEDGE_TRANSFER.md                <-- Exhaustive technical reference manual
├── requirements.txt                     <-- Curated, pinned Python dependencies
├── requirements-lock.txt                <-- Full freeze of all 160+ packages
├── environment.yml                      <-- Conda environment replication file
├── .gitattributes                       <-- Git LFS configuration for datasets/models
├── .gitignore                           <-- Excludes raw PDFs (20 GB) & temporary files
│
├── 01_data_collection/                  <-- TASK 1: Report Ingestion & PDF Parsing
│   ├── README.md                        <-- Multi-engine scraper & PyMuPDF guide
│   ├── download_reports.py              <-- Multi-engine PDF downloader (DDG/Bing/Yahoo)
│   ├── extract_text.py                  <-- Multiprocess PyMuPDF text extractor
│   ├── extract_companies.py             <-- Company manifest builder
│   └── audit_reports.py                 <-- Fiscal year coverage auditor
│
├── 02_passage_extraction/               <-- TASK 2: SDG Keyword Matching & Chunking
│   ├── README.md                        <-- Multilingual regex & sliding-window guide
│   ├── split_passages.py                <-- Sentence-boundary 512-token chunker
│   ├── sdg_keyword_match.py             <-- 17 SDG regex matchers (German & English)
│   ├── pipeline_sdg_filter.py           <-- Standalone DuckDB passage extractor
│   └── kw_data/                         <-- Curated SDG keyword dictionaries (EN & DE)
│
├── 03_human_reliability/                <-- TASK 3: Ground Truth & Dual-Coder Protocol
│   ├── README.md                        <-- Inter-coder agreement & Kappa guide
│   ├── CODING_GUIDELINES.md             <-- Human coder decision manual & flowchart
│   ├── setup_two_coders.py              <-- Generates blinded sheets for Coder A & B
│   ├── analyze_coder_agreement.py       <-- Computes Kappa, Pearson r, Spearman rho
│   └── reconcile_coders.py              <-- Disagreement consensus & golden truth engine
│
├── 04_depth_scoring_engine/             <-- TASK 4: Continuous 0-5 Depth Scoring Engine
│   ├── README.md                        <-- Scoring rubric, model architecture & inference
│   ├── scoring_rubric.md                <-- Official Table 1 matrix & definitions
│   ├── continuous_model.py              <-- Production pipeline class (Transformer + Ridge)
│   ├── train_continuous_depth_model.py  <-- 5-fold CV training script
│   └── predict_depth.py                 <-- Batch inference CLI with auto-rationales
│
├── 05_model_benchmarking/               <-- TASK 5: ML Benchmarks & Tuning (Shah et al.)
│   ├── README.md                        <-- LR vs RF vs KNN vs Hybrid comparison
│   ├── train_shah_benchmark.py          <-- 5-fold stratified CV baseline benchmark
│   ├── finetune_hyperparameters.py      <-- Grid search optimization & cutoff calibration
│   └── train_glassbox_ebm.py            <-- Explainable Boosting Machine (EBM)
│
├── 06_pilot_50_suite/                   <-- TASK 6: Turn-Key 50-Sample Verification Suite
│   ├── README.md                        <-- Pilot Quickstart & verification walkthrough
│   ├── run_pipeline_50.py               <-- One-click runner for 50 samples
│   ├── evaluate_50.py                   <-- Metric benchmark against human consensus
│   ├── model_depth_50.joblib            <-- Serialized continuous model artifact
│   ├── output_predictions_50.xlsx       <-- 50 samples with scores & Table 1 rationales
│   ├── finetuned_benchmark_results.xlsx <-- Fine-tuning comparison table
│   └── data/                            <-- Blank sheets, ground truth & sample extracts
│
├── sdg_filtered_passages/               <-- EXPORT: Dedicated Folder for Supervisor Sharing
│   ├── README.md                        <-- Data dictionary & supervisor summary
│   ├── all_sdg_filtered_passages.csv    <-- 30,434 unique passages (~74 MB)
│   ├── ground_truth_full.csv            <-- 1,242,827 keyword pairs (~158 MB, Git LFS)
│   ├── ground_truth_full.xlsx           <-- Master Excel workbook (~28.4 MB)
│   ├── golden_dataset_1000.xlsx         <-- 1,000 stratified samples for instant Excel review
│   └── summary_by_sdg_category.csv      <-- Breakdown across all 17 SDGs
│
└── data/                                <-- Master datasets & manifests
    ├── companies.csv                    <-- 151 German company master index
    ├── companies.json                   <-- Company search slugs
    └── dbs/sdg_hits.duckdb              <-- DuckDB pre-indexed keyword hits (~98 MB)
```

---

## 9. Environment Setup & Reproduction Manual

### Option A: Using Existing Conda Environment (`escp`)
If running on the authoring workstation:
```powershell
conda activate escp
```

### Option B: Clean Conda Recreation
```powershell
# Create environment from YAML specification
conda env create -f environment.yml
conda activate escp
```

### Option C: Standard Python Virtual Environment (`venv`)
```bash
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows PowerShell:
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 30-Second Turnkey Verification
Run the 50-sample pilot pipeline to verify all model weights and metric outputs:
```powershell
# 1. Run inference on 50 samples
python 06_pilot_50_suite/run_pipeline_50.py

# 2. Evaluate against human ground truth
python 06_pilot_50_suite/evaluate_50.py
```

### Running Model Benchmarks & Fine-Tuning
```powershell
# Run baseline Shah et al. (2020) comparison
python 05_model_benchmarking/train_shah_benchmark.py

# Run fine-tuning grid search
python 05_model_benchmarking/finetune_hyperparameters.py
```

---

## 10. Git LFS Configuration & GitHub Push Walkthrough

### Why Git LFS is Required
GitHub enforces a **strict 100 MB hard limit** on individual file uploads. In this repository:
* `sdg_filtered_passages/ground_truth_full.csv` is **~158 MB**.
* `data/dbs/sdg_hits.duckdb` is **~98 MB**.
* Serialized models and Excel workbooks range from **20 MB to 30 MB**.

Attempting a standard `git push` without Git LFS will result in an immediate rejection:
`remote: error: GH001: Large files detected. File exceeds 100.00 MB`.

### Exact Step-by-Step Push Instructions (Run in PowerShell)

Open PowerShell in the project root directory:

```powershell
# Step 1: Ensure Git LFS is installed and initialized on your system
git lfs install

# Step 2: Initialize Git repository (if not already initialized)
git init

# Step 3: Track large file patterns via Git LFS
git lfs track "*.csv"
git lfs track "*.xlsx"
git lfs track "*.duckdb"
git lfs track "*.joblib"

# Step 4: Ensure .gitattributes and .gitignore are staged first
git add .gitattributes .gitignore
git commit -m "chore: configure Git LFS tracking and ignore raw PDF reports"

# Step 5: Stage all project code, documentation, and data assets
git add .
git commit -m "feat: complete corporate SDG depth scoring engine, ML benchmarks, and filtered corpus"

# Step 6: Set branch to main and link to your GitHub repository
git branch -M main
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPOSITORY_NAME>.git

# Step 7: Push all code and LFS data objects to GitHub
git push -u origin main
```

---

## 11. Technical Gotchas & Robust Engineering Solutions

### 1. XML Control Characters in OpenPyXL (`IllegalCharacterError`)
* **Problem:** Text extracted directly from corporate PDF streams contains raw form feeds (`\x0c`), vertical tabs (`\x0b`), and ASCII control characters (`\x00` to `\x1f`). When writing DataFrames to `.xlsx`, `openpyxl` raises `openpyxl.utils.exceptions.IllegalCharacterError`.
* **Solution:** Vectorized regex sanitization is applied across all string columns before writing:
  ```python
  for col in df.select_dtypes(include="object").columns:
      df[col] = df[col].astype(str).str.replace(
          r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]", "", regex=True
      )
  ```

### 2. Windows Console Unicode Encoding
* **Problem:** Windows PowerShell / CMD terminal defaults to `cp1252` encoding, crashing on mathematical symbols ($\kappa$, $\rho$, $\pm$) or German umlauts (*ä, ö, ü, ß*).
* **Solution:** Explicit UTF-8 stdout reconfiguration at the entrypoint of all scripts:
  ```python
  import sys
  sys.stdout.reconfigure(encoding='utf-8')
  ```

### 3. Binary Class Label Slicing in `predict_proba`
* **Problem:** Scikit-learn sorts binary class labels alphabetically (`['sub', 'sym']`). Assuming index 1 corresponds to `'sub'` causes inverted decision cutoffs.
* **Solution:** Dynamic index resolution:
  ```python
  sub_idx = list(clf.classes_).index('sub')
  sub_probs = clf.predict_proba(X)[:, sub_idx]
  ```

---

## 12. Academic Citations & BibTeX Reference

If you utilize this codebase, methodology, or dataset in academic research, please cite:

```bibtex
@article{izhar2026exploring,
  title={Exploring firm-level SDG engagement in an emerging economy through content analysis},
  author={Izhar, Muhammad and others},
  journal={Journal of Cleaner Production},
  year={2026},
  publisher={Elsevier}
}

@article{shah2020comparative,
  title={A Comparative Analysis of Logistic Regression, Random Forest and KNN Models for Text Classification},
  author={Shah, Kathan and Patel, Harshil and Sanghvi, Devavrat and Shah, Manan},
  journal={Augmented Human Research},
  volume={5},
  number={1},
  pages={12},
  year={2020},
  publisher={Springer}
}

@article{landis1977measurement,
  title={The measurement of observer agreement for categorical data},
  author={Landis, J Richard and Koch, Gary G},
  journal={Biometrics},
  pages={159--174},
  year={1977}
}

@book{krippendorff2004content,
  title={Content Analysis: An Introduction to Its Methodology},
  author={Krippendorff, Klaus},
  edition={2nd},
  year={2004},
  publisher={Sage Publications}
}
```

---

## 🛡️ License
This project is open-source software licensed under the **MIT License**.
