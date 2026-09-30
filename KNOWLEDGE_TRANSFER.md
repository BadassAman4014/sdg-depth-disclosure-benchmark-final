# 📘 Comprehensive Knowledge Transfer & Technical Master Documentation
## Automated Corporate SDG Disclosure Depth Scoring & Machine Learning Benchmarks
### Empirical Study of 151 German Listed Companies (DAX, MDAX, SDAX) Across Fiscal Years 2014–2024

---

## 📑 Table of Contents
1. [Executive Summary & Problem Statement](#1-executive-summary--problem-statement)
2. [Academic Foundations & Theoretical Framework](#2-academic-foundations--theoretical-framework)
3. [Full Corpus Ingestion & Data Pipeline Funnel](#3-full-corpus-ingestion--data-pipeline-funnel)
4. [Human Ground Truth & Inter-Coder Reliability Protocol](#4-human-ground-truth--inter-coder-reliability-protocol)
5. [Continuous 0–5 Depth Scoring Engine (Izhar et al. 2026)](#5-continuous-05-depth-scoring-engine-izhar-et-al-2026)
6. [Machine Learning Benchmarks & Hyperparameter Tuning (Shah et al. 2020)](#6-machine-learning-benchmarks--hyperparameter-tuning-shah-et-al-2020)
7. [Repository File Map & Module Directory](#7-repository-file-map--module-directory)
8. [Step-by-Step Reproduction & Execution Manual](#8-step-by-step-reproduction--execution-manual)
9. [Technical Gotchas, Edge Cases & Solutions](#9-technical-gotchas-edge-cases--solutions)

---

## 1. Executive Summary & Problem Statement

### The Research Challenge
In corporate sustainability research, analyzing Sustainable Development Goal (SDG) reporting presents two major methodological obstacles:
1. **The Greenwashing / Symbolic Disclosure Gap:** Companies frequently include glossy sustainability pledges, broad aspirations, and regulatory tables of contents without providing concrete, measurable, or auditable evidence of operational action. Standard dictionary keyword counters (e.g., counting the frequency of the word *"climate"*) fail because they treat an empty aspiration (*"we strive to protect the climate"*) identically to an audited operational result (*"reduced Scope 1 emissions by 24.5% verified by PwC"*).
2. **Binary vs. Depth Granularity:** Treating disclosures as purely binary (*"mentions SDG"* vs. *"does not mention SDG"*) discards the critical spectrum of disclosure depth. 

### The Solution Developed
This computational framework solves both problems by establishing an end-to-end, scientifically validated natural language processing (NLP) and machine learning pipeline that:
* **Ingests and processes 1,401 full-length corporate reports** (covering 151 German listed equities across 11 fiscal years).
* **Segments text into 453,404 semantically preserved 512-token passages**.
* **Filters 30,434 unique SDG passages** using 6,218 expert-compiled multilingual regex patterns (German and English) across all 17 UN SDGs.
* **Validates human ground truth** with an independent dual-coder protocol achieving **98.0% agreement** and a Quadratic Weighted Cohen's Kappa of **$\kappa_w = 0.9896$**.
* **Scores disclosure depth on a continuous scale ($0.00$ to $5.00$) and discrete levels ($0$ to $5$)** grounded in *Izhar et al. (2026)* Table 1, achieving **$r = 0.9584$** correlation with human consensus and **100% within $\pm 1$ level agreement**.
* **Benchmarks classical and neural ML architectures** grounded in *Shah et al. (2020)*, where fine-tuned Random Forest achieved **86.0% accuracy** and an **83.0% Macro F1-score**.

---

## 2. Academic Foundations & Theoretical Framework

The entire methodology is anchored in five foundational peer-reviewed studies:

### 1. Depth Scoring Rubric: *Izhar et al. (2026)*
* **Citation:** Izhar, M. et al. (2026). *Exploring firm-level SDG engagement in an emerging economy through content analysis.* Journal of Cleaner Production.
* **Core Contribution:** Table 1 and Section 3.4.3 establish that disclosure depth is determined by the intersection of three components: **Targets**, **Actions**, and **Outcomes**.

#### The Table 1 Scoring Matrix
| Score Level | Description / Theoretical Construct | Qualitative Target | Quantitative Target | Implemented Actions | Qualitative Outcome | Quantitative / Audited Outcome | Construct Class |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **0** | **No SDG Information / Pure Boilerplate** | ❌ | ❌ | ❌ | ❌ | ❌ | **Symbolic (`sym`)** |
| **1** | **Qualitative Target Only (Aspiration)** | ✔️ | ❌ | ❌ | ❌ | ❌ | **Symbolic (`sym`)** |
| **2** | **Qual Target + Actions** OR **Quant Target** | ✔️ *(or ❌)* | *(or ✔️)* ❌ | ✔️ *(or ❌)* | ❌ | ❌ | **Substantive (`sub`)** |
| **3** | **Qual Target + Qualitative Outcome** | ✔️ | ❌ | *(often ✔️)* | ✔️ | ❌ | **Substantive (`sub`)** |
| **4** | **Qual Target + Quantitative Outcome** | ✔️ | ❌ | *(often ✔️)* | ❌ | ✔️ | **Substantive (`sub`)** |
| **5** | **Quant Target + Measured / Audited Outcome** | ❌ | ✔️ | *(often ✔️)* | ✔️ | ✔️ | **Substantive (`sub`)** |

### 2. Multi-Model Text Classification Benchmark: *Shah et al. (2020)*
* **Citation:** Shah, K., Patel, H., Sanghvi, D., & Shah, M. (2020). *A Comparative Analysis of Logistic Regression, Random Forest and KNN Models for Text Classification.* Augmented Human Research, 5(1), 12.
* **Core Contribution:** Establishes the comparative baseline between linear models (Logistic Regression), ensemble bagging (Random Forest), and instance-based learners (KNN) in high-dimensional text spaces. We augment their framework with **Multilingual Dense Transformer Embeddings** (*MiniLM-L12-v2*) and **Glassbox Explainable Boosting Machines (EBM)**.

### 3. Complementary Foundations
* **Hummel, K. (2019) & PwC (2018):** Pioneered the criteria of third-party audit verification (Big 4, TÜV) as the hallmark of maximum substantive depth.
* **Landis, J. R., & Koch, G. G. (1977):** Established the statistical benchmark for observer agreement: $\kappa > 0.80$ denotes *Almost Perfect Agreement*.
* **Krippendorff, K. (2004):** *Content Analysis: An Introduction to Its Methodology.* Mandates dual-coder blinding and formal adjudication for scientific reproducibility.

---

## 3. Full Corpus Ingestion & Data Pipeline Funnel

### 1. Corporate Universe
The sample comprises **151 German listed corporations** representing the constituents of the three primary German equity indices:
* **DAX 40:** Major blue chips (e.g., *adidas, Airbus, Allianz, BASF, Bayer, BMW, Continental, Deutsche Bank, Deutsche Telekom, Mercedes-Benz, SAP, Siemens, Volkswagen*).
* **MDAX 50:** Mid-cap industrial leaders (e.g., *Aixtron, Aurubis, Bechtle, Brenntag, Evonik, GEA Group, HelloFresh, Hugo Boss, KION, Lanxess, Puma, Thyssenkrupp, TUI*).
* **SDAX 70:** Small-cap growth firms (e.g., *1&1, Atoss Software, AUTO1, Cancom, Deutz, Fielmann, Hensoldt, Hornbach, Krones, Nordex, Salzgitter, Siltronic, Varta, Verbio, Wacker Chemie*).

### 2. The Ingestion Funnel
```
1. RAW REPORTS: 1,654 PDF Files Downloaded (~20+ GB on disk)
   ├── Tracked in: reports/reports_manifest.csv
   └── Multi-engine crawler with anti-bot rotation (DuckDuckGo, Bing, Yahoo)
          │
          ▼
2. TEXT EXTRACTION: 1,401 Reports Successfully Converted
   ├── Multiprocess PyMuPDF extraction (data/extract_text.log: OK=1401)
   └── Validated minimum length threshold (MIN_TEXT_LEN = 500 chars)
          │
          ▼
3. PASSAGE CHUNKING: 453,404 Raw Text Chunks Produced
   ├── NLTK sentence boundary tokenization + Tiktoken cl100k_base
   ├── Greedy packing up to 512 tokens with context overlap
   └── Logged in: data/split_passages.log (TotalPassages=453404)
          │
          ▼
4. MULTILINGUAL SDG REGEX FILTERING:
   ├── Scanned with 3,128 English + 3,090 German expert regex patterns
   └── Evaluates all 17 SDGs (kw_data/keywords_sdg.json & keywords_sdg_de.json)
          │
          ▼
5. FILTERED DISCLOSURE DATASET:
   ├── 30,434 UNIQUE SDG PASSAGES (~74 MB, all_sdg_filtered_passages.csv)
   └── 1,242,827 KEYWORD MATCH PAIRS (~158 MB, ground_truth_full.csv)
```

### 3. Why 30,434 Unique Passages?
Annual reports average 250–400 pages. More than 90% of pages contain consolidated financial accounting statements (IAS/IFRS balance sheets, equity changes, cash flow notes, tax reconciliations, and board remuneration) which do not discuss corporate sustainability. 
Out of **453,404 total text chunks**, **30,434 passages (~6.7%)** contained verified SDG keywords. This aligns perfectly with academic benchmarks for corporate non-financial disclosures.

---

## 4. Human Ground Truth & Inter-Coder Reliability Protocol

To train and evaluate supervised models rigorously, a dual-coder blinding protocol was executed:

```
Candidate Passages (samples_50.xlsx)
            │
            ├──> coding_sheet_person_A_50.xlsx (Coder 1: Blinded, Forest Green Theme)
            └──> coding_sheet_person_B_50.xlsx (Coder 2: Blinded, Burgundy Theme)
                        │
                        ▼
             Independent Annotation (0-5 Rubric)
                        │
                        ▼
             analyze_coder_agreement.py
```

### Statistical Reliability Results (Person A vs. Person B, N = 50)
* **Exact Percentage Agreement:** **`98.0%`** (49 out of 50 passages scored identically).
* **Within $\pm 1$ Level Agreement:** **`100.0%`** (50 out of 50 passages within 1 score point).
* **Unweighted Cohen's Kappa ($\kappa$):** **`0.9686`** (Far exceeding the 0.80 benchmark for near-perfect agreement).
* **Linear Weighted Kappa ($\kappa_{lin}$):** **`0.9805`**.
* **Quadratic Weighted Kappa ($\kappa_{quad}$):** **`0.9896`** (Academic gold standard for ordinal scales).
* **Pearson Correlation ($r$):** **`0.9943`** ($p < 10^{-45}$).
* **Spearman Rank Correlation ($\rho$):** **`0.9940`**.

### Adjudication of Disagreements
Exactly 1 natural disagreement occurred:
* **Sample ID 5 (1&1 AG):** Person A assigned **Score 2** (focusing on concrete HR actions implemented under `#MYBEST`), while Person B assigned **Score 1** (viewing `#MYBEST` as an internal aspirational policy framework).
* **Resolution:** Adjudicated as **Score 2** because the text reported concrete implemented training modules and quality programs in 2024.

---

## 5. Continuous 0–5 Depth Scoring Engine (*Izhar et al. 2026*)

### 1. Hybrid Engineering Architecture
The scoring model combines dense semantic representations with symbolic, rule-based linguistic features:

1. **Multilingual Dense Transformer Embeddings:**
   * Utilizes `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384-dimensional dense vectors).
   * Encodes semantic contextual meaning uniformly across German and English reporting texts.
2. **Theory-Guided Linguistic Feature Extractors:**
   * `has_audit`: Matches Big 4 auditors (PwC, KPMG, EY, Deloitte), technical inspectors (TÜV, Bureau Veritas), and environmental management certifications (ISO 14001, ISO 50001).
   * `metric_count`: Quantified figures and physical units (`%`, `tCO2e`, `tonnes`, `MWh`, `GWh`, `million EUR`).
   * `action_count`: Past-tense operational action verbs (*"implemented"*, *"installed"*, *"commissioned"*, *"invested"*, *"constructed"*, *"trained"*).
   * `asp_count`: Aspirational modal verbs (*"aim to"*, *"strive"*, *"aspire"*, *"commit"*, *"pledge"*, *"vision"*).
   * `qual_outcome`: Qualitative progress indicators (*"improved"*, *"enhanced"*, *"strengthened"*).
   * `evidence_density`: Normalized compound metric weighting hard evidence against text length:
     $$\text{Evidence Density} = \frac{2.0 \cdot \text{metrics} + 1.5 \cdot \text{actions} + 3.0 \cdot \text{audits}}{\text{word count} + 1}$$
3. **Calibrated Continuous Supervised Regressor:**
   * A penalized Ridge regressor ($L_2$ regularization $\alpha = 1.5$) maps the stacked hybrid representation into a real-valued continuous depth score:
     $$\hat{y}_{\text{continuous}} = \text{clip}\left(\mathbf{w}^T \mathbf{x} + b, 0.00, 5.00\right)$$
   * Binned into Table 1 discrete levels: $\hat{y}_{\text{level}} = \text{round}(\hat{y}_{\text{continuous}})$.

### 2. Empirical Model Performance vs. Human Ground Truth (N = 50)
* **Pearson Correlation ($r$):** **`0.9584`** ($p = 8.77 \times 10^{-28}$) — Near-perfect linear alignment.
* **Spearman Rank Correlation ($\rho$):** **`0.9051`** — High monotonic ranking fidelity.
* **Root Mean Squared Error (RMSE):** **`±0.453`** points on a 6-point scale (under half a level).
* **Mean Absolute Error (MAE):** **`±0.339`** points (average deviation ~1/3 of a score level).
* **Exact Integer Match Accuracy:** **`80.0%`** (40 / 50 exact matches).
* **Within $\pm 1$ Level Agreement:** **`100.0%`** (50 / 50 passages within 1 point of human truth).
* **Quadratic Weighted Cohen's Kappa ($\kappa_w$):** **`0.9072`** (Almost Perfect Agreement).
* **High-Level Binary Accuracy (`sym` vs `sub`):** **`88.0%`** ($\kappa = 0.6907$).

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
*(Notice: All deviations are strictly on adjacent off-diagonals, with zero extreme errors).*

---

## 6. Machine Learning Benchmarks & Hyperparameter Tuning (*Shah et al. 2020*)

Evaluated via **5-Fold Stratified Cross-Validation** on verified human ground truth:

### Comparative Performance Table
| Model Architecture | Hyperparameter Configuration | Accuracy | Precision (`sub`) | Recall (`sub`) | F1-Score (`sub`) | Macro F1 | Type I (FP) | Type II (FN) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1. Logistic Regression** (Default) | $C=1.0$, $L_2$, lbfgs | 74.0% | 58.33% | 46.67% | 51.85% | 67.02% | 14.29% | 53.33% |
| **1. Logistic Regression** *(Fine-Tuned)* | $C=5.0$, $L_1$, liblinear, balanced | **80.0%** | **69.23%** | **60.00%** | **64.29%** | **75.20%** | 11.43% | 40.00% |
| **2. Random Forest** (Default) | $n=100$, depth=6 | 80.0% | 69.23% | 60.00% | 64.29% | 75.20% | 11.43% | 40.00% |
| **2. Random Forest** *(Fine-Tuned [BEST])* | $n=200$, depth=4, feat=0.3, balanced | **86.0%** | **78.57%** | **73.33%** | **75.86%** | **83.00%** | **8.57%** | **26.67%** |
| **3. KNN** (Default) | $k=5$, cosine, uniform | 76.0% | 66.67% | 40.00% | 50.00% | 67.11% | 8.57% | 60.00% |
| **3. KNN** *(Fine-Tuned)* | $k=4$, cosine, uniform | 74.0% | 56.25% | **60.00%** | 58.06% | 69.61% | 20.00% | 40.00% |
| **4. Hybrid Dense Model** (Default) | $C=1.5$, balanced, $\tau=0.50$ | 74.0% | 56.25% | 60.00% | 58.06% | 69.61% | 20.00% | 40.00% |
| **4. Hybrid Dense Model** *(Fine-Tuned)* | $C=0.5$, balanced, $\tau=0.50$ | **76.0%** | 58.82% | **66.67%** | **62.50%** | **72.43%** | 20.00% | **33.33%** |

### Key Theoretical Findings
1. **Random Forest Achieves Overall Superiority:**
   * Setting `max_depth=4` and `max_features=0.3` prevents individual decision trees from overfitting to sparse vocabulary noise. It achieved the highest overall Accuracy (**86.0%**) and lowest greenwashing false-positive rate (**8.57% Type I Error**).
2. **Sparsity Penalty in Logistic Regression:**
   * Switching to $L_1$ regularization (`liblinear`) zeroes out uninformative background tokens, boosting Substantive F1 from **51.85%** to **64.29%**.
3. **Curse of Dimensionality in KNN:**
   * In 500-dimensional TF-IDF space, Euclidean and Cosine distances tend toward equidistance. Fine-tuning to $k=4$ with cosine weighting recovered substantive recall to **60.00%** (up from 40.00%).
4. **Hybrid Dense Model Minimizes Missed Action:**
   * Combining multilingual dense embeddings with theory rules delivers the highest substantive recall (**66.67%**), reducing Type II false negatives to **33.33%**.

---

## 7. Repository File Map & Module Directory

The repository is modularized into dedicated task directories, each containing its own standalone `README.md`:

```
sdg-depth-disclosure-benchmark/
├── README.md                            <-- Master Architecture & Quickstart
├── KNOWLEDGE_TRANSFER.md                <-- This Comprehensive Technical Manual
├── requirements.txt                     <-- Curated, pinned dependencies (from escp)
├── requirements-lock.txt                <-- Full freeze of all 160+ packages
├── environment.yml                      <-- Conda environment replication recipe
│
├── 01_data_collection/                  <-- TASK 1: Report Ingestion & PDF Parsing
│   ├── README.md                        <-- Multi-engine crawler & PyMuPDF guide
│   ├── download_reports.py              <-- Multi-engine PDF downloader (DDG/Bing/Yahoo)
│   ├── extract_text.py                  <-- Multiprocess PyMuPDF text extractor
│   ├── extract_companies.py             <-- Company manifest builder
│   └── audit_reports.py                 <-- Fiscal year coverage auditor
│
├── 02_passage_extraction/               <-- TASK 2: SDG Keyword Matching & Chunking
│   ├── README.md                        <-- Multilingual regex & sliding-window guide
│   ├── split_passages.py                <-- Sentence-boundary 512-token chunker
│   ├── sdg_keyword_match.py             <-- 17 SDG regex matchers (German & English)
│   ├── pipeline_sdg_filter.py           <-- Minimalist DuckDB extractor script
│   └── kw_data/                         <-- Curated SDG keyword dictionaries
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
│   ├── ground_truth_full.csv            <-- 1,242,827 keyword pairs (~158 MB)
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

## 8. Step-by-Step Reproduction & Execution Manual

### Step 1: Environment Setup
```bash
# Clone the repository
git clone https://github.com/<your-username>/sdg-depth-disclosure-benchmark.git
cd sdg-depth-disclosure-benchmark

# Create environment from Conda recipe
conda env create -f environment.yml
conda activate escp

# Or install via pip into an existing Python 3.11 environment
pip install -r requirements.txt
```

### Step 2: Run 50-Sample Pilot Benchmark (30 Seconds)
```bash
# Generate predictions on 50 passages
python 06_pilot_50_suite/run_pipeline_50.py

# Evaluate against human ground truth
python 06_pilot_50_suite/evaluate_50.py
```

### Step 3: Run Text Classification Benchmarks & Fine-Tuning
```bash
# Baseline comparison (Shah et al. 2020)
python 05_model_benchmarking/train_shah_benchmark.py

# Exhaustive grid search optimization
python 05_model_benchmarking/finetune_hyperparameters.py
```

### Step 4: Export Filtered Passages
```bash
python sdg_filtered_passages/pipeline_sdg_filter.py
```

---

## 9. Technical Gotchas, Edge Cases & Solutions

### 1. XML Illegal Control Characters in Excel (`OpenPyXL`)
* **Problem:** Text extracted directly from corporate PDFs contains raw form-feeds (`\x0c`), vertical tabs (`\x0b`), and null bytes (`\x00`). When writing to `.xlsx`, `openpyxl` raises `openpyxl.utils.exceptions.IllegalCharacterError`.
* **Solution:** Apply a vector regex replacement across all string columns before calling `to_excel()`:
  ```python
  for col in df.select_dtypes(include="object").columns:
      df[col] = df[col].astype(str).str.replace(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]", "", regex=True)
  ```

### 2. Windows Console Unicode Encoding
* **Problem:** Windows PowerShell / CMD terminal defaults to `cp1252` encoding, crashing on mathematical symbols ($\kappa$, $\rho$, $\pm$) or non-ASCII characters.
* **Solution:** Force standard UTF-8 stdout reconfiguration at the top of every CLI script:
  ```python
  import sys
  sys.stdout.reconfigure(encoding='utf-8')
  ```

### 3. Binary Class Label Slicing in `predict_proba`
* **Problem:** Scikit-learn orders classes alphabetically (`['sub', 'sym']`). Assuming index 1 corresponds to `'sub'` causes inverted decision cutoffs.
* **Solution:** Dynamically resolve the positive class index:
  ```python
  sub_idx = list(clf.classes_).index('sub')
  sub_probs = clf.predict_proba(X)[:, sub_idx]
  ```

---

## 10. Summary for Academic Citation

```bibtex
@article{izhar2026exploring,
  title={Exploring firm-level SDG engagement in an emerging economy through content analysis},
  author={Izhar, Muhammad and others},
  journal={Journal of Cleaner Production},
  year={2026}
}

@article{shah2020comparative,
  title={A Comparative Analysis of Logistic Regression, Random Forest and KNN Models for Text Classification},
  author={Shah, Kathan and Patel, Harshil and Sanghvi, Devavrat and Shah, Manan},
  journal={Augmented Human Research},
  volume={5},
  number={1},
  pages={12},
  year={2020}
}
```
