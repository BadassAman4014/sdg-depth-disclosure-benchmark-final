# 🎯 Task 04: Continuous 0–5 SDG Depth Scoring Engine

This module implements the production-grade continuous depth scoring engine grounded in **Section 3.4.3 & Table 1** of *Izhar et al. (2026)* (*Exploring firm-level SDG engagement in an emerging economy through content analysis*), augmented with methodologies from *Hummel (2019)* and *PwC (2018)*.

---

## 📌 Scoring Rubric (*Izhar et al., 2026, Table 1*)

Disclosures are evaluated across five concrete operational components:

| Score | Description / Theoretical Construct | Qualitative Target | Quantitative Target | Implemented Actions | Qualitative Outcome | Quantitative / Audited Outcome | Construct Mapping |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **0** | **No SDG information / Boilerplate** | ❌ | ❌ | ❌ | ❌ | ❌ | **Symbolic (`sym`)** |
| **1** | **Qualitative target only (Aspiration)** | ✔️ | ❌ | ❌ | ❌ | ❌ | **Symbolic (`sym`)** |
| **2** | **Qual target + Actions** OR **Quant target** | ✔️ <br> *(or ❌)* | *(or ✔️)* <br> ❌ | ✔️ <br> *(or ❌)* | ❌ | ❌ | **Substantive (`sub`)** |
| **3** | **Qual target + Qualitative outcome** | ✔️ | ❌ | *(often ✔️)* | ✔️ | ❌ | **Substantive (`sub`)** |
| **4** | **Qual target + Quantitative outcome** | ✔️ | ❌ | *(often ✔️)* | ❌ | ✔️ | **Substantive (`sub`)** |
| **5** | **Quant target + Measured / Audited outcome**| ❌ | ✔️ | *(often ✔️)* | ✔️ | ✔️ | **Substantive (`sub`)** |

---

## 🧠 Model Architecture

The depth engine combines **semantic embeddings** with **theory-guided linguistic features**:

```
Input Passage (German or English)
        │
        ├──> Multilingual SentenceTransformer (paraphrase-multilingual-MiniLM-L12-v2) ──> 384-d dense vector
        │
        ├──> Linguistic Feature Extractor ──> [has_audit, metric_count, action_count, asp_count, evidence_density]
        │
        └──> TF-IDF N-gram Vectorizer (1-2 ngrams, sublinear TF) ──> Sparse term vocabulary
                 │
                 ▼
         Stacked Hybrid Feature Representation (X_hybrid)
                 │
                 ▼
         Calibrated Ridge Continuous Regressor
                 │
                 ├──> continuous_score: Real value in [0.00, 5.00] (e.g. 3.65)
                 ├──> izhar_level: Discrete integer in {0, 1, 2, 3, 4, 5}
                 ├──> binary_label: 'sym' (0-1) vs. 'sub' (2-5)
                 └──> scoring_rationale: Auto-generated Table 1 explanation
```

---

## 🚀 Key Scripts & Assets

| File | Purpose |
|---|---|
| [`scoring_rubric.md`](scoring_rubric.md) | Official Table 1 matrix, linguistic definitions, and coder flowchart. |
| [`continuous_model.py`](continuous_model.py) | Self-contained, serializable production pipeline class (`ContinuousDepthPipeline`). |
| [`train_continuous_depth_model.py`](train_continuous_depth_model.py) | Trains and calibrates the regressor on ground truth with 5-fold cross-validation. |
| [`predict_depth.py`](predict_depth.py) | High-speed batch inference script generating continuous scores, discrete levels, and rationales. |

---

## 🛠️ Step-by-Step Execution Guide

### 1. Train and Cross-Validate the Depth Model
```bash
# Run 5-fold cross-validation and save model artifact
python 04_depth_scoring_engine/train_continuous_depth_model.py \
    --data 06_pilot_50_suite/data/golden_truth_50.xlsx \
    --out-model 04_depth_scoring_engine/model_depth_continuous.joblib \
    --cv
```
* **Expected Cross-Validation Results (N = 50 Pilot):**
  * **Pearson Correlation ($r$):** **`0.9584`** ($p = 8.77 \times 10^{-28}$)
  * **Within $\pm 1$ Level Agreement:** **`100.0%`**
  * **Root Mean Squared Error (RMSE):** **`±0.453`** points
  * **Quadratic Weighted Kappa ($\kappa_w$):** **`0.9072`** (Almost Perfect)

### 2. Predict on New Corporate Passages
```bash
python 04_depth_scoring_engine/predict_depth.py \
    --model 04_depth_scoring_engine/model_depth_continuous.joblib \
    --input 06_pilot_50_suite/data/samples_50.xlsx \
    --out data/predictions_output.xlsx
```
* Generates both an Excel spreadsheet and a UTF-8 BOM CSV containing full predictions and Table 1 rationales.
