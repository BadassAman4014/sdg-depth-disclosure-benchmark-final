# 👥 Task 03: Human Inter-Coder Reliability & Ground Truth Validation

This module establishes an audit-proof, scientifically rigorous human ground truth by implementing a blinded, dual-coder annotation protocol with statistical inter-coder reliability metrics grounded in academic literature (*Izhar et al., 2026*; *Krippendorff, 2004*; *Landis & Koch, 1977*).

---

## 📌 Architectural Overview

```
Candidate Passages (samples_50.xlsx / sample_1000.csv)
                      │
                      ▼
            setup_two_coders.py 
           /                    \
          ▼                      ▼
coding_sheet_person_A.xlsx    coding_sheet_person_B.xlsx
  (Blinded Coder 1)             (Blinded Coder 2)
          \                      /
           ▼                    ▼
        analyze_coder_agreement.py ──> Kappa (unweighted, linear, quadratic), r, rho
                      │
                      ▼
             reconcile_coders.py   ──> Adjudication & Golden Truth Generation
                      │
                      ▼
          data/golden_truth.xlsx   ──> Input for Machine Learning Models
```

---

## 🔬 Inter-Coder Reliability Metrics Explained

| Metric | Formula / Construct | Interpretation Thresholds | Why It Matters for Corporate Research |
|---|---|---|---|
| **Exact Agreement** | $P_o = \frac{\sum \text{agree}}{N}$ | $> 80\%$ standard, $> 90\%$ excellent | Measures percentage of passages given the exact same 0–5 depth score. |
| **Within $\pm 1$ Level Agreement** | $\frac{\sum [\|s_A - s_B\| \le 1]}{N}$ | $> 95\%$ required | Reflects that adjacent ordinal categories (e.g., Level 1 vs 2) have minor boundary variance without fundamental disagreement. |
| **Unweighted Cohen's Kappa ($\kappa$)** | $\kappa = \frac{P_o - P_e}{1 - P_e}$ | $\kappa > 0.80$ Almost Perfect | Corrects for agreement occurring by chance across categorical buckets. |
| **Quadratic Weighted Kappa ($\kappa_w$)** | $\kappa_w = 1 - \frac{\sum w_{ij} O_{ij}}{\sum w_{ij} E_{ij}}$ | $\kappa_w > 0.80$ Excellent | Heavily penalizes extreme errors (e.g. Score 0 vs 5) while recognizing near-misses on ordinal scales. |
| **Pearson Correlation ($r$)** | $r = \frac{\text{Cov}(A, B)}{\sigma_A \sigma_B}$ | $r > 0.90$ Strong | Evaluates linear tracking between Coder A and Coder B continuous tendencies. |

---

## 🚀 Key Scripts & Documentation

| File | Purpose |
|---|---|
| [`CODING_GUIDELINES.md`](CODING_GUIDELINES.md) | Official instruction manual for human coders, outlining keyword cues and the Table 1 decision tree. |
| [`setup_two_coders.py`](setup_two_coders.py) | Generates visually formatted, blinded Excel coding sheets for Person A and Person B with soft-yellow input highlights. |
| [`analyze_coder_agreement.py`](analyze_coder_agreement.py) | Full diagnostic script calculating Kappa, Pearson $r$, Spearman $\rho$, confusion matrix, and isolated disagreements. |
| [`reconcile_coders.py`](reconcile_coders.py) | Automated consensus engine that merges completed sheets, flags discrepancies, and applies Table 1 adjudication. |

---

## 🛠️ Step-by-Step Execution Guide

### 1. Generate Blinded Coding Sheets
```bash
python 03_human_reliability/setup_two_coders.py
```
* Produces `coding_sheet_person_A.xlsx` (Forest Green theme) and `coding_sheet_person_B.xlsx` (Burgundy theme).

### 2. Instruct Human Coders
* Provide annotators with [`CODING_GUIDELINES.md`](CODING_GUIDELINES.md). Coders must score each passage from 0 to 5 without communicating.

### 3. Analyze Inter-Coder Agreement
```bash
python 03_human_reliability/analyze_coder_agreement.py \
    --sheet-a 06_pilot_50_suite/data/coding_sheet_person_A_50.xlsx \
    --sheet-b 06_pilot_50_suite/data/coding_sheet_person_B_50.xlsx
```
* **Expected Terminal Output:**
  ```text
  Exact Agreement:              98.0% (49 / 50)
  Within +/- 1 Level Agreement: 100.0% (50 / 50)
  Unweighted Cohen's Kappa:     0.9686
  Quadratic Weighted Kappa:     0.9896  <-- Outstanding Agreement
  Pearson Correlation (r):      0.9943
  ```

### 4. Reconcile Disagreements into Golden Truth
```bash
python 03_human_reliability/reconcile_coders.py \
    --sheet-a 06_pilot_50_suite/data/coding_sheet_person_A_50.xlsx \
    --sheet-b 06_pilot_50_suite/data/coding_sheet_person_B_50.xlsx \
    --out data/golden_truth_verified.xlsx
```
* Exports the final unified dataset for training and testing machine learning models.
