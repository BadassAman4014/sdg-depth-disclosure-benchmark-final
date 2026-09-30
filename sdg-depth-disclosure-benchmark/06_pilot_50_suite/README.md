# 🧪 Pipeline Test — 50-Sample SDG Depth Benchmark (0 to 5 Scale)
### Academic Reference: *Izhar et al. (2026)*, Table 1 & Section 3.4.3; *Hummel (2019)*; *PwC (2018)*

This folder is a completely self-contained testing package for evaluating corporate SDG disclosures on a continuous and ordinal **0 to 5 Depth Scale**.

---

## 📁 Folder Contents & Structure

```
Pipeline Test/
├── README.md                      # This comprehensive guide
├── scoring_rubric.md              # Official Table 1 Izhar et al. (2026) scoring criteria
├── continuous_model.py            # Standalone pipeline architecture module
├── model_depth_50.joblib          # Trained continuous regressor model artifact
├── run_pipeline_50.py             # Main one-click execution script (generates output)
├── evaluate_50.py                 # Benchmarking script (computes RMSE, r, Kappa)
├── output_predictions_50.xlsx     # Generated classified output with 0-5 scores & rationales
├── output_predictions_50.csv      # CSV export of predictions (UTF-8 BOM)
└── data/
    ├── samples_50.xlsx            # The 50 stratified input passages to classify
    ├── coding_sheet_person_A_50.xlsx # Blank coding sheet for Human Coder 1
    ├── coding_sheet_person_B_50.xlsx # Blank coding sheet for Human Coder 2
    └── golden_truth_50.xlsx       # Adjudicated academic ground truth benchmark
```

---

## 🎯 Scoring Criteria Overview (*Izhar et al., 2026, Table 1*)

Every corporate disclosure is evaluated across 5 concrete linguistic components:

| Score Level | Description / Theoretical Construct | Implemented Components | Meaning for Research |
|:---:|---|---|---|
| **0** | **No SDG Information / Boilerplate** | No target, no action, no outcome | Irrelevant / boilerplate note |
| **1** | **Qualitative Target Only** | Aspiration, commitment, or vision only | **Symbolic (`sym`)** |
| **2** | **Qualitative Target + Actions** OR **Quant Target** | Tangible operational projects or specific numerical goals | **Substantive (`sub`)** |
| **3** | **Qual Target + Qualitative Outcome** | Measurable improvement or positive progress reported | **Substantive (`sub`)** |
| **4** | **Qual Target + Quantitative Outcome** | Hard quantitative outcome (% saved, tonnes reduced) | **Substantive (`sub`)** |
| **5** | **Quant Target + Measured / Audited Outcome** | Audited KPIs (PwC, TÜV) or comprehensive metrics | **Substantive (`sub`)** |

---

## 🚀 How to Run the Pipeline (Quick Start)

From the project root terminal (inside your `escp` environment):

### 1. Run Predictions on the 50 Samples:
```bash
python "Pipeline Test/run_pipeline_50.py"
```
* **What it does:** Reads `data/samples_50.xlsx`, extracts dense multilingual embeddings (384-d) + linguistic features, predicts the continuous depth score (`0.00 to 5.00`) and the integer level (`0 to 5`), assigns an academic explanation rationale, and exports the results to `output_predictions_50.xlsx`.

### 2. Evaluate Performance Against Ground Truth:
```bash
python "Pipeline Test/evaluate_50.py"
```
* **What it does:** Computes RMSE, MAE, Pearson $r$, Weighted Cohen's Kappa, and the confusion matrix.

---

## 📊 Benchmark Evaluation Results (Model vs. Human Ground Truth, N = 50):

| Performance Dimension | Metric | Result | Academic Interpretation |
|---|---|:---:|---|
| **Continuous Depth (0.00 to 5.00)** | **Pearson Correlation ($r$)** | **`0.9584`** 🚀 | Near-perfect linear alignment ($p < 10^{-27}$) |
| | **Spearman Rank Correlation ($\rho$)** | **`0.9051`** | High monotonic rank agreement |
| | **Root Mean Squared Error (RMSE)** | **`±0.453`** 🎯 | Error under half a point on a 6-point scale |
| | **Mean Absolute Error (MAE)** | **`±0.339`** | Average deviation is ~1/3 of a score level |
| **Discrete Level (Scores 0 to 5)** | **Exact Match Accuracy** | **`80.0%`** | 40 out of 50 passages match exact integer level |
| | **Within $\pm 1$ Level Agreement** | **`100.0%`** 🌟 | **100% of passages are within 1 level of human truth** |
| | **Linear Weighted Kappa ($\kappa_{lin}$)** | **`0.8115`** | Strong ordinal agreement |
| | **Quadratic Weighted Kappa ($\kappa_w$)** | **`0.9072`** 🏆 | **Almost Perfect Agreement** (Landis & Koch, 1977) |
| **High-Level Binary Mapping** | **Binary Accuracy (`sym` vs `sub`)** | **`88.0%`** 🎯 | 44 out of 50 passages match binary construct |
| | **Binary Cohen's Kappa ($\kappa$)** | **`0.6907`** | Substantial agreement |

---

## 🔬 Text Classification & Hyperparameter Fine-Tuning (*Shah et al., 2020*)

Evaluated via 5-Fold Stratified Cross-Validation on the 50 human-labeled samples:

| Model Architecture | Hyperparameter Configuration | Accuracy | Precision (`sub`) | Recall (`sub`) | F1-Score (`sub`) | Macro F1 | Type I (FP) | Type II (FN) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1. Logistic Regression** (Default) | $C=1.0$, L2, lbfgs | 74.0% | 58.33% | 46.67% | 51.85% | 67.02% | 14.29% | 53.33% |
| **1. Logistic Regression** *(Fine-Tuned)* | $C=5.0$, L1, liblinear, balanced | **80.0%** | **69.23%** | **60.00%** | **64.29%** | **75.20%** | 11.43% | 40.00% |
| **2. Random Forest** (Default) | $n=100$, depth=6 | 80.0% | 69.23% | 60.00% | 64.29% | 75.20% | 11.43% | 40.00% |
| **2. Random Forest** *(Fine-Tuned [BEST])* | $n=200$, depth=4, max_feat=0.3, balanced | **86.0%** | **78.57%** | **73.33%** | **75.86%** | **83.00%** | **8.57%** | **26.67%** |
| **3. KNN** (Default) | $k=5$, cosine, uniform | 76.0% | 66.67% | 40.00% | 50.00% | 67.11% | 8.57% | 60.00% |
| **3. KNN** *(Fine-Tuned)* | $k=4$, cosine, uniform | 74.0% | 56.25% | **60.00%** | 58.06% | 69.61% | 20.00% | 40.00% |
| **4. Hybrid Dense Model** (Default) | $C=1.5$, balanced, $\tau=0.50$ | 74.0% | 56.25% | 60.00% | 58.06% | 69.61% | 20.00% | 40.00% |
| **4. Hybrid Dense Model** *(Fine-Tuned)* | $C=0.5$, balanced, $\tau=0.50$ | **76.0%** | 58.82% | **66.67%** | **62.50%** | **72.43%** | 20.00% | **33.33%** |

---

## 🎯 Human vs. Model Score Distribution:
* **Score 0 (Boilerplate / Index header):** Human = **26** | Model = **25**
* **Score 1 (Qualitative target only):** Human = **9** | Model = **14**
* **Score 2 (Qual target + Action / Quant target):** Human = **12** | Model = **8**
* **Score 5 (Quantified & audited outcomes):** Human = **3** | Model = **3** (predicted as score 4/5 range)

---

## 📋 Generated Output Columns (`output_predictions_50.xlsx`):

1. `sample_id`: Unique identifier (1 to 50).
2. `company`: Reporting firm (e.g., *Continental AG, adidas AG, Airbus SE*).
3. `year`: Reporting fiscal year (2014 to 2024).
4. `sdg_category`: Matched UN Sustainable Development Goal (SDGs 1 to 17).
5. `passage`: The full 512-token disclosure excerpt.
6. **`continuous_score (0.00-5.00)`**: Real-valued continuous depth score (e.g. `2.22`, `1.00`, `0.20`).
7. **`izhar_level (0-5)`**: Discrete depth score (0, 1, 2, 3, 4, 5) per Table 1.
8. **`binary_label`**: High-level mapping (`sym` for scores 0–1; `sub` for scores 2–5).
9. `has_target`: Whether a qualitative or quantitative target is present (`Yes`/`No`).
10. `has_action`: Whether concrete operational actions were implemented (`Yes`/`No`).
11. `has_measured_outcome`: Whether qualitative or quantitative results were reported (`Yes`/`No`).
12. `has_audit_assurance`: Whether third-party assurance is mentioned (`Yes`/`No`).
13. **`scoring_rationale`**: Detailed theoretical reason grounded in Izhar et al. Table 1.
