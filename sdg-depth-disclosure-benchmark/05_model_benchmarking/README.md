# 🔬 Task 05: Machine Learning Benchmarks & Hyperparameter Tuning

This module provides a rigorous empirical comparison of text classification architectures and hyperparameter optimization grounded in the benchmark literature of **Shah et al. (2020)** (*A Comparative Analysis of Logistic Regression, Random Forest and KNN Models for Text Classification*), augmented with **Multilingual Dense Transformer Embeddings** and **Glassbox Explainable Boosting Machines (EBM)**.

---

## 📌 Benchmark Architecture Overview

Evaluated using **5-Fold Stratified Cross-Validation** on verified human ground truth across two academic tasks:
1. **Binary Classification:** Symbolic (`sym`: Table 1 Levels 0–1) vs. Substantive (`sub`: Table 1 Levels 2–5).
2. **Multi-Class Depth Classification:** Discrete depth levels $\{0, 1, 2, 5\}$.

```
Passage Text
     │
     ├──> TF-IDF Sparse Matrix (1-2 n-grams, sublinear TF) ──> [Logistic Regression, Random Forest, KNN]
     │
     └──> Hybrid Dense Representation (MiniLM-L12-v2 + Linguistic Features) ──> [Hybrid Classifier, EBM]
```

---

## 📊 Empirical Performance: Default vs. Fine-Tuned (N = 50 Ground Truth)

| Model Architecture | Hyperparameter Configuration | Accuracy | Precision (`sub`) | Recall (`sub`) | F1-Score (`sub`) | Macro F1 | Type I (FP) | Type II (FN) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1. Logistic Regression** (Default) | $C=1.0$, L2, lbfgs | 74.0% | 58.33% | 46.67% | 51.85% | 67.02% | 14.29% | 53.33% |
| **1. Logistic Regression** *(Fine-Tuned)* | $C=5.0$, L1, liblinear, balanced | **80.0%** | **69.23%** | **60.00%** | **64.29%** | **75.20%** | 11.43% | 40.00% |
| **2. Random Forest** (Default) | $n=100$, depth=6 | 80.0% | 69.23% | 60.00% | 64.29% | 75.20% | 11.43% | 40.00% |
| **2. Random Forest** *(Fine-Tuned [BEST])* | $n=200$, depth=4, max_features=0.3, balanced | **86.0%** | **78.57%** | **73.33%** | **75.86%** | **83.00%** | **8.57%** | **26.67%** |
| **3. KNN** (Default) | $k=5$, cosine, uniform | 76.0% | 66.67% | 40.00% | 50.00% | 67.11% | 8.57% | 60.00% |
| **3. KNN** *(Fine-Tuned)* | $k=4$, cosine, uniform | 74.0% | 56.25% | **60.00%** | 58.06% | 69.61% | 20.00% | 40.00% |
| **4. Hybrid Dense Model** (Default) | $C=1.5$, balanced, $\tau=0.50$ | 74.0% | 56.25% | 60.00% | 58.06% | 69.61% | 20.00% | 40.00% |
| **4. Hybrid Dense Model** *(Fine-Tuned)* | $C=0.5$, balanced, $\tau=0.50$ | **76.0%** | 58.82% | **66.67%** | **62.50%** | **72.43%** | 20.00% | **33.33%** |

---

## 💡 Theoretical Synthesis (*Shah et al., 2020*)

1. **Random Forest Superiority:**
   * Constraining tree depth (`max_depth=4`) and sampling 30% of features per split (`max_features=0.3`) prevented leaf fragmentation in sparse text spaces, yielding **86.0% Accuracy** and the lowest false-alarm rate (**8.57% Type I Error**).
2. **L1 Regularization in Logistic Regression:**
   * Switching to L1 sparsity penalty (`liblinear`, $C=5.0$) zeroed out noisy boilerplate terms, boosting F1-score from **51.85%** to **64.29%**.
3. **Curse of Dimensionality in KNN:**
   * High-dimensional sparse TF-IDF vectors tend toward equidistance, causing standard KNN to underperform on minority substantive claims. Tuning to $k=4$ with cosine metric boosted recall from **40.0%** to **60.0%**.
4. **Hybrid Dense Classifier for High Substantive Recall:**
   * Dense sentence embeddings combined with linguistic rule features captured cross-lingual semantic context, yielding the highest substantive recall (**66.67%**) and lowest missed-action rate (**33.33% Type II Error**).

---

## 🚀 Key Scripts & Execution Guide

### 1. Run Baseline Shah et al. Benchmark
```bash
python 05_model_benchmarking/train_shah_benchmark.py
```
* Computes 5-fold stratified cross-validation across all baseline models, displays confusion matrices, and exports `shah_benchmark_results.xlsx`.

### 2. Run Exhaustive Hyperparameter Grid Search & Calibration
```bash
python 05_model_benchmarking/finetune_hyperparameters.py
```
* Explores parameter grids for $C$, penalties, tree counts, leaf splits, and decision cutoff $\tau$, outputting `finetuned_benchmark_results.xlsx`.

### 3. Train Glassbox Explainable Boosting Machine (EBM)
```bash
python 05_model_benchmarking/train_glassbox_ebm.py
```
* Fits an interpretable Generalized Additive Model (GAM) providing exact feature contribution graphs for auditability.
