#!/usr/bin/env python3
"""
evaluate_reliability.py — Statistical Reliability and Machine Learning Benchmarking Suite.

Implements the evaluation methodologies from:
  1. De Kok (2025) - Management Science (Inter-Coder Reliability, Cohen's Kappa, Type I/II Errors, F1)
  2. Shah et al. (2020) - Comparative Analysis of Logistic Regression, Random Forest, and KNN

Features:
  - Computes Inter-Coder Reliability (Cohen's Kappa κ, % Agreement) between Coder 1 and Coder 2
  - Trains TF-IDF + Logistic Regression, KNN (k-Nearest Neighbors), and Random Forest classifiers
  - Runs 5-Fold Stratified Cross-Validation on the human ground truth
  - Evaluates AI (ChatGPT) predictions vs. Human Consensus vs. ML benchmarks
  - Computes Accuracy, Precision, Recall, Macro/Weighted F1, Type I & Type II Error Rates, Confusion Matrices

Usage:
    python scripts/evaluate_reliability.py --input data/golden_dataset_500.xlsx
"""

import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def compute_inter_coder_reliability(df: pd.DataFrame) -> dict:
    """Compute Cohen's Kappa and percentage agreement between Coder 1 and Coder 2."""
    valid = df[df["coder_1"].notna() & df["coder_2"].notna() & (df["coder_1"] != "") & (df["coder_2"] != "")]
    if len(valid) == 0:
        return {"status": "no_dual_coding_data", "count": 0}

    c1 = valid["coder_1"].str.strip().str.lower()
    c2 = valid["coder_2"].str.strip().str.lower()

    kappa = cohen_kappa_score(c1, c2)
    pct_agreement = (c1 == c2).mean() * 100

    return {
        "status": "ok",
        "sample_size": len(valid),
        "cohens_kappa": round(kappa, 4),
        "pct_agreement": round(pct_agreement, 2),
    }


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, pos_label: str = "sub") -> dict:
    """Compute comprehensive classification metrics per De Kok (2025)."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred, labels=["sym", "sub"])
    # cm: [[TN (sym->sym), FP (sym->sub)], [FN (sub->sym), TP (sub->sub)]]
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    
    # Type I error (FP rate) = FP / (FP + TN)
    type_1 = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    # Type II error (FN rate) = FN / (FN + TP)
    type_2 = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "accuracy": round(acc * 100, 2),
        "precision": round(prec * 100, 2),
        "recall": round(rec * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "macro_f1": round(macro_f1 * 100, 2),
        "type_1_error": round(type_1 * 100, 2),
        "type_2_error": round(type_2 * 100, 2),
        "confusion_matrix": {
            "TN (sym as sym)": int(tn),
            "FP (sym as sub)": int(fp),
            "FN (sub as sym)": int(fn),
            "TP (sub as sub)": int(tp),
        },
    }


def evaluate_models(df: pd.DataFrame):
    """Train and evaluate Logistic Regression, KNN, and Random Forest benchmark models via 5-Fold CV."""
    # Filter rows with ground truth
    valid = df[df["human_consensus"].notna() & (df["human_consensus"] != "")].copy()
    if len(valid) < 10:
        log.warning("Fewer than 10 labeled rows in human_consensus. Please ensure ground truth labels ('sym'/'sub') are filled.")
        return None

    y = valid["human_consensus"].str.strip().str.lower().values
    X_text = valid["passage"].values

    log.info(f"Vectorizing {len(X_text)} passages using TF-IDF (1-2 ngrams)...")
    vectorizer = TfidfVectorizer(max_features=2500, ngram_range=(1, 2), stop_words="english")
    X = vectorizer.fit_transform(X_text)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 1. Logistic Regression (Shah et al. 2020)
    log_reg = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    y_pred_lr = cross_val_predict(log_reg, X, y, cv=cv)
    metrics_lr = compute_classification_metrics(y, y_pred_lr)

    # 2. KNN (k-Nearest Neighbors) (Shah et al. 2020)
    knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
    y_pred_knn = cross_val_predict(knn, X, y, cv=cv)
    metrics_knn = compute_classification_metrics(y, y_pred_knn)

    # 3. Random Forest (Shah et al. 2020)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    y_pred_rf = cross_val_predict(rf, X, y, cv=cv)
    metrics_rf = compute_classification_metrics(y, y_pred_rf)

    results = {
        "Logistic Regression": metrics_lr,
        "KNN (k=5)": metrics_knn,
        "Random Forest": metrics_rf,
    }

    # 4. Check if ChatGPT predictions are present
    if "chatgpt_prediction" in valid.columns and valid["chatgpt_prediction"].notna().sum() > 0:
        gpt_valid = valid[valid["chatgpt_prediction"].notna() & (valid["chatgpt_prediction"] != "")]
        y_true_gpt = gpt_valid["human_consensus"].str.strip().str.lower().values
        y_pred_gpt = gpt_valid["chatgpt_prediction"].str.strip().str.lower().values
        results["ChatGPT (AI)"] = compute_classification_metrics(y_true_gpt, y_pred_gpt)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate AI and ML reliability on the Golden Dataset")
    parser.add_argument("--input", default="data/golden_dataset_500.xlsx", help="Path to golden dataset file (.xlsx or .csv)")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        log.error(f"File not found: {in_path}")
        return

    df = pd.read_excel(in_path) if in_path.suffix == ".xlsx" else pd.read_csv(in_path)
    log.info(f"Loaded {len(df)} rows from {in_path}")

    # 1. Inter-Coder Reliability
    print("\n" + "=" * 75)
    print("  1. INTER-CODER RELIABILITY (Human Coder 1 vs. Human Coder 2)")
    print("=" * 75)
    icr = compute_inter_coder_reliability(df)
    if icr["status"] == "ok":
        print(f"  Sample Size:        {icr['sample_size']} items")
        print(f"  Cohen's Kappa (κ):  {icr['cohens_kappa']}  (>=0.80 = Excellent reliability)")
        print(f"  Raw Agreement:      {icr['pct_agreement']}%")
    else:
        print("  [INFO] Coder 1 and Coder 2 columns are currently blank for human labeling.")
        print("         Once annotated, running this script will compute Cohen's Kappa automatically.")

    # 2. ML Benchmarking & AI Reliability
    print("\n" + "=" * 75)
    print("  2. MACHINE LEARNING & AI CLASSIFICATION RELIABILITY COMPARISON")
    print("=" * 75)
    models_res = evaluate_models(df)
    if models_res:
        summary_rows = []
        for model_name, m in models_res.items():
            summary_rows.append({
                "Model": model_name,
                "Accuracy (%)": m["accuracy"],
                "Precision (%)": m["precision"],
                "Recall (%)": m["recall"],
                "F1-Score (%)": m["f1_score"],
                "Macro F1 (%)": m["macro_f1"],
                "Type I Error (%)": m["type_1_error"],
                "Type II Error (%)": m["type_2_error"],
            })
        summary_df = pd.DataFrame(summary_rows)
        print(summary_df.to_string(index=False))
        print("\n" + "=" * 75)


if __name__ == "__main__":
    main()
