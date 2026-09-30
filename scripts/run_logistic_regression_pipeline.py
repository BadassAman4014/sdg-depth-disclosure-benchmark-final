#!/usr/bin/env python3
"""
run_logistic_regression_pipeline.py — Comprehensive Logistic Regression & AI Reliability Pipeline.

Grounded in:
  1. De Kok (2025, Management Science) — Construct validity, error analysis, benchmark comparison.
  2. Shah et al. (2020) — Text classification with Logistic Regression, Random Forest, and KNN.

Features:
  - Validates Human Ground Truth (Coder 1, Coder 2, Consensus).
  - Computes Inter-Coder Reliability (Cohen's Kappa κ, % agreement).
  - Trains TF-IDF + Logistic Regression with 5-Fold Stratified Cross-Validation.
  - Benchmarks against KNN and Random Forest.
  - Compares with AI / ChatGPT classifications.
  - Extracts top predictive features (interpretability).
  - (Optional) Scales trained Logistic Regression to classify all 63,185 passages in the corpus.

Usage:
    # 1. Run evaluation & validation on your human coding sheet:
    python scripts/run_logistic_regression_pipeline.py --input data/golden_dataset_500.xlsx

    # 2. Scale predictions to ALL 63,185 passages across all 149 companies:
    python scripts/run_logistic_regression_pipeline.py --input data/golden_dataset_500.xlsx --predict-full
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

FULL_CORPUS_FILE = Path("data/ground_truth_full.xlsx")
OUT_FULL_CLASSIFIED = Path("data/ground_truth_full_classified.xlsx")


def load_ground_truth_data(in_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load dataset and separate labeled vs unlabeled rows."""
    df = pd.read_excel(in_path) if in_path.suffix == ".xlsx" else pd.read_csv(in_path)
    
    # Identify target ground truth column
    target_col = None
    for candidate in ["human_consensus", "ground_truth", "coder_1"]:
        if candidate in df.columns and df[candidate].notna().sum() > 0:
            target_col = candidate
            break

    if target_col is None:
        log.warning("No ground truth labels found yet! Please annotate 'coder_1' or 'human_consensus'.")
        valid_df = pd.DataFrame()
    else:
        # Filter valid rows (non-empty 'sym' or 'sub')
        valid_mask = df[target_col].astype(str).str.strip().str.lower().isin(["sym", "sub"])
        valid_df = df[valid_mask].copy()
        valid_df["target_label"] = valid_df[target_col].astype(str).str.strip().str.lower()
        log.info(f"Loaded {len(valid_df)} labeled samples using column '{target_col}'")

    return df, valid_df


def compute_inter_coder_agreement(df: pd.DataFrame):
    """Compute Cohen's Kappa if dual coding is present."""
    if "coder_1" in df.columns and "coder_2" in df.columns:
        valid = df[df["coder_1"].notna() & df["coder_2"].notna() & (df["coder_1"] != "") & (df["coder_2"] != "")]
        if len(valid) >= 5:
            c1 = valid["coder_1"].astype(str).str.strip().str.lower()
            c2 = valid["coder_2"].astype(str).str.strip().str.lower()
            kappa = cohen_kappa_score(c1, c2)
            pct = (c1 == c2).mean() * 100
            print("\n" + "=" * 78)
            print("  INTER-CODER RELIABILITY (Coder 1 vs. Coder 2)")
            print("=" * 78)
            print(f"  Dual-coded Sample Size: {len(valid)} items")
            print(f"  Cohen's Kappa (κ):      {kappa:.4f}  (>=0.80 = Strong Academic Agreement)")
            print(f"  Raw Agreement:          {pct:.2f}%\n")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute comprehensive classification performance metrics (De Kok 2025)."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label="sub", zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label="sub", zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label="sub", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=["sym", "sub"])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    type_1 = (fp / (fp + tn) * 100) if (fp + tn) > 0 else 0.0
    type_2 = (fn / (fn + tp) * 100) if (fn + tp) > 0 else 0.0

    return {
        "Accuracy (%)": round(acc * 100, 2),
        "Precision (sub) (%)": round(prec * 100, 2),
        "Recall (sub) (%)": round(rec * 100, 2),
        "F1 (sub) (%)": round(f1 * 100, 2),
        "Macro F1 (%)": round(macro_f1 * 100, 2),
        "Type I Error (%)": round(type_1, 2),
        "Type II Error (%)": round(type_2, 2),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)
    }


def train_and_evaluate_models(valid_df: pd.DataFrame) -> Tuple[Dict, TfidfVectorizer, LogisticRegression]:
    """Train Logistic Regression, KNN, Random Forest and evaluate AI."""
    y = valid_df["target_label"].values
    X_text = valid_df["passage"].fillna("").astype(str).values

    log.info(f"Extracting TF-IDF features for {len(X_text)} passages...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=3000,
        sublinear_tf=True,
        stop_words="english",
        min_df=2,
    )
    X = vectorizer.fit_transform(X_text)

    # 5-Fold Stratified Cross Validation
    n_splits = min(5, len(valid_df) // 2) if len(valid_df) >= 10 else 2
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # 1. Logistic Regression
    log_reg = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=42)
    y_pred_lr = cross_val_predict(log_reg, X, y, cv=cv)
    metrics_lr = compute_metrics(y, y_pred_lr)

    # 2. KNN (k=5)
    knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
    y_pred_knn = cross_val_predict(knn, X, y, cv=cv)
    metrics_knn = compute_metrics(y, y_pred_knn)

    # 3. Random Forest
    rf = RandomForestClassifier(n_estimators=150, max_depth=15, random_state=42)
    y_pred_rf = cross_val_predict(rf, X, y, cv=cv)
    metrics_rf = compute_metrics(y, y_pred_rf)

    results = {
        "Logistic Regression (5-Fold CV)": metrics_lr,
        "KNN (k=5) (5-Fold CV)": metrics_knn,
        "Random Forest (5-Fold CV)": metrics_rf,
    }

    # 4. Check if AI Labelling is present
    for ai_col in ["AI Labelling", "chatgpt_prediction"]:
        if ai_col in valid_df.columns:
            ai_preds = valid_df[ai_col].astype(str).str.strip().str.lower().values
            valid_ai_mask = np.isin(ai_preds, ["sym", "sub"])
            if valid_ai_mask.sum() >= 5:
                results[f"AI ({ai_col})"] = compute_metrics(y[valid_ai_mask], ai_preds[valid_ai_mask])
                break

    # Fit final Logistic Regression model on all training data
    log_reg.fit(X, y)
    return results, vectorizer, log_reg


def print_top_features(vectorizer: TfidfVectorizer, model: LogisticRegression, n: int = 15):
    """Print top diagnostic words/ngrams associated with Substantive vs Symbolic disclosures."""
    feature_names = np.array(vectorizer.get_feature_names_out())
    # Substantive class index (positive coefficients)
    classes = list(model.classes_)
    if "sub" in classes:
        sub_idx = classes.index("sub")
        coef = model.coef_[0] if len(model.coef_) == 1 else model.coef_[sub_idx]
        
        top_sub_idx = np.argsort(coef)[-n:][::-1]
        top_sym_idx = np.argsort(coef)[:n]

        print("\n" + "=" * 78)
        print("  TOP DIAGNOSTIC LINGUISTIC FEATURES (Logistic Regression Coefficients)")
        print("=" * 78)
        print("  Top Substantive Indicators (predict 'sub'):")
        for idx in top_sub_idx:
            print(f"    + {feature_names[idx]:<25} (coef: {coef[idx]:+.4f})")

        print("\n  Top Symbolic Indicators (predict 'sym'):")
        for idx in top_sym_idx:
            print(f"    - {feature_names[idx]:<25} (coef: {coef[idx]:+.4f})")
        print("=" * 78 + "\n")


def predict_full_corpus(vectorizer: TfidfVectorizer, model: LogisticRegression):
    """Scale trained Logistic Regression to classify all 63,185 passages across reports."""
    log.info(f"Loading full corpus from {FULL_CORPUS_FILE}...")
    df_full = pd.read_excel(FULL_CORPUS_FILE)

    log.info(f"Transforming {len(df_full):,} passages with TF-IDF...")
    X_full = vectorizer.transform(df_full["passage"].fillna("").astype(str).values)

    log.info("Generating Logistic Regression predictions & probabilities...")
    classes = list(model.classes_)
    sub_col_idx = classes.index("sub") if "sub" in classes else 1

    probs = model.predict_proba(X_full)[:, sub_col_idx]
    preds = model.predict(X_full)

    df_full["log_reg_prediction"] = preds
    df_full["log_reg_substantive_prob"] = np.round(probs, 4)

    log.info(f"Saving classified full corpus to {OUT_FULL_CLASSIFIED}...")
    df_full.to_excel(OUT_FULL_CLASSIFIED, index=False, engine="openpyxl")
    csv_out = OUT_FULL_CLASSIFIED.with_suffix(".csv")
    df_full.to_csv(csv_out, index=False, encoding="utf-8-sig")

    log.info(f"Successfully classified all {len(df_full):,} passages!")
    log.info(f"Distribution: {pd.Series(preds).value_counts().to_dict()}")


def main():
    parser = argparse.ArgumentParser(description="Logistic Regression & AI Reliability Evaluation")
    parser.add_argument("--input", default="data/golden_dataset_500.xlsx", help="Input golden dataset path")
    parser.add_argument("--predict-full", action="store_true", help="Apply trained Logistic Regression to all 63,185 passages")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        log.error(f"File not found: {in_path}")
        return

    # 1. Load data
    df_raw, valid_df = load_ground_truth_data(in_path)

    # 2. Inter-Coder Reliability
    compute_inter_coder_agreement(df_raw)

    if valid_df.empty or len(valid_df) < 5:
        print("\n" + "=" * 78)
        print("  [STATUS] Human Ground Truth Coding in Progress")
        print("=" * 78)
        print(f"  Current labeled rows: {len(valid_df)}")
        print("  Please open 'data/human_coding_sheet_500.xlsx' or 'data/golden_dataset_500.xlsx'")
        print("  and fill in 'coder_1', 'coder_2', or 'human_consensus' with 'sym' or 'sub'.")
        print("  Once you label any number of rows (e.g. 50, 100, or 500), re-run:")
        print(f"    python scripts/run_logistic_regression_pipeline.py --input {args.input}")
        print("=" * 78 + "\n")
        return

    # 3. Model Training & Cross-Validation
    results, vectorizer, log_reg = train_and_evaluate_models(valid_df)

    # 4. Display Results Table
    print("\n" + "=" * 78)
    print("  MODEL PERFORMANCE & AI RELIABILITY BENCHMARK (De Kok 2025; Shah et al. 2020)")
    print("=" * 78)
    table_rows = []
    for model_name, m in results.items():
        table_rows.append({
            "Model": model_name,
            "Accuracy": f"{m['Accuracy (%)']}%",
            "Precision": f"{m['Precision (sub) (%)']}%",
            "Recall": f"{m['Recall (sub) (%)']}%",
            "F1-Score": f"{m['F1 (sub) (%)']}%",
            "Macro F1": f"{m['Macro F1 (%)']}%",
            "Type I Err": f"{m['Type I Error (%)']}%",
            "Type II Err": f"{m['Type II Error (%)']}%",
        })
    res_df = pd.DataFrame(table_rows)
    print(res_df.to_string(index=False))
    print("=" * 78)

    # 5. Top Predictive Features
    print_top_features(vectorizer, log_reg)

    # 6. Optional Full-Corpus Inference
    if args.predict_full:
        predict_full_corpus(vectorizer, log_reg)


if __name__ == "__main__":
    main()
