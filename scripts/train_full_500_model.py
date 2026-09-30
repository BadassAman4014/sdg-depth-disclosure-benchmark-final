#!/usr/bin/env python3
"""
train_full_500_model.py — Train and evaluate Logistic Regression & ML models on all 500 samples.

Options:
  1. Train on Coder 1 (Person A, N=500)
  2. Train on Coder 2 (Person B, N=500)
  3. Train on Full 500 Consensus (Agreed + Tie-broken Disagreements)

Saves trained model & vectorizer to 'models/' for full-corpus deployment.
"""

import argparse
import joblib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DATASET_500 = Path("data/golden_dataset_500.xlsx")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def compute_metrics(y_true, y_pred):
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


def prepare_full_500_dataset(df: pd.DataFrame, target_source: str = "consensus_resolved") -> pd.DataFrame:
    df_clean = df.copy()
    c1 = df_clean["coder_1"].astype(str).str.strip().str.lower()
    c2 = df_clean["coder_2"].astype(str).str.strip().str.lower()
    ai = df_clean["AI Labelling"].astype(str).str.strip().str.lower()

    if target_source == "coder_1":
        df_clean["train_label"] = c1
    elif target_source == "coder_2":
        df_clean["train_label"] = c2
    else:  # consensus_resolved
        # Where A == B: use agreed label. Where A != B: resolve by majority vote with AI evidence
        resolved = []
        for a, b, a_ai in zip(c1, c2, ai):
            if a in ["sym", "sub"] and a == b:
                resolved.append(a)
            elif a in ["sym", "sub"] and b in ["sym", "sub"]:
                # Tie-break with AI labelling (majority 2 out of 3)
                if a == a_ai:
                    resolved.append(a)
                elif b == a_ai:
                    resolved.append(b)
                else:
                    resolved.append(a)
            elif a in ["sym", "sub"]:
                resolved.append(a)
            elif b in ["sym", "sub"]:
                resolved.append(b)
            else:
                resolved.append("sym")
        df_clean["train_label"] = resolved

    # Update human_consensus in sheet
    df_clean["human_consensus"] = df_clean["train_label"]
    df_clean.to_excel(DATASET_500, index=False, engine="openpyxl")
    df_clean.to_csv(DATASET_500.with_suffix(".csv"), index=False, encoding="utf-8-sig")

    return df_clean


def main():
    parser = argparse.ArgumentParser(description="Train Logistic Regression on all 500 samples")
    parser.add_argument("--source", default="consensus_resolved", choices=["consensus_resolved", "coder_1", "coder_2"], help="Target ground truth source")
    args = parser.parse_args()

    log.info(f"Loading {DATASET_500}...")
    df = pd.read_excel(DATASET_500)
    df_train = prepare_full_500_dataset(df, target_source=args.source)

    y = df_train["train_label"].values
    X_text = df_train["passage"].fillna("").astype(str).values

    print("\n" + "=" * 80)
    print(f"  TRAINING FULL 500-SAMPLE MACHINE LEARNING MODELS (Source: {args.source})")
    print("=" * 80)
    print(f"  Total Training Observations (N):  {len(y)}")
    print(f"  Class Distribution:               'sym' = {(y == 'sym').sum()}, 'sub' = {(y == 'sub').sum()}")
    print("=" * 80)

    # 1. Feature Extraction: TF-IDF unigrams and bigrams
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=4000,
        sublinear_tf=True,
        stop_words="english",
        min_df=2,
    )
    X = vectorizer.fit_transform(X_text)

    # 2. 5-Fold Stratified Cross-Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # A. Logistic Regression
    lr = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=42)
    y_pred_lr = cross_val_predict(lr, X, y, cv=cv)
    m_lr = compute_metrics(y, y_pred_lr)

    # B. Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
    y_pred_rf = cross_val_predict(rf, X, y, cv=cv)
    m_rf = compute_metrics(y, y_pred_rf)

    # C. KNN (k=5)
    knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
    y_pred_knn = cross_val_predict(knn, X, y, cv=cv)
    m_knn = compute_metrics(y, y_pred_knn)

    # D. AI Baseline
    ai_preds = df_train["AI Labelling"].astype(str).str.strip().str.lower().values
    m_ai = compute_metrics(y, ai_preds)

    # Display Benchmark Table
    results = {
        "Logistic Regression (5-Fold CV)": m_lr,
        "Random Forest (5-Fold CV)": m_rf,
        "KNN (k=5) (5-Fold CV)": m_knn,
        "AI Baseline (AI Labelling)": m_ai,
    }

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
    print("\n" + res_df.to_string(index=False))
    print("=" * 80)

    # 3. Train Final Model on 100% of the 500 samples
    log.info("Fitting final Logistic Regression model on all 500 samples...")
    lr.fit(X, y)

    # 4. Save artifacts
    model_path = MODEL_DIR / "logistic_regression_500.joblib"
    vec_path = MODEL_DIR / "tfidf_vectorizer_500.joblib"
    joblib.dump(lr, model_path)
    joblib.dump(vectorizer, vec_path)

    log.info(f"\nSaved trained model and vectorizer to:")
    log.info(f"  - {model_path}")
    log.info(f"  - {vec_path}")

    # Top Features
    feature_names = np.array(vectorizer.get_feature_names_out())
    classes = list(lr.classes_)
    sub_idx = classes.index("sub") if "sub" in classes else 0
    coef = lr.coef_[0] if len(lr.coef_) == 1 else lr.coef_[sub_idx]

    top_sub = np.argsort(coef)[-12:][::-1]
    top_sym = np.argsort(coef)[:12]

    print("\n" + "=" * 80)
    print("  TOP DIAGNOSTIC PREDICTORS (Learned by Logistic Regression on 500 samples)")
    print("=" * 80)
    print("  Substantive (+):")
    for i in top_sub:
        print(f"    + {feature_names[i]:<25} (coef: {coef[i]:+.4f})")
    print("\n  Symbolic (-):")
    for i in top_sym:
        print(f"    - {feature_names[i]:<25} (coef: {coef[i]:+.4f})")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
