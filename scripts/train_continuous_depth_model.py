#!/usr/bin/env python3
"""
train_continuous_depth_model.py — Continuous SDG Depth Regressor (0.0 to 5.0 Scale).

Implements:
  1. Continuous Score Extraction grounded in Izhar et al. (2026) Table 1 / Hummel (2019) / PwC (2018).
  2. Hybrid Feature Extraction (Dense Multilingual Embeddings + Theory Linguistic Evidence).
  3. Continuous ElasticNet / Ridge Regressor predicting exact real-valued depth S in [0.0, 5.0].
  4. 5-Fold Cross-Validation evaluation (R², RMSE, MAE).
  5. Serializes model to 'models/continuous_depth_regressor_0_to_5.joblib'.
  6. Exports 'data/golden_dataset_500_continuous_0_to_5.xlsx'.
"""

import logging
import re
import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import hstack, csr_matrix
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding='utf-8')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

GOLDEN_500 = Path("data/golden_dataset_500.xlsx")
MODEL_OUT = Path("models/continuous_depth_regressor_0_to_5.joblib")
OUT_EXCEL = Path("data/golden_dataset_500_continuous_0_to_5.xlsx")

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

AUDIT_REGEX = re.compile(r"\b(?:verified|assured|audited|certified)\s+by\b|\b(?:pwc|kpmg|ey|deloitte|tüv|tuev|bureau\s+veritas|iso\s*14001|iso\s*50001)\b", re.I)
METRIC_REGEX = re.compile(r"\b\d+(?:[\.,]\d+)?\s*(?:%|percent|prozent|tco2|tco2e|tonnes?|kwh|mwh|gwh|million|billion|mio|mrd|eur|€|\$)\b", re.I)
ACTION_REGEX = re.compile(r"\b(?:implemented|installed|deployed|commissioned|invested|allocated|achieved|constructed|built|umgesetzt|eingeführt|investiert|reduziert|trained)\b", re.I)
ASPIRATIONAL_REGEX = re.compile(r"\b(?:strive|aim|aspire|commit|intend|pledge|seek|streben|beabsichtigen|vision|code\s+of\s+conduct|compliance\s+policy)\b", re.I)
QUAL_OUTCOME_REGEX = re.compile(r"\b(?:resulted in|led to|improved|enhanced|strengthened|achieved progress|positive impact|verbessert|gestärkt)\b", re.I)


def compute_grounded_depth_score(passage: str, consensus_label: str) -> float:
    """Computes academic ground truth continuous depth score (0.0 to 5.0) per Table 1."""
    p_str = str(passage)
    
    has_audit = bool(AUDIT_REGEX.search(p_str))
    metric_count = len(METRIC_REGEX.findall(p_str))
    action_count = len(ACTION_REGEX.findall(p_str))
    has_asp = bool(ASPIRATIONAL_REGEX.search(p_str))
    has_qual_outcome = bool(QUAL_OUTCOME_REGEX.search(p_str))
    
    # Base score
    if consensus_label == "sym":
        if has_asp:
            score = 1.0 + min(0.5, action_count * 0.1)
        else:
            score = 0.0 + min(0.8, metric_count * 0.1)
    else:  # sub
        base = 2.0
        # Target/Actions present
        if action_count >= 1:
            base += 0.5
        # Qualitative outcome
        if has_qual_outcome:
            base += 0.5
        # Quantitative outcome
        if metric_count >= 1:
            base += min(1.2, metric_count * 0.3)
        # Audit assurance
        if has_audit:
            base += 0.8
        score = min(5.0, base)
        
    return round(float(score), 2)


def extract_linguistic_features(passages: list[str]) -> np.ndarray:
    feats = []
    for text in passages:
        p_str = str(text)
        word_count = len(p_str.split())
        
        has_audit = float(bool(AUDIT_REGEX.search(p_str)))
        metric_count = float(len(METRIC_REGEX.findall(p_str)))
        action_count = float(len(ACTION_REGEX.findall(p_str)))
        asp_count = float(len(ASPIRATIONAL_REGEX.findall(p_str)))
        qual_outcome = float(bool(QUAL_OUTCOME_REGEX.search(p_str)))
        
        evidence_density = (metric_count * 2.0 + action_count * 1.5 + has_audit * 3.0) / (word_count + 1)
        
        feats.append([
            has_audit,
            metric_count,
            action_count,
            asp_count,
            qual_outcome,
            evidence_density,
            float(word_count),
        ])
    return np.array(feats, dtype=np.float32)


class ContinuousDepthPipeline:
    """Production Pipeline for predicting continuous SDG Depth in [0.0, 5.0]."""
    
    def __init__(self, embedder, tfidf, scaler, regressor):
        self.embedder = embedder
        self.tfidf = tfidf
        self.scaler = scaler
        self.regressor = regressor

    def transform_features(self, passages: list[str]) -> csr_matrix:
        dense_embs = self.embedder.encode(passages, normalize_embeddings=True, show_progress_bar=False)
        ling_feats = extract_linguistic_features(passages)
        ling_scaled = self.scaler.transform(ling_feats)
        tfidf_feats = self.tfidf.transform(passages)
        
        X_dense_ling = np.hstack([dense_embs, ling_scaled])
        X_hybrid = hstack([X_dense_ling, tfidf_feats]).tocsr()
        return X_hybrid

    def predict(self, passages: list[str]) -> np.ndarray:
        X = self.transform_features(passages)
        preds = self.regressor.predict(X)
        # Clip strictly between 0.0 and 5.0
        return np.clip(np.round(preds, 2), 0.0, 5.0)


def main():
    log.info(f"Loading dataset from {GOLDEN_500}...")
    df = pd.read_excel(GOLDEN_500)
    passages = df["passage"].fillna("").astype(str).tolist()
    consensus = df["human_consensus"].astype(str).str.strip().str.lower().tolist()

    log.info("Deriving Ground Truth Continuous Depth Scores (0.0 to 5.0) per Table 1...")
    y_continuous = np.array([compute_grounded_depth_score(p, c) for p, c in zip(passages, consensus)])
    df["ground_truth_depth_0_to_5"] = y_continuous

    log.info("Extracting Multilingual Sentence Embeddings...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    dense_embs = embedder.encode(passages, normalize_embeddings=True, show_progress_bar=False)

    log.info("Extracting Linguistic Evidence & Scaling...")
    ling_feats = extract_linguistic_features(passages)
    scaler = StandardScaler()
    ling_scaled = scaler.fit_transform(ling_feats)

    log.info("Fitting TF-IDF Vectorizer...")
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=3000, sublinear_tf=True, stop_words="english", min_df=2)
    tfidf_feats = tfidf.fit_transform(passages)

    X_dense_ling = np.hstack([dense_embs, ling_scaled])
    X_hybrid = hstack([X_dense_ling, tfidf_feats]).tocsr()

    log.info("Evaluating 5-Fold Cross-Validation on Continuous Depth Regressor...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    regressor = Ridge(alpha=2.0, random_state=42)
    
    y_cv_preds = cross_val_predict(regressor, X_hybrid, y_continuous, cv=kf)
    y_cv_preds = np.clip(y_cv_preds, 0.0, 5.0)

    # Metrics
    r2 = r2_score(y_continuous, y_cv_preds)
    rmse = np.sqrt(mean_squared_error(y_continuous, y_cv_preds))
    mae = mean_absolute_error(y_continuous, y_cv_preds)

    # Accuracy when discretized to integer levels (0, 1, 2, 3, 4, 5)
    int_true = np.round(y_continuous).astype(int)
    int_pred = np.round(y_cv_preds).astype(int)
    exact_match_acc = (int_true == int_pred).mean() * 100
    within_one_acc = (np.abs(int_true - int_pred) <= 1).mean() * 100

    print("\n" + "=" * 80)
    print("  CONTINUOUS SDG DEPTH REGRESSOR BENCHMARK (0.0 to 5.0 Scale)")
    print("=" * 80)
    print(f"  Observations (N):            {len(df):,} corporate passages")
    print(f"  R² Score (Variance Exp.):    {r2:.4f} ({r2*100:.1f}%)")
    print(f"  RMSE (Root Mean Sq. Error):  ±{rmse:.3f} points (on 0 to 5 scale)")
    print(f"  MAE (Mean Absolute Error):   ±{mae:.3f} points")
    print(f"  Exact Integer Match:         {exact_match_acc:.2f}%")
    print(f"  Within ±1 Point Match:       {within_one_acc:.2f}%")
    print("=" * 80)

    # Train final model on all 500 samples
    regressor.fit(X_hybrid, y_continuous)
    pipeline = ContinuousDepthPipeline(
        embedder=embedder,
        tfidf=tfidf,
        scaler=scaler,
        regressor=regressor
    )

    MODEL_OUT.parent.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_OUT)
    log.info(f"Saved Continuous Pipeline to {MODEL_OUT}")

    # Add predicted continuous depth score to dataframe
    df["predicted_depth_0_to_5"] = np.round(pipeline.predict(passages), 2)
    df.to_excel(OUT_EXCEL, index=False)
    log.info(f"Saved enriched continuous dataset to {OUT_EXCEL}")

    # Display 5 sample predictions
    print("\n  Sample Continuous Predictions:")
    print("  " + "-" * 75)
    for i in [0, 1, 2, 3, 4]:
        p = df.loc[i, "passage"][:80].replace("\n", " ")
        true_s = df.loc[i, "ground_truth_depth_0_to_5"]
        pred_s = df.loc[i, "predicted_depth_0_to_5"]
        print(f"  Row {i+1:2d} | True: {true_s:.2f} | Pred: {pred_s:.2f} | Text: \"{p}...\"")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
