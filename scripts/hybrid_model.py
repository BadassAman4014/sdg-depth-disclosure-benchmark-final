#!/usr/bin/env python3
"""
hybrid_model.py — Production Hybrid Feature Classifier (Embeddings + Theory Features + TF-IDF).
"""

import re
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import hstack, csr_matrix
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

AUDIT_REGEX = re.compile(r"\b(?:verified|assured|audited|certified)\s+by\b|\b(?:pwc|kpmg|ey|deloitte|tüv|tuev|bureau\s+veritas|iso\s*14001|iso\s*50001)\b", re.I)
METRIC_REGEX = re.compile(r"\b\d+(?:[\.,]\d+)?\s*(?:%|percent|prozent|tco2|tco2e|tonnes?|kwh|mwh|gwh|million|billion|mio|mrd|eur|€|\$)\b", re.I)
ACTION_REGEX = re.compile(r"\b(?:implemented|installed|deployed|commissioned|invested|allocated|achieved|constructed|built|umgesetzt|eingeführt|investiert|reduziert)\b", re.I)
ASPIRATIONAL_REGEX = re.compile(r"\b(?:strive|aim|aspire|commit|intend|pledge|seek|streben|beabsichtigen|vision|code\s+of\s+conduct|compliance\s+policy)\b", re.I)


def extract_linguistic_features(passages: list[str]) -> np.ndarray:
    feats = []
    for text in passages:
        p_str = str(text)
        word_count = len(p_str.split())
        
        has_audit = float(bool(AUDIT_REGEX.search(p_str)))
        metric_count = float(len(METRIC_REGEX.findall(p_str)))
        action_count = float(len(ACTION_REGEX.findall(p_str)))
        asp_count = float(len(ASPIRATIONAL_REGEX.findall(p_str)))
        
        evidence_ratio = (metric_count * 2.0 + action_count * 1.5 + has_audit * 3.0) / (word_count + 1)
        
        feats.append([
            has_audit,
            metric_count,
            action_count,
            asp_count,
            evidence_ratio,
            float(word_count),
        ])
    return np.array(feats, dtype=np.float32)


class HybridClassifierPipeline:
    def __init__(self, embedder, tfidf, scaler, clf, threshold: float = 0.40):
        self.embedder = embedder
        self.tfidf = tfidf
        self.scaler = scaler
        self.clf = clf
        self.threshold = threshold

    def transform_features(self, passages: list[str]) -> csr_matrix:
        dense_embs = self.embedder.encode(passages, normalize_embeddings=True, show_progress_bar=False)
        ling_feats = extract_linguistic_features(passages)
        ling_scaled = self.scaler.transform(ling_feats)
        tfidf_feats = self.tfidf.transform(passages)
        
        X_dense_ling = np.hstack([dense_embs, ling_scaled])
        X_hybrid = hstack([X_dense_ling, tfidf_feats]).tocsr()
        return X_hybrid

    def predict_proba(self, passages: list[str]) -> np.ndarray:
        X = self.transform_features(passages)
        # Class 1 is substantive ('sub')
        probs = self.clf.predict_proba(X)[:, 1]
        return probs

    def predict(self, passages: list[str]) -> list[str]:
        probs = self.predict_proba(passages)
        preds = ["sub" if p >= self.threshold else "sym" for p in probs]
        return preds


def train_and_save_pipeline(dataset_path: str = "data/golden_dataset_500.xlsx", model_out_path: str = "models/hybrid_feature_classifier.joblib"):
    df = pd.read_excel(dataset_path)
    
    y_raw = df["human_consensus"].astype(str).str.strip().str.lower().values
    y_binary = np.array([1 if label == "sub" else 0 for label in y_raw])
    passages = df["passage"].fillna("").astype(str).tolist()

    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    dense_embs = embedder.encode(passages, normalize_embeddings=True, show_progress_bar=False)

    ling_feats = extract_linguistic_features(passages)
    scaler = StandardScaler()
    ling_scaled = scaler.fit_transform(ling_feats)

    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=3000, sublinear_tf=True, stop_words="english", min_df=2)
    tfidf_feats = tfidf.fit_transform(passages)

    X_dense_ling = np.hstack([dense_embs, ling_scaled])
    X_hybrid = hstack([X_dense_ling, tfidf_feats]).tocsr()

    clf = LogisticRegression(C=1.5, max_iter=1000, class_weight="balanced", random_state=42)
    clf.fit(X_hybrid, y_binary)

    pipeline = HybridClassifierPipeline(
        embedder=embedder,
        tfidf=tfidf,
        scaler=scaler,
        clf=clf,
        threshold=0.40,
    )

    Path(model_out_path).parent.mkdir(exist_ok=True)
    joblib.dump(pipeline, model_out_path)
    print(f"Successfully saved HybridClassifierPipeline to {model_out_path}")
    return pipeline


if __name__ == "__main__":
    train_and_save_pipeline()
