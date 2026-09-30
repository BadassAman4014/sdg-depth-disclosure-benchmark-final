#!/usr/bin/env python3
"""
continuous_model.py — Standalone Continuous SDG Depth Regressor Pipeline Module.
"""

import re
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import hstack, csr_matrix
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

AUDIT_REGEX = re.compile(r"\b(?:verified|assured|audited|certified)\s+by\b|\b(?:pwc|kpmg|ey|deloitte|tüv|tuev|bureau\s+veritas|iso\s*14001|iso\s*50001)\b", re.I)
METRIC_REGEX = re.compile(r"\b\d+(?:[\.,]\d+)?\s*(?:%|percent|prozent|tco2|tco2e|tonnes?|kwh|mwh|gwh|million|billion|mio|mrd|eur|€|\$)\b", re.I)
ACTION_REGEX = re.compile(r"\b(?:implemented|installed|deployed|commissioned|invested|allocated|achieved|constructed|built|umgesetzt|eingeführt|investiert|reduziert|trained)\b", re.I)
ASPIRATIONAL_REGEX = re.compile(r"\b(?:strive|aim|aspire|commit|intend|pledge|seek|streben|beabsichtigen|vision|code\s+of\s+conduct|compliance\s+policy)\b", re.I)
QUAL_OUTCOME_REGEX = re.compile(r"\b(?:resulted in|led to|improved|enhanced|strengthened|achieved progress|positive impact|verbessert|gestärkt)\b", re.I)


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
        return np.clip(np.round(preds, 2), 0.0, 5.0)


def train_and_save_model(golden_path: Path, output_model_path: Path):
    df = pd.read_excel(golden_path)
    passages = df["passage"].fillna("").astype(str).tolist()
    y_continuous = df["continuous_depth_score (0.0-5.0)"].values

    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    dense_embs = embedder.encode(passages, normalize_embeddings=True, show_progress_bar=False)

    ling_feats = extract_linguistic_features(passages)
    scaler = StandardScaler()
    ling_scaled = scaler.fit_transform(ling_feats)

    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=1500, sublinear_tf=True, stop_words="english", min_df=1)
    tfidf_feats = tfidf.fit_transform(passages)

    X_dense_ling = np.hstack([dense_embs, ling_scaled])
    X_hybrid = hstack([X_dense_ling, tfidf_feats]).tocsr()

    regressor = Ridge(alpha=1.5, random_state=42)
    regressor.fit(X_hybrid, y_continuous)

    pipeline = ContinuousDepthPipeline(
        embedder=embedder,
        tfidf=tfidf,
        scaler=scaler,
        regressor=regressor
    )

    output_model_path.parent.mkdir(exist_ok=True, parents=True)
    joblib.dump(pipeline, output_model_path)
    print(f"Successfully serialized model to: {output_model_path}")
    return pipeline


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    gt = base / "data" / "golden_truth_50.xlsx"
    out = base / "model_depth_50.joblib"
    train_and_save_model(gt, out)
