#!/usr/bin/env python3
"""
train_continuous_depth_model.py — Train and Cross-Validate the Continuous SDG Depth Model.

Academic Reference:
  Izhar, M. et al. (2026). Exploring firm-level SDG engagement in an emerging economy through content analysis.
  Section 3.4.3 & Table 1 (0 to 5 continuous & ordinal depth scale).
"""

import sys
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, cohen_kappa_score
from scipy.stats import pearsonr, spearmanr

# Local pipeline module
from continuous_model import ContinuousDepthPipeline, train_and_save_model, extract_linguistic_features

sys.stdout.reconfigure(encoding='utf-8')


def evaluate_cross_validation(df: pd.DataFrame, n_splits: int = 5):
    print("=" * 80)
    print(f"  RUNNING {n_splits}-FOLD CROSS-VALIDATION ON GROUND TRUTH")
    print("=" * 80)
    
    passages = df["passage"].fillna("").astype(str).tolist()
    y_true = df["izhar_table1_score (0-5)"].values.astype(float)
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(df))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(passages)):
        train_df = df.iloc[train_idx]
        val_passages = [passages[i] for i in val_idx]
        
        # Fit fold pipeline
        pipe = ContinuousDepthPipeline(embedder=None, tfidf=None, scaler=None, regressor=None)
        # Train fold
        from sentence_transformers import SentenceTransformer
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        from scipy.sparse import hstack
        
        emb = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        train_p = train_df["passage"].fillna("").astype(str).tolist()
        y_train = train_df["izhar_table1_score (0-5)"].values.astype(float)
        
        d_train = emb.encode(train_p, normalize_embeddings=True, show_progress_bar=False)
        d_val = emb.encode(val_passages, normalize_embeddings=True, show_progress_bar=False)
        
        l_train = extract_linguistic_features(train_p)
        l_val = extract_linguistic_features(val_passages)
        sc = StandardScaler()
        l_train_sc = sc.fit_transform(l_train)
        l_val_sc = sc.transform(l_val)
        
        tf = TfidfVectorizer(ngram_range=(1, 2), max_features=1000, sublinear_tf=True, stop_words="english", min_df=1)
        tf_train = tf.fit_transform(train_p)
        tf_val = tf.transform(val_passages)
        
        X_tr = hstack([d_train, l_train_sc, tf_train]).tocsr()
        X_v = hstack([d_val, l_val_sc, tf_val]).tocsr()
        
        reg = Ridge(alpha=1.5, random_state=42)
        reg.fit(X_tr, y_train)
        
        p_val = np.clip(reg.predict(X_v), 0.0, 5.0)
        oof_preds[val_idx] = p_val
        
    rmse = np.sqrt(mean_squared_error(y_true, oof_preds))
    mae = mean_absolute_error(y_true, oof_preds)
    r, p_val = pearsonr(y_true, oof_preds)
    rho, _ = spearmanr(y_true, oof_preds)
    
    oof_int = np.clip(np.round(oof_preds).astype(int), 0, 5)
    y_int = y_true.astype(int)
    exact_acc = (y_int == oof_int).mean() * 100
    within_1 = (np.abs(y_int - oof_int) <= 1).mean() * 100
    kw = cohen_kappa_score(y_int, oof_int, weights="quadratic")
    
    print(f"Out-of-Fold Cross-Validation Metrics (N={len(df)}):")
    print(f"  * Pearson Correlation (r):         {r:.4f} (p={p_val:.2e})")
    print(f"  * Spearman Rank Correlation (rho): {rho:.4f}")
    print(f"  * Root Mean Squared Error (RMSE):  +/- {rmse:.3f} points")
    print(f"  * Mean Absolute Error (MAE):       +/- {mae:.3f} points")
    print(f"  * Exact Level Match Accuracy:      {exact_acc:.1f}%")
    print(f"  * Within +/- 1 Level Agreement:    {within_1:.1f}%")
    print(f"  * Quadratic Weighted Kappa:        {kw:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Continuous Depth Regressor")
    parser.add_argument("--data", default="06_pilot_50_suite/data/golden_truth_50.xlsx", help="Golden truth Excel file")
    parser.add_argument("--out-model", default="04_depth_scoring_engine/model_depth_continuous.joblib", help="Output model path")
    parser.add_argument("--cv", action="store_true", help="Run 5-fold cross-validation")
    args = parser.parse_args()
    
    df_data = pd.read_excel(args.data)
    if args.cv:
        evaluate_cross_validation(df_data)
        
    train_and_save_model(Path(args.data), Path(args.out_model))
