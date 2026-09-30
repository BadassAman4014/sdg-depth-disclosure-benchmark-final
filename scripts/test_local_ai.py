import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.metrics.pairwise import cosine_similarity
import re

# 1. Load the 100 ground truth samples as our reference anchor bank
df_anchor = pd.read_excel('Updated Ground Truths - 100 samples for evaluating models.xlsx')
y_true = df_anchor['ground_truth'].str.strip().str.lower().values
passages = df_anchor['passage'].tolist()

print(f"Loaded {len(df_anchor)} reference samples (sym: {(y_true=='sym').sum()}, sub: {(y_true=='sub').sum()})")

# 2. Load sentence transformer
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
embeddings = model.encode(passages, normalize_embeddings=True, show_progress_bar=False)

# 3. Test leave-one-out / nearest prototype classification
preds = []
for i in range(len(passages)):
    # Mask current item
    mask = np.ones(len(passages), dtype=bool)
    mask[i] = False
    
    emb_curr = embeddings[i].reshape(1, -1)
    emb_others = embeddings[mask]
    y_others = y_true[mask]
    
    # Compute similarity to all other samples
    sims = cosine_similarity(emb_curr, emb_others)[0]
    
    # Top 7 nearest neighbors weighted vote
    top_k_idx = np.argsort(sims)[-7:]
    top_sims = sims[top_k_idx]
    top_labels = y_others[top_k_idx]
    
    sub_score = sum(w for w, l in zip(top_sims, top_labels) if l == 'sub')
    sym_score = sum(w for w, l in zip(top_sims, top_labels) if l == 'sym')
    
    # Linguistic indicators (quantified KPIs, percentages, audit verification)
    p_lower = passages[i].lower()
    has_metrics = bool(re.search(r'\b\d+(\.\d+)?\s*(%|percent|tonnes?|tco2|kwh|mwh|gwh|€|\$|million|billion)\b', p_lower))
    has_audit = bool(re.search(r'\b(verified by|assured by|pwc|kpmg|ey|deloitte|tüv|iso\s*\d+|bureau veritas)\b', p_lower))
    
    if has_audit:
        sub_score += 0.35
    if has_metrics:
        sub_score += 0.15
        
    pred = 'sub' if sub_score > sym_score else 'sym'
    preds.append(pred)

acc = accuracy_score(y_true, preds)
f1 = f1_score(y_true, preds, pos_label='sub')
print(f"\nLocal AI Validation Accuracy: {acc*100:.2f}% | F1 (Substantive): {f1*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_true, preds, digits=3))
