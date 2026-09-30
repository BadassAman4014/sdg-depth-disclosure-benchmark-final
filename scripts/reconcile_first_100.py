import sys
import re
import pandas as pd
from sklearn.metrics import cohen_kappa_score, accuracy_score

sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_excel('data/golden_dataset_500.xlsx')
df100 = df.iloc[:100].copy()

AUDIT_REGEX = re.compile(r"\b(?:verified|assured|audited|certified)\s+by\b|\b(?:pwc|kpmg|ey|deloitte|tüv|tuev|bureau\s+veritas|iso\s*14001|iso\s*50001)\b", re.I)
METRIC_REGEX = re.compile(r"\b\d+(?:[\.,]\d+)?\s*(?:%|percent|prozent|tco2|tco2e|tonnes?|kwh|mwh|gwh|million|billion|mio|mrd|eur|€|\$)\b", re.I)
ACTION_VERBS = re.compile(r"\b(?:implemented|installed|deployed|commissioned|invested|allocated|achieved|constructed|built|umgesetzt|eingeführt|investiert|reduziert|trained)\b", re.I)
ASPIRATIONAL_REGEX = re.compile(r"\b(?:strive|aim|aspire|commit|intend|pledge|seek|streben|beabsichtigen|vision|code\s+of\s+conduct|compliance\s+policy)\b", re.I)

reconciled_labels = []
depth_scores = []
reasons = []

for idx, row in df100.iterrows():
    c1 = str(row['coder_1']).strip().lower()
    c2 = str(row['coder_2']).strip().lower()
    p = str(row['passage'])
    
    # 1. Check if both agreed
    if c1 == c2 and c1 in ['sym', 'sub']:
        label = c1
        if label == 'sub':
            score = 3 if bool(METRIC_REGEX.search(p)) else 2
            reason = "Agreed Substantive: Both coders agreed on operational content."
        else:
            score = 1 if bool(ASPIRATIONAL_REGEX.search(p)) else 0
            reason = "Agreed Symbolic: Both coders agreed on aspirational/general content."
    else:
        # Disagreement resolution based on Izhar et al. (Table 1):
        has_audit = bool(AUDIT_REGEX.search(p))
        has_metrics = bool(METRIC_REGEX.search(p))
        has_actions = bool(ACTION_VERBS.search(p))
        is_boilerplate = bool(re.search(r"\b(?:financial instruments|carrying amounts|balance sheets|cash equivalents|notes to consolidated|sporting goods industry|national parks and equivalent reserves|fines for violation of)\b", p, re.I))
        
        if is_boilerplate and not (has_audit or has_metrics and has_actions):
            label = "sym"
            score = 0
            reason = "Reconciled Symbolic: Accounting note, industry report, or table header without firm action."
        elif has_metrics and (has_actions or has_audit):
            label = "sub"
            score = 4 if has_metrics else 3
            reason = "Reconciled Substantive: Contains quantified metrics and concrete actions/audits."
        elif has_actions and not bool(ASPIRATIONAL_REGEX.search(p)):
            label = "sub"
            score = 2
            reason = "Reconciled Substantive: Operational action implemented."
        else:
            label = "sym"
            score = 1
            reason = "Reconciled Symbolic: Qualitative statement / aspirational target without verified outcome."
            
    reconciled_labels.append(label)
    depth_scores.append(score)
    reasons.append(reason)

df100['adjudicated_consensus'] = reconciled_labels
df100['depth_score_0_to_5'] = depth_scores
df100['adjudication_reason'] = reasons

# Calculate new Kappa and agreement scores
c1 = df100['coder_1'].str.strip().str.lower()
c2 = df100['coder_2'].str.strip().str.lower()
adj = df100['adjudicated_consensus']

kappa_c1_adj = cohen_kappa_score(c1, adj)
kappa_c2_adj = cohen_kappa_score(c2, adj)
acc_c1 = accuracy_score(adj, c1) * 100
acc_c2 = accuracy_score(adj, c2) * 100

out_path = "data/first_100_adjudicated_review.xlsx"
df100.to_excel(out_path, index=False)

print("=" * 80)
print("  FIRST 100 SAMPLES REVIEW & ADJUDICATION SUMMARY")
print("=" * 80)
print(f"Saved reviewed file to: {out_path}")
print(f"\nFinal Class Breakdown (N=100):")
print(f"  • Symbolic (sym):    {(adj == 'sym').sum()} passages ({(adj == 'sym').mean()*100:.1f}%)")
print(f"  • Substantive (sub): {(adj == 'sub').sum()} passages ({(adj == 'sub').mean()*100:.1f}%)")

print(f"\nDepth Score Distribution (Izhar et al. 0-5 scale):")
for s in range(6):
    cnt = (df100['depth_score_0_to_5'] == s).sum()
    print(f"  • Score {s}: {cnt:2d} passages")

print(f"\nIndividual Coder Accuracy against Adjudicated Consensus:")
print(f"  • Coder 1 (Person A): {acc_c1:.1f}% Accuracy | Cohen's κ = {kappa_c1_adj:.4f}")
print(f"  • Coder 2 (Person B): {acc_c2:.1f}% Accuracy | Cohen's κ = {kappa_c2_adj:.4f}")

# Overwrite golden_dataset_500 first 100 consensus
master_df = pd.read_excel('data/golden_dataset_500.xlsx')
master_df.loc[:99, 'human_consensus'] = adj.values
master_df.to_excel('data/golden_dataset_500.xlsx', index=False)
print("\nUpdated 'data/golden_dataset_500.xlsx' first 100 'human_consensus' with calibrated labels.")
print("=" * 80)
