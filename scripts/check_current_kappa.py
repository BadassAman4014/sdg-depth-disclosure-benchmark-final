import pandas as pd
from sklearn.metrics import cohen_kappa_score

df = pd.read_excel('data/golden_dataset_500.xlsx')

c1 = df['coder_1'].astype(str).str.strip().str.lower()
c2 = df['coder_2'].astype(str).str.strip().str.lower()

# 1. First 100 rows (The Overlap)
mask_100 = (df.index < 100) & c1.isin(['sym', 'sub']) & c2.isin(['sym', 'sub'])
c1_100 = c1[mask_100]
c2_100 = c2[mask_100]

kappa_100 = cohen_kappa_score(c1_100, c2_100)
agree_100 = (c1_100 == c2_100).mean() * 100
disagree_100 = (c1_100 != c2_100).sum()

# 2. All 500 rows
mask_500 = c1.isin(['sym', 'sub']) & c2.isin(['sym', 'sub'])
c1_500 = c1[mask_500]
c2_500 = c2[mask_500]

kappa_500 = cohen_kappa_score(c1_500, c2_500)
agree_500 = (c1_500 == c2_500).mean() * 100
disagree_500 = (c1_500 != c2_500).sum()

print("=" * 75)
print("  CURRENT SHEET INTER-CODER RELIABILITY & COHEN'S KAPPA")
print("=" * 75)
print(f"1. First 100 Samples (Overlap):")
print(f"   - Valid Dual-Coded:       {len(c1_100)} / 100")
print(f"   - Raw Agreement:          {agree_100:.2f}% ({len(c1_100) - disagree_100} agreed, {disagree_100} disagreed)")
print(f"   - Cohen's Kappa (κ):      {kappa_100:.4f}")

print(f"\n2. Full 500 Samples:")
print(f"   - Valid Dual-Coded:       {len(c1_500)} / 500")
print(f"   - Raw Agreement:          {agree_500:.2f}% ({len(c1_500) - disagree_500} agreed, {disagree_500} disagreed)")
print(f"   - Cohen's Kappa (κ):      {kappa_500:.4f}")
print("=" * 75)
