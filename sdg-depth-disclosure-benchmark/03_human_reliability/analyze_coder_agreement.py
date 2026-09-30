import sys
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix, accuracy_score
from scipy.stats import pearsonr, spearmanr

sys.stdout.reconfigure(encoding='utf-8')

df_a = pd.read_excel('Pipeline Test/data/coding_sheet_person_A_50.xlsx')
df_b = pd.read_excel('Pipeline Test/data/coding_sheet_person_B_50.xlsx')

score_a = df_a['person_A_score_0_to_5']
score_b = df_b['person_B_score_0_to_5']

print(f"Total samples: {len(df_a)}")
print(f"Person A non-null count: {score_a.notna().sum()}")
print(f"Person B non-null count: {score_b.notna().sum()}")

# Clean values
y_a = score_a.astype(int).values
y_b = score_b.astype(int).values

exact_acc = (y_a == y_b).mean() * 100
within_1 = (np.abs(y_a - y_b) <= 1).mean() * 100

kappa_unweighted = cohen_kappa_score(y_a, y_b)
kappa_linear = cohen_kappa_score(y_a, y_b, weights='linear')
kappa_quad = cohen_kappa_score(y_a, y_b, weights='quadratic')

pearson_r, p_val = pearsonr(y_a, y_b)
spearman_rho, _ = spearmanr(y_a, y_b)

print("\n" + "=" * 75)
print("  INTER-CODER RELIABILITY (Person A vs. Person B on 50 Samples)")
print("=" * 75)
print(f"Exact Agreement:              {exact_acc:.1f}% ({int(exact_acc*len(y_a)/100)} / {len(y_a)})")
print(f"Within +/- 1 Level Agreement: {within_1:.1f}% ({int(within_1*len(y_a)/100)} / {len(y_a)})")
print(f"Unweighted Cohen's Kappa:     {kappa_unweighted:.4f}")
print(f"Linear Weighted Kappa:        {kappa_linear:.4f}")
print(f"Quadratic Weighted Kappa:     {kappa_quad:.4f}")
print(f"Pearson Correlation (r):      {pearson_r:.4f}")
print(f"Spearman Correlation (rho):   {spearman_rho:.4f}")

# Binary agreement (sym = 0, 1; sub = 2, 3, 4, 5)
bin_a = np.array(['sub' if s >= 2 else 'sym' for s in y_a])
bin_b = np.array(['sub' if s >= 2 else 'sym' for s in y_b])
bin_acc = (bin_a == bin_b).mean() * 100
bin_kappa = cohen_kappa_score(bin_a, bin_b)

print(f"\nBinary Mapping Agreement (sym: 0-1 vs sub: 2-5):")
print(f"Binary Agreement:             {bin_acc:.1f}%")
print(f"Binary Cohen's Kappa:         {bin_kappa:.4f}")

print("\nScore distributions:")
print("Person A:", pd.Series(y_a).value_counts().sort_index().to_dict())
print("Person B:", pd.Series(y_b).value_counts().sort_index().to_dict())

# Disagreements
disagreements = df_a[y_a != y_b].copy()
print(f"\nTotal Disagreements: {len(disagreements)}")
if len(disagreements) > 0:
    print("Disagreements detail:")
    for idx, row in disagreements.iterrows():
        print(f"Sample ID {row['sample_id']:2d} | Person A: {y_a[idx]} | Person B: {y_b[idx]} | Keyword: {row['keyword']} | Text: {str(row['passage'])[:80]}...")
