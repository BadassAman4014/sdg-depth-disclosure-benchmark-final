import pandas as pd
import re

def clean_str(s):
    if not isinstance(s, str):
        return ""
    s = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]", "", s)
    return " ".join(s.strip().split())

df_100 = pd.read_excel('Updated Ground Truths - 100 samples for evaluating models.xlsx')
df_full = pd.read_excel('data/ground_truth_full.xlsx')
df_500 = pd.read_excel('data/golden_dataset_500.xlsx')

df_100['p_norm'] = df_100['passage'].apply(clean_str)
df_full['p_norm'] = df_full['passage'].apply(clean_str)
df_500['p_norm'] = df_500['passage'].apply(clean_str)

# 1. Check exact matches with full corpus
exact_full_matches = df_100[df_100['p_norm'].isin(df_full['p_norm'])]
# 2. Check prefix matches (first 80 chars)
df_100['p_prefix'] = df_100['p_norm'].str[:80]
df_full['p_prefix'] = df_full['p_norm'].str[:80]
prefix_full_matches = df_100[df_100['p_prefix'].isin(df_full['p_prefix'])]

print(f"Total rows in 100-samples template: {len(df_100)}")
print(f"Exact passage matches in 63,185 full corpus: {len(exact_full_matches)}")
print(f"Prefix passage matches in 63,185 full corpus: {len(prefix_full_matches)}")
