import pandas as pd

df_a = pd.read_excel('Pipeline Test/data/coding_sheet_person_A_50.xlsx')
df_b = pd.read_excel('Pipeline Test/data/coding_sheet_person_B_50.xlsx')
df_single = pd.read_excel('Pipeline Test/data/coding_sheet_50_samples.xlsx')

print("=== Coder A ===")
print("Columns:", df_a.columns.tolist())
col_a = [c for c in df_a.columns if 'score' in c.lower()][0]
print(df_a[['sample_id', col_a]].head(15))

print("\n=== Coder B ===")
print("Columns:", df_b.columns.tolist())
col_b = [c for c in df_b.columns if 'score' in c.lower()][0]
print(df_b[['sample_id', col_b]].head(15))

print("\n=== Single Coding Sheet ===")
print("Columns:", df_single.columns.tolist())
col_s = [c for c in df_single.columns if 'score' in c.lower()][0]
print(df_single[['sample_id', col_s]].head(15))

# Compare A vs B
s_a = df_a[col_a].dropna()
s_b = df_b[col_b].dropna()

print(f"\nNon-null in A: {len(s_a)}, Non-null in B: {len(s_b)}")
if len(s_a) == len(s_b):
    diff = (s_a != s_b).sum()
    print(f"Number of differences between A and B: {diff}")
    if diff > 0:
        for i in range(len(s_a)):
            if s_a.iloc[i] != s_b.iloc[i]:
                print(f"Row {df_a.loc[i, 'sample_id']} | Person A: {s_a.iloc[i]} | Person B: {s_b.iloc[i]} | Text: {str(df_a.loc[i, 'passage'])[:60]}...")
