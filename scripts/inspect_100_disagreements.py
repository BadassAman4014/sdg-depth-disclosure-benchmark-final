import sys
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_excel('data/golden_dataset_500.xlsx')
df100 = df.iloc[:100].copy()

c1 = df100['coder_1'].astype(str).str.strip().str.lower()
c2 = df100['coder_2'].astype(str).str.strip().str.lower()
ai = df100['AI Labelling'].astype(str).str.strip().str.lower()

disagreements = df100[c1 != c2].copy()
print(f"Total disagreements in first 100: {len(disagreements)}")
print("\nFirst 15 disagreements:")
for idx, row in disagreements.head(15).iterrows():
    p = str(row['passage'])[:140].replace('\n', ' ').encode('ascii', 'replace').decode('ascii')
    print(f"Row {row['sample_id']:2d} | C1: {row['coder_1']} | C2: {row['coder_2']} | AI: {row['AI Labelling']} | Text: {p}...")
