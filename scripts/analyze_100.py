import sys
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_excel('data/golden_dataset_500.xlsx')
df100 = df.iloc[:100].copy()

# Look at distribution of C1, C2, AI in first 100:
print("First 100 Distribution:")
print("Coder 1:", df100['coder_1'].value_counts().to_dict())
print("Coder 2:", df100['coder_2'].value_counts().to_dict())
print("AI Labelling:", df100['AI Labelling'].value_counts().to_dict())

# Agreement between C1 and AI:
agree_c1_ai = (df100['coder_1'] == df100['AI Labelling']).mean() * 100
agree_c2_ai = (df100['coder_2'] == df100['AI Labelling']).mean() * 100
print(f"\nAgreement Coder 1 vs AI: {agree_c1_ai:.1f}%")
print(f"Agreement Coder 2 vs AI: {agree_c2_ai:.1f}%")
