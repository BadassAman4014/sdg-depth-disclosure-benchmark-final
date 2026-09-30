import pandas as pd
import numpy as np

df = pd.read_excel('data/golden_dataset_500.xlsx')
y_raw = df['human_consensus'].astype(str).str.strip().str.lower().values
print("Values in human_consensus:\n", pd.Series(y_raw).value_counts())
y_binary = np.array([1 if label == 'sub' else 0 for label in y_raw])
print("\ny_binary counts (1=sub, 0=sym):\n", pd.Series(y_binary).value_counts())
