import pandas as pd
from sklearn.metrics import confusion_matrix

df = pd.read_excel('data/golden_dataset_500.xlsx')
c1 = df['coder_1'].astype(str).str.strip().str.lower()
c2 = df['coder_2'].astype(str).str.strip().str.lower()

cm = confusion_matrix(c1, c2, labels=['sym', 'sub'])
print("Confusion Matrix (Rows=Coder 1, Cols=Coder 2):")
print(f"               Coder 2: sym    Coder 2: sub")
print(f"Coder 1: sym      {cm[0,0]:<15} {cm[0,1]:<15}")
print(f"Coder 1: sub      {cm[1,0]:<15} {cm[1,1]:<15}")

print("\nValue counts:")
print("Coder 1 (Person A):\n", c1.value_counts())
print("\nCoder 2 (Person B):\n", c2.value_counts())
