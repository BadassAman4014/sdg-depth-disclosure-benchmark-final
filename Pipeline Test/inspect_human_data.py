import pandas as pd

df_a = pd.read_excel('Pipeline Test/data/coding_sheet_person_A_50.xlsx')
df_b = pd.read_excel('Pipeline Test/data/coding_sheet_person_B_50.xlsx')

print("Coder A columns:", df_a.columns.tolist())
print("Coder A shape:", df_a.shape)
print("\nCoder A sample scores:")
print(df_a.iloc[:10, [0, 1, 5, 6] if len(df_a.columns) > 6 else list(range(len(df_a.columns)))])

print("\nCoder B columns:", df_b.columns.tolist())
print("Coder B shape:", df_b.shape)
print("\nCoder B sample scores:")
print(df_b.iloc[:10, [0, 1, 5, 6] if len(df_b.columns) > 6 else list(range(len(df_b.columns)))])
