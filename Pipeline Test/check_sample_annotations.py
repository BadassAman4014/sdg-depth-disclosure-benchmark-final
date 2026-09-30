import pandas as pd

df = pd.read_excel('Pipeline Test/data/coding_sheet_person_A_50.xlsx')
print("Sample rows with human annotations:")
for i in range(10):
    row = df.iloc[i]
    print(f"Row {row['sample_id']:2d} | Score: {row['person_A_score_0_to_5']} | Target: {row['has_target (Yes/No)']} | Action: {row['has_action (Yes/No)']} | Outcome: {row['has_measured_outcome (Yes/No)']} | Notes: {str(row['notes_and_evidence'])[:40]}...")
