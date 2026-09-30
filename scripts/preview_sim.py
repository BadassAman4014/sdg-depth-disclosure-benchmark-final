import pandas as pd

df = pd.read_excel('data/izhar_simulation_50_samples.xlsx')
print("Columns:", df.columns.tolist())
print("\nFirst 5 samples:")
for i, r in df.head(5).iterrows():
    print(f"\n[Sample {r['sample_id']}] Company: {r['company']} ({r['year']}) | SDG: {r['sdg_category']} | Keyword: '{r['keyword']}'")
    print(f"Passage: {r['passage'][:140]}...")
    print(f"Coder 1 Score: {r['coder1_izhar_score']} ({r['coder1_label']}) - Rationale: {r['coder1_notes']}")
    print(f"Coder 2 Score: {r['coder2_izhar_score']} ({r['coder2_label']}) - Rationale: {r['coder2_notes']}")
    print(f"Consensus Score: {r['consensus_izhar_score']} ({r['human_consensus_label']}) | AI Labelling: {r['AI Labelling']}")
