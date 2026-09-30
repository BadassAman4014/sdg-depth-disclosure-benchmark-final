import os
import json
import pandas as pd

def audit_reports(reports_dir="reports"):
    manifest_path = os.path.join(reports_dir, "reports_manifest.csv")
    companies_file = "data/companies.json"
    
    if not os.path.exists(companies_file):
        print("Error: data/companies.json not found. Run scripts/extract_companies.py first.")
        return
        
    with open(companies_file, "r", encoding="utf-8") as f:
        companies = json.load(f)
        
    total_companies = len(companies)
    
    if not os.path.exists(manifest_path):
        print(f"Manifest not found at {manifest_path}. No downloads recorded yet.")
        return
        
    df = pd.read_csv(manifest_path)
    
    # Filter successful downloads
    df_success = df[df["status"].str.contains("SUCCESS", na=False)]
    
    print("=" * 80)
    print("ANNUAL REPORTS ARCHIVE AUDIT")
    print("=" * 80)
    print(f"Total Companies in Scope: {total_companies}")
    print(f"Total Download Attempts in Manifest: {len(df)}")
    print(f"Successfully Downloaded Reports: {len(df_success)}")
    
    total_size_mb = df_success["file_size_mb"].sum()
    print(f"Total Disk Usage: {total_size_mb:.2f} MB ({total_size_mb/1024:.2f} GB)")
    
    # Yearly breakdown
    print("\n--- Download Count by Fiscal Year ---")
    year_summary = df_success.groupby("year")["company_id"].count().reset_index()
    year_summary.columns = ["Fiscal Year", "Reports Acquired"]
    print(year_summary.to_string(index=False))
    
    # Company breakdown
    print("\n--- Top Companies by Reports Acquired ---")
    comp_summary = df_success.groupby("company_name")["year"].count().reset_index()
    comp_summary.columns = ["Company Name", "Reports Count"]
    comp_summary.sort_values(by="Reports Count", ascending=False, inplace=True)
    print(comp_summary.head(15).to_string(index=False))
    
    # Export full matrix to CSV
    pivot = df_success.pivot_table(index="company_name", columns="year", values="status", aggfunc="count", fill_value=0)
    matrix_path = os.path.join(reports_dir, "coverage_matrix.csv")
    pivot.to_csv(matrix_path)
    print(f"\nFull company coverage matrix saved to: {matrix_path}")
    print("=" * 80)

if __name__ == "__main__":
    audit_reports()
