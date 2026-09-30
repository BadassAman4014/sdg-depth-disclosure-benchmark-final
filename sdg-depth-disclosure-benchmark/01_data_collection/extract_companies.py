import os
import re
import pandas as pd
import json

def sanitize_folder_name(name):
    """Sanitize company name for clean filesystem folder creation."""
    # Replace special chars like & with 'and', remove forbidden filesystem chars
    clean = str(name).strip()
    clean = clean.replace('&', 'and')
    clean = clean.replace('/', ' ')
    clean = clean.replace('\\', ' ')
    clean = re.sub(r'[<>:"|?*]', '', clean)
    clean = re.sub(r'\s+', '_', clean)
    clean = re.sub(r'_+', '_', clean).strip('_')
    return clean

def extract_companies():
    excel_path = "GRI score.xlsx"
    df = pd.read_excel(excel_path, header=None)
    
    # Header row is index 0: Identifier (RIC), Company Name, ISIN
    # Row index 1: FY-10 ... FY-1
    # Data starts at row index 3
    companies = []
    
    for idx in range(3, len(df)):
        row = df.iloc[idx]
        ric = row[0] if pd.notna(row[0]) else None
        company_name = str(row[1]).strip() if pd.notna(row[1]) else None
        isin = str(row[2]).strip() if pd.notna(row[2]) else None
        
        if not company_name or company_name == 'nan':
            continue
            
        folder_name = sanitize_folder_name(company_name)
        
        companies.append({
            "id": len(companies) + 1,
            "ric": str(ric).strip() if ric else None,
            "company_name": company_name,
            "isin": isin,
            "folder_name": folder_name
        })
        
    os.makedirs("data", exist_ok=True)
    
    # Save as JSON and CSV
    with open("data/companies.json", "w", encoding="utf-8") as f:
        json.dump(companies, f, indent=2, ensure_ascii=False)
        
    df_out = pd.DataFrame(companies)
    df_out.to_csv("data/companies.csv", index=False, encoding="utf-8")
    
    print(f"Successfully extracted {len(companies)} companies to data/companies.json and data/companies.csv")
    return companies

if __name__ == "__main__":
    extract_companies()
