import os
import sys
import re
import time
import json
import random
import argparse
import urllib.parse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import pandas as pd
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from ddgs import DDGS

# Ensure output is unbuffered
sys.stdout.reconfigure(line_buffering=True)

manifest_lock = threading.Lock()

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15",
]

def get_headers():
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,de;q=0.8",
        "Connection": "keep-alive"
    }

def score_url(url: str, company_name: str, year: int) -> int:
    """Score a candidate PDF URL based on relevance to the company's full annual report."""
    url_lower = url.lower()
    score = 0
    str_year = str(year)
    short_year = str_year[2:]
    
    # Year matching
    if str_year in url_lower:
        score += 35
    elif f"ar{short_year}" in url_lower or f"gb{short_year}" in url_lower or f"fy{short_year}" in url_lower or f"_{short_year}" in url_lower:
        score += 25
    else:
        for other_yr in range(2010, 2030):
            if other_yr != year and str(other_yr) in url_lower:
                score -= 50
                
    # High-value annual report keywords
    annual_indicators = [
        "annual-report", "annual_report", "annualreport", "annual-results",
        "geschaeftsbericht", "geschaftsbericht", "geschaeftsberichte", "gb_",
        "integrated-report", "integrated_report", "financial-report",
        "annual-financial-report", "jahresbericht", "annualreports.com",
        f"ar{str_year}", f"ar_{str_year}", f"ar{short_year}", f"ar_{short_year}",
        f"gb{str_year}", f"gb_{str_year}", f"gb{short_year}", f"gb_{short_year}"
    ]
    for kw in annual_indicators:
        if kw in url_lower:
            score += 30
            
    # Penalize non-annual or partial documents
    penalty_keywords = [
        "presentation", "factsheet", "fact_sheet", "fact-sheet", "invitation", "einladung",
        "notice", "remuneration", "verguetung", "corporate-governance", "proxy",
        "press-release", "interim", "half-year", "halbjahr", "q1", "q2", "q3", "q4",
        "9m", "6m", "3m", "quarterly", "quartal", "flyer", "leaflet", "esg-data", "summary",
        "statutes", "satzung", "voting", "stimmrechte"
    ]
    for pkw in penalty_keywords:
        if pkw in url_lower:
            score -= 40
            
    # Check company name tokens
    tokens = [t.lower() for t in re.split(r'[\s\-_&]+', company_name) 
              if len(t) > 2 and t.lower() not in ['ag', 'se', 'kgaa', 'gmbh', 'inc', 'plc', 'group', 'the', 'holdings', 'holding', 'corporate']]
    for token in tokens:
        if token in url_lower:
            score += 15
            
    if url_lower.endswith('.pdf') or '.pdf?' in url_lower or '.pdf#' in url_lower:
        score += 10
        
    return score

def extract_pdf_links_from_html(page_url: str, target_year: int) -> list:
    """Scrape candidate PDF links from investor relations/download center web pages."""
    try:
        r = requests.get(page_url, headers=get_headers(), timeout=10)
        if r.status_code != 200:
            return []
        soup = BeautifulSoup(r.text, 'html.parser')
        pdfs = []
        str_yr = str(target_year)
        short_yr = str_yr[2:]
        
        for a in soup.find_all('a', href=True):
            href = a['href']
            full_url = urllib.parse.urljoin(page_url, href)
            url_lower = full_url.lower()
            if '.pdf' in url_lower:
                link_text = a.get_text().lower()
                if (str_yr in url_lower or str_yr in link_text or f"ar{short_yr}" in url_lower or f"gb{short_yr}" in url_lower):
                    if any(kw in url_lower or kw in link_text for kw in ['annual', 'report', 'geschaeft', 'bericht', 'financial', 'integrated', 'download']):
                        pdfs.append(full_url)
        return list(set(pdfs))
    except Exception:
        return []

def search_pdf_candidates(company_name: str, year: int, isin: str = "") -> list:
    """Execute multi-query search to discover candidate PDFs."""
    queries = [
        f"{company_name} Annual Report {year}",
        f"{company_name} Geschäftsbericht {year}",
        f"{company_name} Annual Report {year} filetype:pdf"
    ]
    
    found_urls = set()
    html_pages_to_scrape = []
    
    for q in queries:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(q, max_results=8))
                
            for r in results:
                href = r.get("href", "")
                if not href.startswith("http"):
                    continue
                if href.lower().endswith(".pdf") or ".pdf?" in href.lower() or ".pdf#" in href.lower():
                    found_urls.add(href)
                else:
                    if any(kw in href.lower() for kw in ["report", "investor", "annualreports.com", "financial", "download", "eqs-news"]):
                        html_pages_to_scrape.append(href)
                        
            if len(found_urls) >= 3:
                break
            time.sleep(random.uniform(0.3, 0.8))
        except Exception:
            time.sleep(0.5)
            continue
            
    for html_page in html_pages_to_scrape[:3]:
        extracted = extract_pdf_links_from_html(html_page, year)
        for ep in extracted:
            found_urls.add(ep)
            
    scored = [(u, score_url(u, company_name, year)) for u in found_urls]
    scored = [item for item in scored if item[1] > 0]
    scored.sort(key=lambda x: x[1], reverse=True)
    
    return [item[0] for item in scored]

def validate_and_save_pdf(pdf_bytes: bytes, target_path: str, min_size_kb: int = 150) -> tuple:
    """Validate PDF integrity, header magic bytes, and page count."""
    if len(pdf_bytes) < min_size_kb * 1024:
        return False, f"File too small ({round(len(pdf_bytes)/1024, 1)} KB < {min_size_kb} KB)", 0
        
    if not pdf_bytes.startswith(b"%PDF-"):
        return False, "Invalid PDF header magic bytes", 0
        
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    temp_path = target_path + f".tmp_{threading.get_ident()}"
    with open(temp_path, "wb") as f:
        f.write(pdf_bytes)
        
    try:
        reader = PdfReader(temp_path)
        num_pages = len(reader.pages)
        if num_pages < 12:
            os.remove(temp_path)
            return False, f"PDF has only {num_pages} pages (too short for an annual report)", num_pages
            
        if os.path.exists(target_path):
            os.remove(target_path)
        os.rename(temp_path, target_path)
        return True, "Valid PDF", num_pages
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False, f"Corrupted PDF: {str(e)}", 0

def download_candidate(url: str, target_path: str, timeout: int = 25) -> tuple:
    """Download candidate PDF and validate."""
    try:
        resp = requests.get(url, headers=get_headers(), timeout=timeout, stream=True)
        if resp.status_code != 200:
            return False, f"HTTP {resp.status_code}", 0, 0
            
        content = resp.content
        size_mb = round(len(content) / (1024 * 1024), 2)
        valid, msg, pages = validate_and_save_pdf(content, target_path)
        return valid, msg, size_mb, pages
    except Exception as e:
        return False, f"Download error: {str(e)}", 0, 0

def process_company_year(company: dict, year: int, reports_dir: str, overwrite: bool = False) -> dict:
    """Process single company for a specific fiscal year."""
    folder_name = company["folder_name"]
    company_name = company["company_name"]
    isin = company.get("isin", "")
    ric = company.get("ric", "")
    
    year_dir = os.path.join(reports_dir, folder_name, str(year))
    expected_filename = f"{folder_name}_Annual_Report_{year}.pdf"
    target_path = os.path.join(year_dir, expected_filename)
    
    if not overwrite and os.path.exists(target_path):
        try:
            size_mb = round(os.path.getsize(target_path) / (1024 * 1024), 2)
            reader = PdfReader(target_path)
            pages = len(reader.pages)
            if pages >= 12 and size_mb >= 0.15:
                return {
                    "company_id": company.get("id"),
                    "company_name": company_name,
                    "isin": isin,
                    "ric": ric,
                    "year": year,
                    "filename": expected_filename,
                    "file_path": target_path,
                    "file_size_mb": size_mb,
                    "page_count": pages,
                    "download_url": "CACHED",
                    "status": "SUCCESS (Cached)",
                    "timestamp": datetime.now().isoformat()
                }
        except Exception:
            pass
            
    print(f"[{threading.current_thread().name}] Searching {company_name} ({year})...")
    candidates = search_pdf_candidates(company_name, year, isin)
    
    if not candidates:
        print(f"  [{threading.current_thread().name}] [NOT FOUND] No PDF candidates discovered for {company_name} ({year})")
        return {
            "company_id": company.get("id"),
            "company_name": company_name,
            "isin": isin,
            "ric": ric,
            "year": year,
            "filename": "",
            "file_path": "",
            "file_size_mb": 0,
            "page_count": 0,
            "download_url": "",
            "status": "NOT_FOUND",
            "timestamp": datetime.now().isoformat()
        }
        
    for url in candidates[:6]:
        print(f"  [{threading.current_thread().name}] Attempting: {url[:80]}...")
        valid, msg, size_mb, pages = download_candidate(url, target_path)
        if valid:
            print(f"  [{threading.current_thread().name}] [SUCCESS] Acquired: {expected_filename} ({size_mb} MB, {pages} pages)")
            return {
                "company_id": company.get("id"),
                "company_name": company_name,
                "isin": isin,
                "ric": ric,
                "year": year,
                "filename": expected_filename,
                "file_path": target_path,
                "file_size_mb": size_mb,
                "page_count": pages,
                "download_url": url,
                "status": "SUCCESS",
                "timestamp": datetime.now().isoformat()
            }
        else:
            print(f"  [{threading.current_thread().name}] [SKIP] {msg}")
            
    return {
        "company_id": company.get("id"),
        "company_name": company_name,
        "isin": isin,
        "ric": ric,
        "year": year,
        "filename": "",
        "file_path": "",
        "file_size_mb": 0,
        "page_count": 0,
        "download_url": candidates[0] if candidates else "",
        "status": "FAILED: Candidates rejected",
        "timestamp": datetime.now().isoformat()
    }

def update_manifest(manifest_path: str, record: dict):
    """Thread-safe append or update record in reports manifest CSV."""
    with manifest_lock:
        columns = [
            "company_id", "company_name", "isin", "ric", "year",
            "filename", "file_path", "file_size_mb", "page_count",
            "download_url", "status", "timestamp"
        ]
        
        if os.path.exists(manifest_path):
            try:
                df = pd.read_csv(manifest_path)
            except Exception:
                df = pd.DataFrame(columns=columns)
        else:
            df = pd.DataFrame(columns=columns)
            
        mask = (df["company_name"] == record["company_name"]) & (df["year"] == record["year"])
        if mask.any():
            df = df[~mask]
            
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
        df.sort_values(by=["company_id", "year"], inplace=True)
        df.to_csv(manifest_path, index=False, encoding="utf-8")

def process_company_all_years(company: dict, years: list, reports_dir: str, manifest_path: str, overwrite: bool) -> tuple:
    """Worker task processing all years for a company."""
    c_name = company["company_name"]
    print(f"\n===> Worker {threading.current_thread().name} starting {c_name} (ID: {company['id']})")
    comp_success = 0
    for yr in years:
        rec = process_company_year(company, yr, reports_dir, overwrite)
        update_manifest(manifest_path, rec)
        if "SUCCESS" in rec["status"]:
            comp_success += 1
        time.sleep(random.uniform(0.3, 0.8))
    print(f"<=== Worker {threading.current_thread().name} finished {c_name} ({comp_success}/{len(years)} acquired)")
    return company["id"], comp_success

def main():
    parser = argparse.ArgumentParser(description="Multi-worker annual reports downloader.")
    parser.add_argument("--start-idx", type=int, default=1, help="Start company ID (1-based)")
    parser.add_argument("--end-idx", type=int, default=None, help="End company ID (inclusive)")
    parser.add_argument("--start-year", type=int, default=2014, help="Earliest year (default: 2014)")
    parser.add_argument("--end-year", type=int, default=2024, help="Latest year (default: 2024)")
    parser.add_argument("--companies", type=str, default="", help="Comma-separated company names or substrings")
    parser.add_argument("--years", type=str, default="", help="Comma-separated specific years")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel worker threads (default: 4)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing reports")
    parser.add_argument("--reports-dir", type=str, default="reports", help="Destination reports directory")
    args = parser.parse_args()
    
    companies_file = "data/companies.json"
    if not os.path.exists(companies_file):
        print("Companies data file not found. Running extract_companies.py...")
        import subprocess
        subprocess.run([sys.executable, "scripts/extract_companies.py"], check=True)
        
    with open(companies_file, "r", encoding="utf-8") as f:
        companies = json.load(f)
        
    if args.companies:
        target_names = [n.strip().lower() for n in args.companies.split(",")]
        companies = [c for c in companies if any(tn in c["company_name"].lower() for tn in target_names)]
    else:
        start_idx = args.start_idx
        end_idx = args.end_idx if args.end_idx is not None else len(companies)
        companies = [c for c in companies if start_idx <= c["id"] <= end_idx]
        
    if args.years:
        years = [int(y.strip()) for y in args.years.split(",") if y.strip().isdigit()]
    else:
        years = list(range(args.start_year, args.end_year + 1))
        
    os.makedirs(args.reports_dir, exist_ok=True)
    manifest_path = os.path.join(args.reports_dir, "reports_manifest.csv")
    
    print("=" * 80)
    print(f"MULTI-WORKER ANNUAL REPORTS DOWNLOADER (WORKERS: {args.workers})")
    print(f"Total Companies in Batch: {len(companies)}")
    print(f"Years: {years}")
    print(f"Output Directory: {os.path.abspath(args.reports_dir)}")
    print("=" * 80)
    
    total_acquired = 0
    with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="Worker") as executor:
        futures = {
            executor.submit(process_company_all_years, comp, years, args.reports_dir, manifest_path, args.overwrite): comp
            for comp in companies
        }
        for future in as_completed(futures):
            comp = futures[future]
            try:
                c_id, succ = future.result()
                total_acquired += succ
            except Exception as exc:
                print(f"Company {comp['company_name']} generated an exception: {exc}")
                
    print("\n" + "=" * 80)
    print(f"Multi-Worker Processing Finished: {total_acquired} reports acquired.")
    print(f"Manifest saved to: {manifest_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
