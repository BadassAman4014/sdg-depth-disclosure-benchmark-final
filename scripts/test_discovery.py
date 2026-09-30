from ddgs import DDGS
import requests
from bs4 import BeautifulSoup
import urllib.parse
import re

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
}

def extract_pdf_links_from_html(page_url, target_year):
    try:
        r = requests.get(page_url, headers=headers, timeout=10)
        if r.status_code != 200:
            return []
        soup = BeautifulSoup(r.text, 'html.parser')
        pdfs = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            full_url = urllib.parse.urljoin(page_url, href)
            if '.pdf' in full_url.lower():
                # Check if target year or annual report is in link or text
                link_text = a.get_text().lower()
                url_lower = full_url.lower()
                str_yr = str(target_year)
                if (str_yr in url_lower or str_yr in link_text) and any(kw in url_lower or kw in link_text for kw in ['annual', 'report', 'geschaeft', 'bericht', 'financial']):
                    pdfs.append(full_url)
        return list(set(pdfs))
    except Exception as e:
        return []

def test_discovery(company_name, year):
    print(f"\n=== Discovering {company_name} ({year}) ===")
    query = f"{company_name} Annual Report {year}"
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=8))
        
    direct_pdfs = []
    html_pages = []
    
    for r in results:
        href = r.get('href', '')
        if href.lower().endswith('.pdf') or '.pdf?' in href.lower() or '.pdf#' in href.lower():
            direct_pdfs.append(href)
        else:
            html_pages.append(href)
            
    print(f"Direct PDFs found ({len(direct_pdfs)}):")
    for p in direct_pdfs:
        print("  [PDF]", p)
        
    print(f"HTML pages found ({len(html_pages)}):")
    for h in html_pages[:4]:
        extracted = extract_pdf_links_from_html(h, year)
        print(f"  [HTML] {h[:60]} -> {len(extracted)} matching PDFs")
        for ep in extracted[:2]:
            print("     ->", ep)

test_discovery("Adidas AG", 2020)
test_discovery("Allianz SE", 2017)
test_discovery("BMW AG", 2019)
