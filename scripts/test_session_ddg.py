import urllib.parse
import requests
from bs4 import BeautifulSoup
import time
import random

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://html.duckduckgo.com/"
})

def search_ddg(query):
    url = "https://html.duckduckgo.com/html/"
    resp = session.post(url, data={"q": query}, timeout=10)
    if resp.status_code != 200:
        print(f"Status {resp.status_code}")
        return []
    soup = BeautifulSoup(resp.text, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "uddg=" in href:
            target = urllib.parse.unquote(href.split("uddg=")[1].split("&")[0])
            if ".pdf" in target.lower():
                links.append(target)
    return links

queries = [
    "Adidas AG Annual Report 2019 filetype:pdf",
    "Adidas AG Annual Report 2020 filetype:pdf",
    "Adidas AG Annual Report 2021 filetype:pdf",
    "Allianz SE Annual Report 2018 filetype:pdf",
    "BASF SE Annual Report 2020 filetype:pdf"
]

for q in queries:
    print("Querying:", q)
    results = search_ddg(q)
    print(f"Found {len(results)} PDFs:")
    for r in results[:3]:
        print("  -", r)
    time.sleep(random.uniform(2.0, 3.5))
