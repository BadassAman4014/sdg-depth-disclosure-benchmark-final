import urllib.parse
import requests
from bs4 import BeautifulSoup
import time

def test_ddg_get(q):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/"
    }
    url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(q)}"
    r = requests.get(url, headers=headers, timeout=10)
    print("DDG GET status:", r.status_code, "Length:", len(r.text))
    soup = BeautifulSoup(r.text, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        if "uddg=" in a["href"]:
            target = urllib.parse.unquote(a["href"].split("uddg=")[1].split("&")[0])
            if ".pdf" in target.lower():
                links.append(target)
    print("DDG GET links:", len(links))
    for l in links[:3]:
        print("  ->", l)

def test_qwant(q):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    }
    url = f"https://api.qwant.com/v3/search/web?q={urllib.parse.quote(q)}&count=10&locale=en_US&offset=0&device=desktop"
    r = requests.get(url, headers=headers, timeout=10)
    print("Qwant status:", r.status_code)
    if r.status_code == 200:
        try:
            data = r.json()
            items = data.get("data", {}).get("result", {}).get("items", {}).get("main", [])
            print("Qwant items:", len(items))
            for it in items[:3]:
                print("  ->", it.get("url"))
        except Exception as e:
            print("Qwant parse error:", e)

print("--- DDG GET Test ---")
test_ddg_get("Adidas AG Annual Report 2020 filetype:pdf")
print("\n--- Qwant Test ---")
test_qwant("Adidas AG Annual Report 2020 filetype:pdf")
