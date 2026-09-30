import urllib.parse
import requests
from bs4 import BeautifulSoup
import time

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

def test_bing(query):
    url = f"https://www.bing.com/search?q={urllib.parse.quote(query)}"
    r = requests.get(url, headers=headers, timeout=10)
    print("Bing status:", r.status_code)
    soup = BeautifulSoup(r.text, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("http") and ("pdf" in href.lower() or "report" in href.lower() or "investor" in href.lower()):
            links.append(href)
    print(f"Bing found {len(links)} links:")
    for l in links[:5]:
        print("  -", l)

def test_ddg_lite(query):
    url = "https://lite.duckduckgo.com/lite/"
    r = requests.post(url, data={"q": query}, headers=headers, timeout=10)
    print("DDG Lite status:", r.status_code)
    soup = BeautifulSoup(r.text, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "uddg=" in href:
            target = urllib.parse.unquote(href.split("uddg=")[1].split("&")[0])
            if target.startswith("http") and ("pdf" in target.lower() or "report" in target.lower()):
                links.append(target)
    print(f"DDG Lite found {len(links)} links:")
    for l in links[:5]:
        print("  -", l)

print("--- Testing Bing ---")
test_bing("Adidas AG Annual Report 2020 filetype:pdf")

time.sleep(2)
print("\n--- Testing DDG Lite ---")
test_ddg_lite("Adidas AG Annual Report 2020 filetype:pdf")
