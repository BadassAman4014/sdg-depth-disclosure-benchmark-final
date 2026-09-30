import base64
import urllib.parse
import requests
from bs4 import BeautifulSoup

def decode_bing_url(u_param):
    if u_param.startswith('a1'):
        b64 = u_param[2:]
        b64 += '=' * (-len(b64) % 4)
        try:
            return base64.b64decode(b64).decode('utf-8', errors='ignore')
        except Exception:
            return ""
    return u_param

def search_bing(query):
    url = f"https://www.bing.com/search?q={urllib.parse.quote(query)}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }
    r = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(r.text, "html.parser")
    urls = []
    for b in soup.select("li.b_algo h2 a"):
        href = b.get("href", "")
        if "u=" in href:
            u_param = href.split("u=")[1].split("&")[0]
            real = decode_bing_url(u_param)
            if real:
                urls.append(real)
        elif href.startswith("http"):
            urls.append(href)
    return urls

print("Searching Bing for Adidas 2020 PDF:")
urls = search_bing('"Adidas AG" "Annual Report" "2020" filetype:pdf')
for u in urls:
    print(" ->", u)

print("\nSearching Bing for BMW 2017 PDF:")
urls = search_bing('"Bayerische Motoren Werke AG" "Annual Report" 2017 filetype:pdf')
for u in urls:
    print(" ->", u)
