import urllib.parse
import requests
from bs4 import BeautifulSoup

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

def test_bing_parse(query):
    url = f"https://www.bing.com/search?q={urllib.parse.quote(query)}"
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.text, "html.parser")
    for b_algo in soup.select("li.b_algo"):
        a = b_algo.find("a")
        if a and a.get("href"):
            print("Bing algo link:", a["href"], "| Title:", a.get_text()[:60])

def test_yahoo(query):
    url = f"https://search.yahoo.com/search?p={urllib.parse.quote(query)}"
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.text, "html.parser")
    for a in soup.select("h3 a, .title a, a[href*='.pdf']"):
        href = a.get("href", "")
        if "RU=" in href:
            import re
            m = re.search(r'RU=([^/]+)/RK', href)
            if m:
                href = urllib.parse.unquote(m.group(1))
        if href.startswith("http"):
            print("Yahoo link:", href, "| Title:", a.get_text()[:60])

print("Bing:")
test_bing_parse("Adidas AG Annual Report 2020 filetype:pdf")

print("\nYahoo:")
test_yahoo("Adidas AG Annual Report 2020 filetype:pdf")
