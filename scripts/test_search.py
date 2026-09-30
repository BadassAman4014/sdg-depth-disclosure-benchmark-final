from duckduckgo_search import DDGS
from googlesearch import search
import time

print("Testing DDGS...")
try:
    with DDGS() as ddgs:
        results = list(ddgs.text('Adidas AG Annual Report 2020 filetype:pdf', max_results=5))
        for r in results:
            print("DDGS:", r.get('href'), "|", r.get('title'))
except Exception as e:
    print("DDGS Error:", e)

print("\nTesting Google Search...")
try:
    g_results = list(search('Adidas AG Annual Report 2020 filetype:pdf', num_results=5))
    for gr in g_results:
        print("Google:", gr)
except Exception as e:
    print("Google Error:", e)
