from duckduckgo_search import DDGS

try:
    results = list(DDGS().text("Adidas AG Annual Report 2020 filetype:pdf", max_results=10))
    print(f"DDGS returned {len(results)} results:")
    for r in results:
        print("Title:", r.get("title"))
        print("URL:", r.get("href"))
        print("Body:", r.get("body")[:80])
        print("-" * 40)
except Exception as e:
    print("DDGS Error:", e)
