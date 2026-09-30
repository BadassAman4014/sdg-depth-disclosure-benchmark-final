import requests
from bs4 import BeautifulSoup
import json

def test_eqs_news():
    # EQS News search endpoint
    url = "https://www.eqs-news.com/api/news"
    # Or search page
    search_url = "https://www.eqs-news.com/news/financial-news/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }
    r = requests.get(search_url, headers=headers)
    print("EQS News status:", r.status_code)

test_eqs_news()
