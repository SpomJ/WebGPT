import requests

from duckduckgo_search import DDGS
import googlesearch
from html2text import HTML2Text

def search_qtext(q):
    try:
        return DDGS().text(q, max_results=1)[0]['body']
    except IndexError:
        return None

def search_url(q, n=1):
    try:
        return list(googlesearch.search(q, num_results=n))
    except Exception:
        try:
            return list(map(lambda x: x['href'], DDGS().text(q, max_results=n)))
        except Exception:
            return None


def get_page(url):
    return requests.get(url, timeout=15).text

def parse_html(html):
    return HTML2Text().handle(html)

def search_full(q, n=1):
    urls = search_url(q, n)
    print(urls)
    texts = []
    for url in urls:
        try:
            p = get_page(url)
            texts.append(parse_html(p))
        except Exception:
            pass
    print(texts)
    return texts

