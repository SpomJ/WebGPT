import requests

from duckduckgo_search import DDGS
import googlesearch
from html2text import HTML2Text

def search_qtext(q):
    try:
        return DDGS().text(q, max_results=1)[0]['body']
    except IndexError:
        return None

def search_url(q):
    try:
        return googlesearch.search(q, num_results=1).next()
    except Exception:
        try:
            return DDGS().text(q, max_results=1)[0]['href']
        except Exception:
            return None


def get_page(url):
    return requests.get(url, timeout=15).text

def parse_html(html):
    return HTML2Text().handle(html)

def search_full(q):
    try:
        u = search_url(q)
        # print(u)
        p = get_page(u)
        # print(p[:50])
        h = parse_html(p)
        return h
    except:
        return


