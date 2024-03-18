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
    except requests.exceptions.HTTPError:
        try:
            return DDGS().text(q, max_results=1)[0]['href']
        except IndexError:
            return None

def get_page(url):
    return requests.get(url).text

def parse_html(html):
    return HTML2Text().html2text(html)

def search_full(q):
    try:
    	return parse_html(get_page(search_url(q)))


