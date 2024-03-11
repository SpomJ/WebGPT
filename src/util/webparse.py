import requests
import search

from bs4 import BeautifulSoup

def get_page(url):
    return requests.get(url).text

def extract_main_text(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    main_text = ""

    for li in soup.find_all('li'):
        main_text += li.get_text() + "\n"

    return main_text

def text_from_query(q):
    r = search.results(q, include_all=1)[0]['href']
    print(r)
    p = get_page(r)
    print(p)
    o = extract_main_text(p)
    return o
