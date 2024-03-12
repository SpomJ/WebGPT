import requests
from html2text import HTML2Text

def get_page(url):
    return HTML2Text().handle(requests.get(url).text)

