import requests
from bs4 import BeautifulSoup
soup = BeautifulSoup('html.parser')


class WebScraper:
    def __init__(self, url):
        self.url = url

    def scrape(self):
        response = requests.get(self.url)
        if response.status_code == 200:
            return response.text
        else:
            raise Exception(f"Failed to retrieve content from {self.url} with status code {response.status_code}")