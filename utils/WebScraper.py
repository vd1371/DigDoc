import os
import json
import requests
import time
import random
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from googlesearch import search


class WebScraper:
    def __init__(self, **params):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.session = requests.Session()
        self.max_pages = params.get("max_pages", 10)
        self.max_depth = params.get("max_depth", 1)
        self.load_from_cache = params.get("load_from_cache", False)
        self.project_path = params.get("project_path", "./")

        self.cache_direc = os.path.join(self.project_path, "scrapping_cache.json")
        if not os.path.exists(self.cache_direc):
            with open(self.cache_direc, "w") as f:
                json.dump({}, f)

        if self.load_from_cache:
            with open(self.cache_direc, "r") as f:
                self.cache = json.load(f)
        else:
            self.cache = {}

    def add_to_cache(self, url, content, page_urls):
        self.cache[url] = {
            "content": content,
            "page_urls": page_urls
        }
        with open(self.cache_direc, "w") as f:
            json.dump(self.cache, f)

    def get_page(self, url):
        try:
            response = self.session.get(url, headers=self.headers)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None

    def parse_content(self, html):
        soup = BeautifulSoup(html, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Get text
        text = soup.get_text()

        # Break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())

        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))

        # Drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)

        return text

    def scrape_website(self, start_urls_or_url):

        visited = set()
        assert self.max_depth > 0, "max_depth must be greater than 0, at least 1"

        to_visit = {i: [] for i in range(self.max_depth)}
        if isinstance(start_urls_or_url, list):
            to_visit[0] = start_urls_or_url
        else:
            to_visit[0] = [start_urls_or_url]

        scraped_content = []
        for depth in to_visit:
            for url in to_visit[depth]:

                if url in visited or len(visited) >= self.max_pages:
                    continue

                if url in self.cache and self.load_from_cache:

                    print (f"Using cached content for {url}")
                    scraped_content.append((url, self.cache[url]["content"]))
                    visited.add(url)

                    # Add new urls to visit
                    for new_url in self.cache[url]["page_urls"]:
                        if new_url not in visited and depth + 1 < self.max_depth:
                            to_visit[depth + 1].append(new_url)

                    continue

                print(f"Scraping: {url}")
                html = self.get_page(url)
                if html:
                    content = self.parse_content(html)
                    scraped_content.append((url, content))
                    visited.add(url)

                    # Find more links
                    soup = BeautifulSoup(html, 'html.parser')

                    new_urls = []
                    for link in soup.find_all('a', href=True):
                        new_url = urljoin(url, link['href'])

                        if not is_valid_url(to_visit, new_url):
                            continue

                        new_urls.append(new_url)

                        if new_url not in visited and depth + 1 < self.max_depth:
                            to_visit[depth + 1].append(new_url)

                    self.add_to_cache(url, content, new_urls)

                # Add a random delay between requests
                time.sleep(random.uniform(1, 3))

        return scraped_content

    def scrape_google_search(self, target, query):

        if target != "all":
            query += f" site:{target}"

        urls = []
        for url in search(query, num_results=self.max_pages):
            urls.append(url)

        return self.scrape_website(urls)


def is_valid_url(to_visit, new_url):

    if "javascript:void(0)" in new_url:
        return False

    if to_visit[0][0].startswith("https://scholar.google.com") and \
            not new_url.startswith("https://scholar.google.com/citations?view_op=view_citation"):
        return False

    return True