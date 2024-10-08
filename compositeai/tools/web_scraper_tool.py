import requests
from bs4 import BeautifulSoup

from compositeai.tools import BaseTool


class WebScrapeTool(BaseTool):
    name: str = "scrape_website"
    description: str = "Scrape the content of a website given the URL as a string"

    def func(self, url: str) -> str:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")
                text = soup.get_text()
                if len(text) > 16000:
                    return "Requested content exceeds maximum length."
                return text
            else:
                return f"Website scrape failed: status code {response.status_code}"
        except Exception as e:
            return f"Error using scrape_website: {e}"