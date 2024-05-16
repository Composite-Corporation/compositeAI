import requests
from bs4 import BeautifulSoup

from compositeai.tools import BaseTool

class WebScrapeTool(BaseTool):
    def __init__(self) -> None:

        def _scrape_website(url: str):
            """
            Useful for retrieving the text content of a website given a URL.
            """
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")
                text = soup.get_text()
                if len(text) > 16000:
                    return "Requested content exceeds maximum length."
                return text
            else:
                return "Website scrape failed."
            
        super().__init__(func=_scrape_website)