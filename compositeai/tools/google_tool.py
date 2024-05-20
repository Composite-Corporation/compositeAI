import requests
import json
import os
from typing import Any
from dotenv import load_dotenv

from compositeai.tools import BaseTool


class GoogleSerperApiTool(BaseTool):
    name: str = "google_search"
    description: str = "Retrieve Google search results using the Googler Serper API"


    def __init__(self, **data):
        super().__init__(**data)
        load_dotenv()
        self._SERP_API_KEY = os.getenv("SERP_API_KEY")


    def func(self, query: str) -> Any:
        url = "https://google.serper.dev/search"
        payload = json.dumps({
            "q": query
        })
        headers = {
            'X-API-KEY': self._SERP_API_KEY,
            'Content-Type': 'application/json'
        }
        response = requests.request("POST", url, headers=headers, data=payload)
        return response.json()["organic"]