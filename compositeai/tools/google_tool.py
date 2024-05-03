import requests
import json
import os
from dotenv import load_dotenv

from compositeai.tools import BaseTool

load_dotenv()

class GoogleSerperApiTool(BaseTool):
    def __init__(self) -> None:
        _SERP_API_KEY = os.getenv("SERP_API_KEY")

        def _google_serp_api(query: str):
            """
            Return Google search results based on a query.
            """
            url = "https://google.serper.dev/search"
            payload = json.dumps({
                "q": query
            })
            headers = {
                'X-API-KEY': _SERP_API_KEY,
                'Content-Type': 'application/json'
            }
            response = requests.request("POST", url, headers=headers, data=payload)
            return response.json()["organic"]
        
        super().__init__(func=_google_serp_api)