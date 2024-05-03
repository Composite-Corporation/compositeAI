import inspect, json, os, requests
from typing import List
from pydantic import BaseModel
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

class ParamDesc(BaseModel):
    name: str
    type: str
    required: bool


class ToolSchema(BaseModel):
    name: str
    description: str
    arguments: List[ParamDesc]


class BaseTool():
    def __init__(self, func: callable) -> None:
        # Get function details
        func_name = func.__name__
        func_doc = func.__doc__
        signature = inspect.signature(func)
        arguments = []
        for name, param in signature.parameters.items():
            required = True if param.default is inspect.Parameter.empty else False
            arguments.append(ParamDesc(name=name, type=param.annotation.__name__, required=required))

        # Validation 
        if not func_name:
            raise ValueError("Function of tool must have a name.")
        if not func_doc:
            raise ValueError("Function of tool must have a docstring.")

        # Set function description object
        self.tool = ToolSchema(name=func_name, description=func_doc, arguments=arguments)

    def get_schema(self):
        return self.tool


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
                if len(text) > 50000:
                    return "Requested content exceeds maximum length."
                return text
            else:
                return "Website scrape failed."
            
        super().__init__(func=_scrape_website)
