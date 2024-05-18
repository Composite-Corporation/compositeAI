from typing import Optional, List
from pydantic import BaseModel

from compositeai.tools import BaseTool

    
class BaseDriver(BaseModel):
    model: str

    def _iterate(
        self,
        messages: List,
        tools: Optional[List[BaseTool]],
    ):
        raise NotImplementedError("Subclasses must implement this method.")
