from abc import ABC, abstractmethod
from typing import Optional, List, Dict
from pydantic import BaseModel
from dotenv import load_dotenv

from compositeai.tools import BaseTool

load_dotenv()

class BaseDriver(ABC):
    def __init__(self, model: str) -> None:
       self.model = model

    @abstractmethod
    def _iterate(
        self, 
        messages: List,
        tools: Optional[List[BaseTool]],
    ):
        raise NotImplementedError("BaseDriver should not be utilized.")
    
