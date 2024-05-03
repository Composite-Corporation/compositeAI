from abc import ABC, abstractmethod
from typing import Optional, List, Dict
from pydantic import BaseModel
from dotenv import load_dotenv

from compositeai.tools import BaseTool

load_dotenv()

class DriverToolCall(BaseModel):
    call_id: str
    tool_name: str
    arguments: Dict

class DriverUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class DriverResponse(BaseModel):
    role: str
    content: str | None
    tool_call: DriverToolCall | None
    usage: DriverUsage

class BaseDriver(ABC):
    def __init__(self, model: str) -> None:
       self.model = model

    @abstractmethod
    def _iterate(
        self, 
        messages: List,
        tools: Optional[List[BaseTool]],
    ) -> DriverResponse:
        raise NotImplementedError("BaseDriver should not be utilized.")
    

