from typing import Optional, List
from pydantic import BaseModel, Field

from compositeai.tools import BaseTool


class DriverUsage(BaseModel):
    completion_tokens: int = Field("Completion tokens used for LLM response", ge=0)
    prompt_tokens: int = Field("Prompt tokens used to generate LLM response", ge=0)
    total_tokens: int = Field("Total tokens used to generate LLM response", ge=0)


class DriverToolCall(BaseModel):
    id: str = Field("ID of tool call")
    name: str = Field("Name of tool/function called")
    args: str = Field("JSON string of arguments")


class DriverResponse(BaseModel):
    usage: DriverUsage = Field("Usage data to generate LLM response")


class DriverPlan(DriverResponse):
    content: str = Field("Message content generated in LLM thought")


class DriverAction(DriverResponse):
    tool_calls: Optional[List[DriverToolCall]] = Field(description="List of tools used for function calls", default=None)


class DriverObservation(DriverResponse):
    content: str = Field("Message content generated in LLM observations")

    
class BaseDriver(BaseModel):
    model: str = Field("Name of the LLM model to use from specific driver")


    def _plan(
        self, 
        system_prompt: str, 
        tools: List[BaseTool],
    ) -> DriverPlan:
        raise NotImplementedError("Subclasses must implement this method.")
    
    
    def _action(
        self, 
        system_prompt: str, 
        tools: List[BaseTool],
        plan: DriverPlan,
    ) -> DriverAction:
        raise NotImplementedError("Subclasses must implement this method.")
    
    
    def _observe(
        self, 
        system_prompt: str,
        tools: List[BaseTool],
        action: DriverAction
    ) -> DriverObservation:
        raise NotImplementedError("Subclasses must implement this method.")
    