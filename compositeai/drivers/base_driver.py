from typing import Optional, List, Literal
from abc import abstractmethod
from pydantic import BaseModel, Field
from enum import Enum

from compositeai.tools import BaseTool


##### Classes relating to the response info of a driver

class DriverUsage(BaseModel):
    completion_tokens: int = Field("Completion tokens used for LLM response", ge=0)
    prompt_tokens: int = Field("Prompt tokens used to generate LLM response", ge=0)
    total_tokens: int = Field("Total tokens used to generate LLM response", ge=0)


class DriverToolCall(BaseModel):
    id: str = Field("ID of tool call")
    name: str = Field("Name of tool/function called")
    args: str = Field("JSON string of arguments")


class DriverResponse(BaseModel):
    content: Optional[str] = Field(default=None, description="Text generated by the LLM")
    tool_calls: Optional[List[DriverToolCall]] = Field(default=None, description="Tool calls from the LLM")
    usage: DriverUsage = Field("Usage data to generate LLM response")


##### Classes relating to the input data for a driver

class DriverMessage(BaseModel):
    role: str


class SystemMessage(DriverMessage):
    role: Literal["system"]
    content: str
    name: Optional[str] = Field(default=None)


class UserMessage(DriverMessage):
    role: Literal["user"]
    content: str
    name: Optional[str] = Field(default=None)


class AssistantMessage(DriverMessage):
    role: Literal["assistant"]
    content: Optional[str] = Field(default=None)
    name: Optional[str] = Field(default=None)
    tool_calls: Optional[List[DriverToolCall]] = Field(default=None)


class ToolMessage(DriverMessage):
    role: Literal["tool"]
    content: str
    tool_call_id: str


class DriverToolChoice(Enum):
    NONE = 'none'
    AUTO = 'auto'
    REQUIRED = 'required'


class DriverInput(BaseModel):
    messages: List[DriverMessage]
    max_tokens: Optional[int] = Field(default=None, ge=0)
    temperature: Optional[float] = Field(default=0.0, ge=0.0, le=2.0)
    tools: Optional[List[BaseTool]] = Field(default=None)
    tool_choice: Optional[DriverToolChoice] = Field(default=None)


##### Base driver class

class BaseDriver(BaseModel):
    model: str = Field("Name of the LLM model to use from specific driver")

    @abstractmethod
    def generate(
        self,
        input: DriverInput
    ) -> DriverResponse:
        """Use driver LLM to generate a response (including function calling)"""
        raise NotImplementedError("Method must be implemented by a subclass")
    