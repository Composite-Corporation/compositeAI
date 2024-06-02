from typing import Generator, List, Optional, Union
from pydantic import BaseModel, Field, validator
from abc import abstractmethod

from compositeai.tools import BaseTool
from compositeai.drivers import BaseDriver


class AgentFinishTool(BaseTool):
    name: str = "agent_finish"
    description: str = "Return the final result once you believe you have completed the task at hand"

    def func(self, result: str) -> str:
        return result
    

class AgentResult(BaseModel):
    content: str = Field("Final result of agent execution")


class BaseAgent(BaseModel):
    driver: BaseDriver
    name: str = Field("Name of the AI agent")
    description: str = Field("Description of AI agent role")
    is_entry: Optional[bool] = Field(default=False, description="Setting to True makes this agent the entry point for the user")
    tools: Optional[List[BaseTool]] = Field(default=None)
    max_iterations: Optional[int] = Field(default=10, ge=0)


    @validator('tools', pre=True)
    def add_default_tools(cls, value):
        """Add default tools to tool list"""
        if not value:
            value = [AgentFinishTool()]
        else:
            value.append(AgentFinishTool())
        return value
    

    @abstractmethod
    def execute(self, task: str, input: Optional[str] = None, stream: bool = False) -> Union[Generator, AgentResult]:
        """Use driver LLM to generate a response (including function calling)"""
        raise NotImplementedError("Method must be implemented by a subclass")