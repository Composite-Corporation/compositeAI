from typing import Optional, List
from pydantic import BaseModel, Field

from compositeai.tools import BaseTool
from compositeai.drivers import BaseDriver


class _AgentFinishTool(BaseTool):
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