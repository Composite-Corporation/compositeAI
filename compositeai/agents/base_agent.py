from typing import Generator, List, Optional, Union, Any
from pydantic import BaseModel, Field
from abc import abstractmethod

from compositeai.tools import BaseTool
from compositeai.drivers import BaseDriver
    

class AgentOutput(BaseModel):
    content: Any = Field("An output from agent execution")


class AgentResult(AgentOutput):
    content: Any = Field("Final result of agent execution")


class AgentStep(AgentOutput):
    content: Any = Field("Intermediate step of agent execution")


class AgentExecution(BaseModel):
    steps: List[AgentStep] = Field("Intermediate steps that the agent has taken during execution")
    result: AgentResult = Field("Final result of agent execution")


class BaseAgent(BaseModel):
    driver: BaseDriver
    name: str = Field("Name of the AI agent")
    description: str = Field("Description of AI agent role")
    is_entry: Optional[bool] = Field(default=False, description="Setting to True makes this agent the entry point for the user")
    tools: Optional[List[BaseTool]] = Field(default=None)
    max_iterations: Optional[int] = Field(default=10, ge=0)
    

    def execute(self, task: str, input: Optional[str] = None, stream: bool = False) -> Union[Generator, AgentExecution]:
        # Initial processing on task/input
        self.exec_init(task=task, input=input)

        def _execute_stream() -> Generator:
            for _ in range(self.max_iterations):
                output = self.iterate()
                yield output
                if isinstance(output, AgentResult):
                    return
            # At this point, maximum number of iterations reached
            raise RuntimeError("Maximum number of iterations reached.")
        
        def _execute_no_stream() -> AgentExecution:
            steps = []
            for _ in range(self.max_iterations):
                output = self.iterate()
                if isinstance(output, AgentStep):
                    steps.append(output)
                if isinstance(output, AgentResult):
                    return AgentExecution(steps=steps, result=output)
            # At this point, maximum number of iterations reached
            raise RuntimeError("Maximum number of iterations reached.")
        
        if stream:
            return _execute_stream()
        else:
            return _execute_no_stream()

    
    @abstractmethod
    def exec_init(self, task: str, input: Optional[str] = None) -> None:
        """Used to initial processing of the task or given prior input - called first in execute"""
        raise NotImplementedError("Method must be implemented by a subclass")


    @abstractmethod
    def iterate(self) -> AgentOutput:
        """An iteration of a the agent execution that returns a useful output - called in execute"""
        raise NotImplementedError("Method must be implemented by a subclass")