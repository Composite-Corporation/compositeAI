from typing import Generator, List, Optional, Union
from pydantic import BaseModel, Field, PrivateAttr, validator
import textwrap
import json

from compositeai.drivers.base_driver import BaseDriver
from compositeai.drivers.openai_driver import OpenAIDriver
from compositeai.tools import BaseTool
from compositeai.tools.default_agent_tools import _AgentFinishTool


class AgentResult(BaseModel):
    content: str = Field("Final result of agent execution")


class Agent(BaseModel):
    driver: BaseDriver
    description: str = Field("Description of AI agent role")
    tools: Optional[List[BaseTool]] = Field(default=None)
    max_iterations: int = Field(default=10, ge=0)
    _scratchpad: List[str] = PrivateAttr(default=[])


    @validator('tools', pre=True)
    def add_default_tools(cls, value):
        """Add default tools to tool list"""
        if not value:
            value = [_AgentFinishTool()]
        else:
            value.append(_AgentFinishTool())
        return value
    

    def directed_edge(self, agent: 'Agent'):
        # Check if arg is instance of Agent class
        if not isinstance(agent, Agent):
            raise ValueError("Argument must be an instance of Agent.")
        raise NotImplementedError("Method not implemented.")
    

    def execute(self, task: str, input: Optional[str] = None, stream: bool = False) -> Union[Generator, AgentResult]:
        # Add initial data to conversation history and agent scratchpad
        self._scratchpad.append(f"task: {task}")
        if input:
            self._scratchpad.append(input)

        # Loop through iterations
        for _ in range(self.max_iterations):
            for chunk in self._iterate():
                if stream:
                    yield chunk
                if isinstance(chunk, AgentResult):
                    return chunk

        # At this point, maximum number of iterations reached
        raise RuntimeError("Maximum number of iterations reached.")
    
    
    def refresh_memory(self):
        memory = f"""
            YOU ARE AN AI AGENT GIVEN THE FOLLOWING ROLE:
            {self.description}

            YOU ARE TO COMPLETE THE TASK BASED ON THE FOLLOWING PROGRESS:
            {self._scratchpad}
        """
        return textwrap.dedent(memory)
    

    def _iterate(self) -> Generator:
        memory = self.refresh_memory()
        plan = self.driver._plan(
            system_prompt=memory,
            tools=self.tools,
        )
        self._scratchpad.append(f"plan: {plan.content}")
        yield plan

        memory = self.refresh_memory()
        action = self.driver._action(
            system_prompt=memory,
            tools=self.tools,
        )
        tool_calls = action.tool_calls
        json_tool_calls = []
        for tool_call in tool_calls:
            json_tool_calls.append(tool_call.json())
            if tool_call.name == _AgentFinishTool().func.__name__:
                result = json.loads(tool_call.args)["result"]
                yield AgentResult(content=result)
                return
        self._scratchpad.append(f"action(s): Calling the following tools - {json_tool_calls}")
        yield action
        
        memory = self.refresh_memory()
        observation = self.driver._observe(
            system_prompt=memory,
            tools=self.tools,
            action=action,
        )
        self._scratchpad.append(f"observation(s): {observation.content}")
        yield observation
            