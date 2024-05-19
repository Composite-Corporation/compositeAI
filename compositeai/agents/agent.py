from typing import List, Optional, Callable, Any
from pydantic import BaseModel, Field, PrivateAttr, validator
import json

from compositeai.drivers.base_driver import BaseDriver
from compositeai.drivers.openai_driver import OpenAIDriver
from compositeai.tools import BaseTool


def _finish(result: str):
    """
    Return the final result once you believe you have completed the task at hand.
    """
    return result

class _AgentFinish(BaseTool):
    func: Callable = _finish


class Agent(BaseModel):
    driver: Optional[BaseDriver] = Field(default=OpenAIDriver(model="gpt-4-turbo"))
    description: str
    tools: Optional[List[BaseTool]] = Field(default=None)
    max_iterations: int = Field(default=10, ge=0)

    # Private attributes
    _scratchpad: List = PrivateAttr(default=[])
    _conversation_history: List = PrivateAttr(default=[])
    _memory: List = PrivateAttr(default=[])


    @validator('tools', pre=True)
    def add_default_tools(cls, value):
        """Add default tools to tool list"""
        if not value:
            value = [_AgentFinish()]
        else:
            value.append(_AgentFinish())
        return value
    

    def directed_edge(self, agent: 'Agent'):
        # Check if arg is instance of Agent class
        if not isinstance(agent, Agent):
            raise ValueError("Argument must be an instance of Agent.")
        
        raise NotImplementedError("Method not implemented.")
    

    def execute(self, task: str, input: Optional[Any] = None) -> str:
        # Clear memory at start
        self._memory.clear()

        # Instantiate task based on input string
        if input:
            task = f"""
            {task}

            Here is what you are given:
            {input}
            """

        # Instantiate system/user messages based on agent description and task
        system_message = {"role": "system", "content": self.description}
        user_message = {"role": "user", "content": task}

        # Primitive memory system
        self._memory.append(system_message)
        self._memory.append(user_message)

        # Loop through iterations
        for _ in range(self.max_iterations):
            # Obtain response from driver iteration and append to memory
            driver_response = self.driver._iterate(messages=self._memory, tools=self.tools)
            message = driver_response.choices[0].message
            self._memory.append(message)
            print(driver_response)
            print("\n")

            # Usage data 
            usage = driver_response.usage

            # If function called, loop through each, obtain schema, and append to memory
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    # Get function call info
                    tool_call_id = tool_call.id
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    # Iterate through provided tools to check if driver_response function call matches one
                    no_match_flag = True
                    for tool in self.tools:
                        # If match, run tool function on arguments for result, and append to memory
                        if tool.get_schema().name == function_name:
                            no_match_flag = False
                            function_result = tool.func(**function_args)
                            self._memory.append({"role": "tool", "content": str(function_result), "tool_call_id": tool_call_id})

                            # If agent_finish called, return result
                            if function_name == "_finish":
                                return function_result
                    
                    # If driver_response function call matches none of the given tools
                    if no_match_flag:
                        raise Exception("Driver called function, function call does not match any of the provided tools.")
        
        # At this point, maximum number of iterations reached
        raise Exception("Maximum number of iterations reached.")
            