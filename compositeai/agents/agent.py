from typing import List, Optional, Callable, Any
import json

from compositeai.drivers.base_driver import BaseDriver
from compositeai.tools import BaseTool


def _finish(result: str):
    """
    Return the final result once you believe you have completed the task at hand.
    """
    return result


class _AgentFinish(BaseTool):
    func: Callable = _finish


class Agent():
    def __init__(
        self, 
        driver: BaseDriver,
        description: str,
        tools: Optional[List[BaseTool]],
        max_iterations: int = 10,
    ) -> None:
        self.driver = driver
        self.description = description
        self.max_iterations = max_iterations
        self.memory = []

        # Logic to add agent finish tool to final tools list
        if not tools:
            self.tools = [_AgentFinish()]
        else:
            tools.append(_AgentFinish())
            self.tools = tools


    def directed_edge(self, agent: 'Agent'):
        # Check if arg is instance of Agent class
        if not isinstance(agent, Agent):
            raise ValueError("Argument must be an instance of Agent.")
        
        raise NotImplementedError("Method not implemented.")
        

    def execute(self, task: str, input: Optional[Any] = None) -> str:
        # Clear memory at start
        self.memory.clear()

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
        self.memory.append(system_message)
        self.memory.append(user_message)

        # Loop through iterations
        for _ in range(self.max_iterations):
            # Obtain response from driver iteration and append to memory
            driver_response = self.driver._iterate(messages=self.memory, tools=self.tools)
            message = driver_response.choices[0].message
            self.memory.append(message)
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
                            self.memory.append({"role": "tool", "content": str(function_result), "tool_call_id": tool_call_id})

                            # If agent_finish called, return result
                            if function_name == "_finish":
                                return function_result
                    
                    # If driver_response function call matches none of the given tools
                    if no_match_flag:
                        raise Exception("Driver called function, function call does not match any of the provided tools.")
        
        # At this point, maximum number of iterations reached
        raise Exception("Maximum number of iterations reached.")
            