from typing import List, Optional, Any
import json

from compositeai.drivers.base_driver import BaseDriver
from compositeai.tools import BaseTool

class _AgentFinish(BaseTool):
    def __init__(self) -> None:
        # Agent finish function
        def _finish():
            """
            Only use when you believe you have completed the task at hand.
            """
            return
        
        super().__init__(func=_finish)

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

    def execute(self, task: str, input: Optional[Any] = None):
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
        agent_finish = False
        for _ in range(self.max_iterations):
            # Obtain response from driver iteration and append to memory
            driver_response = self.driver._iterate(messages=self.memory, tools=self.tools)
            message = driver_response.choices[0].message
            self.memory.append(message)
            print(driver_response)
            print("\n")

            # Ussage data 
            usage = driver_response.usage

            # If function called, loop through each, obtain schema, and append to memory
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    # Get function call info
                    tool_call_id = tool_call.id
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    # Get function call schema and append to memory
                    for tool in self.tools:
                        if tool.get_schema().name == function_name:
                            function_result = tool.func(**function_args)
                    self.memory.append({"role": "tool", "content": str(function_result), "tool_call_id": tool_call_id})

                    # If function call is agent finish, set flag to True
                    if function_name == "_finish":
                        agent_finish = True
                        break

            # If agent_finish flag, break
            if agent_finish:
                break

        print(self.memory)
            