from typing import List, Optional, Any
import json

from compositeai.drivers.base_driver import BaseDriver
from compositeai.tools import BaseTool

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
        self.tools = tools
        self.max_iterations = max_iterations
        self.memory = []

    def execute(self, task: str, input: Optional[Any] = None):
        self.memory.clear()
        if input:
            task = f"""
            {task}

            Here is what you are given:
            {input}
            """
        system_message = {"role": "system", "content": self.description}
        user_message = {"role": "user", "content": task}
        self.memory.append(system_message)
        self.memory.append(user_message)

        for _ in range(self.max_iterations):
            driver_response = self.driver._iterate(messages=self.memory, tools=self.tools)
            print(driver_response)
            print("\n")
            usage = driver_response.usage
            message = driver_response.choices[0].message
            self.memory.append(message)
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_call_id = tool_call.id
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    for tool in self.tools:
                        if tool.get_schema().name == function_name:
                            function_result = tool.func(**function_args)
                    self.memory.append({"role": "tool", "content": str(function_result), "tool_call_id": tool_call_id})
        print(self.memory)
            