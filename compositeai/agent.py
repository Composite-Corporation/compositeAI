from typing import List, Optional, Any

from compositeai.drivers.base_driver import BaseDriver, DriverResponse
from tools import BaseTool

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
        system_message = {"role": "system", "content": self.description},
        user_message = {"role": "user", "content": task},
        self.memory.append(system_message)
        self.memory.append(user_message)

        for _ in range(self.max_iterations):
            driver_response = self.driver._iterate(messages=self.memory, tools=self.tools)
            if not driver_response.tool_call:
                driver_message = driver_response.message
                self.memory.append({"role": driver_message.role, "content": driver_message.content})
            else:
                pass
            