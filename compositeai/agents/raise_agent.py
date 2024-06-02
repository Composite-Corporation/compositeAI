from typing import Generator, List, Optional, Union
from pydantic import BaseModel, PrivateAttr
import textwrap
import json

from compositeai.agents.base_agent import AgentResult, AgentFinishTool, BaseAgent
from compositeai.drivers.base_driver import DriverInput, DriverToolChoice, DriverToolCall
from compositeai.tools import BaseTool


class RAISEAgent(BaseAgent):
    _scratchpad: List[str] = PrivateAttr(default=[])
    

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
            ~~~YOU ARE AN AI AGENT GIVEN THE FOLLOWING ROLE:
            {self.description}

            ~~~SO FAR YOUR SRATCHPAD CONTAINS THE FOLLOWING:
            {self._scratchpad}

            ~~~NOW PROCEED WITH NOVEL TASKS:
        """
        return textwrap.dedent(memory)
    

    def _iterate(self) -> Generator:
        memory = self.refresh_memory()
        plan = self._plan(
            system_prompt=memory,
            tools=self.tools,
        )
        self._scratchpad.append(
           f"ALL PREVIOUS PLANS ARE COMPLETE ***** NEW PLAN: {plan}")
        yield plan

        memory = self.refresh_memory()
        actions = self._action(
            system_prompt=memory,
        )
        self._scratchpad.append(f"action(s): Calling the following tools - {actions}")
        yield actions
        
        memory = self.refresh_memory()
        observation = self._observe(
            system_prompt=memory,
            actions=actions,
        )
        self._scratchpad.append(f"observation(s): {observation}")
        yield observation
            

    def _plan(
        self, 
        system_prompt: str, 
        tools: List[BaseTool],
    ) -> str:
        tools_json_desc = [tool.get_schema().json() for tool in tools]
        system_prompt = f"""
            {system_prompt}

            YOU CAN USE THE FOLLOWING TOOLS:
            {tools_json_desc}

            WRITE A BRIEF PLAN FOR WHAT YOU SHOULD DO AT THIS POINT IN TIME:
        """
        system_prompt = textwrap.dedent(system_prompt)
        messages = [{"role": "system", "content": system_prompt}]
        driver_input = DriverInput(
            messages=messages,
        )
        response = self.driver.generate(input=driver_input)
        content = response.content
        return content
    
    
    def _action(
        self, 
        system_prompt: str, 
    ) -> List[DriverToolCall]:
        system_prompt = f"""
            {system_prompt}

            USE THE GIVEN TOOLS TO EXECUTE YOUR PLAN.
        """
        system_prompt = textwrap.dedent(system_prompt)
        messages = [{"role": "system", "content": system_prompt}]
        driver_input = DriverInput(
            messages=messages,
            tools=self.tools,
            tool_choice=DriverToolChoice.REQUIRED
        )
        response = self.driver.generate(input=driver_input)
        tool_calls = response.tool_calls
        return tool_calls


    def _observe(
        self, 
        system_prompt: str,
        actions: List[DriverToolCall]
    ):
        data = []
        for tool_call in actions:
            # Get function call info
            function_name = tool_call.name
            function_args = json.loads(tool_call.args)

            # Iterate through provided tools to check if driver_response function call matches one
            no_match_flag = True
            for tool in self.tools:
                # If match, run tool function on arguments for result, and append to memory
                if tool.get_schema().name == function_name:
                    no_match_flag = False
                    function_result = str(tool.func(**function_args))
                    if function_name == "agent_finish":
                        return AgentResult(content=function_result)
                    data.append(function_result)
            
            # If driver_response function call matches none of the given tools
            if no_match_flag:
                raise Exception("Driver called function, function call does not match any of the provided tools.")
            
        # Once data has been obtained from the results of function calls, filter for useful insights as observations
        system_prompt = f"""
            {system_prompt}

            HERE ARE IS WHAT YOU OBSERVED FROM YOUR PREVIOUS ACTION(S):
            {data}

            EXTRACT THE MOST RELEVANT DATA FROM YOUR OBSERVATIONS:
        """
        system_prompt = textwrap.dedent(system_prompt)
        messages = [{"role": "system", "content": system_prompt}]
        driver_input = DriverInput(
            messages=messages
        )
        response = self.driver.generate(input=driver_input)
        content = response.content
        return content