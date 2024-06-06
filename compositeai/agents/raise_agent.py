from typing import Union, List, Optional
from pydantic import BaseModel, PrivateAttr
from enum import Enum
import textwrap
import json

from compositeai.agents.base_agent import AgentOutput, AgentStep, AgentResult, AgentExecution, BaseAgent
from compositeai.drivers.base_driver import DriverInput, DriverToolChoice, DriverToolCall
from compositeai.tools import BaseTool


class NextStep(Enum):
    PLAN = 'plan'
    ACTION = 'action'
    OBSERVE = 'observe'


class RAISEAgent(BaseAgent):
    _scratchpad: List[str] = PrivateAttr(default=[])
    _next_step: NextStep = PrivateAttr(default=NextStep.PLAN)
    _next_actions: List[DriverToolCall] = PrivateAttr(default=[])
    

    def initialize(self, task: str, input: Optional[str] = None) -> None:
        # Add initial data to conversation history and agent scratchpad
        self._scratchpad.append(f"task: {task}")
        if input:
            self._scratchpad.append(input)
    

    def iterate(self) -> AgentOutput:
        # Regenerate memory system
        memory = self.refresh_memory()

        # Run iteration based on next step
        match self._next_step:
            case NextStep.PLAN:
                return self._plan(
                    system_prompt=memory,
                    tools=self.tools,
                )
            case NextStep.ACTION:
                return self._action(
                    system_prompt=memory,
                )
            case NextStep.OBSERVE:
                return self._observe(
                    system_prompt=memory,
                    actions=self._next_actions,
                )

    
    def refresh_memory(self):
        memory = f"""
            ~~~YOU ARE AN AI AGENT GIVEN THE FOLLOWING ROLE:
            {self.description}

            ~~~SO FAR YOUR SRATCHPAD CONTAINS THE FOLLOWING:
            {self._scratchpad}

            ~~~NOW PROCEED WITH NOVEL TASKS:
        """
        return textwrap.dedent(memory)
            

    def _plan(
        self, 
        system_prompt: str, 
        tools: List[BaseTool],
    ) -> AgentStep:
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
        plan = response.content
        self._scratchpad.append(f"ALL PREVIOUS PLANS ARE COMPLETE ***** NEW PLAN: {plan}")
        self._next_step = NextStep.ACTION
        return AgentStep(content=plan)
    
    
    def _action(
        self, 
        system_prompt: str, 
    ) -> AgentStep:
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
        self._scratchpad.append(f"action(s): Calling the following tools - {tool_calls}")
        self._next_step = NextStep.OBSERVE
        self._next_actions = tool_calls
        return AgentStep(content=tool_calls)


    def _observe(
        self, 
        system_prompt: str,
        actions: List[DriverToolCall]
    ) -> Union[AgentStep, AgentResult]:
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
        observation = response.content
        self._scratchpad.append(f"observation(s): {observation}")
        self._next_step = NextStep.PLAN
        return AgentStep(content=observation)