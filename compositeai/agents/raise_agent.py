from typing import Union, List, Optional
from pydantic import BaseModel, ConfigDict, PrivateAttr
from enum import Enum
import json

from compositeai.agents.base_agent import (
    AgentOutput, 
    AgentStep, 
    AgentResult, 
    AgentExecution, 
    BaseAgent,
)
from compositeai.drivers.base_driver import (
    DriverInput, 
    DriverToolChoice, 
    DriverToolCall,
    DriverMessage,
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolMessage,
)
from compositeai.tools import (
    BaseTool,
)


class NextStep(Enum):
    PLAN = 'plan'
    ACTION = 'action'
    OBSERVE = 'observe'


class RAISEAgent(BaseAgent):
    _memory: List[DriverMessage] = PrivateAttr(default=[])
    _next_step: NextStep = PrivateAttr(default=NextStep.PLAN)
    _next_actions: List[DriverToolCall] = PrivateAttr(default=[])


    def __init__(self, **data):
        # Superclass init
        super().__init__(**data)
        # Add agent description as system message for LLM
        self._memory.append(
            SystemMessage(
                role="system",
                content=self.description,
            ),
        )


    def exec_init(self, task: str, input: Optional[str] = None) -> None:
        # Add additional input to task, if given
        if input:
            task = f"""
            {task}

            SOME INFO YOU ARE GIVEN TO START THE TASK:
            {input}
            """

        # Add task to LLM as a user message
        self._memory.append(UserMessage(role="user", content=task))
        

    def iterate(self) -> AgentOutput:
        # Run iteration based on next step
        match self._next_step:
            case NextStep.PLAN:
                return self._plan()
            case NextStep.ACTION:
                return self._action()
            case NextStep.OBSERVE:
                return self._observe(actions=self._next_actions)
            

    def _plan(self) -> AgentStep:
        plan_prompt = "WRITE A BRIEF PLAN FOR WHAT YOU SHOULD DO AT THIS POINT IN TIME:"
        messages = self._memory + [SystemMessage(role="system", content=plan_prompt)]
        driver_input = DriverInput(
            messages=messages,
            temperature=0.0,
        )
        response = self.driver.generate(input=driver_input)
        plan = response.content
        self._memory.append(AssistantMessage(role="assistant", content=plan))
        self._next_step = NextStep.ACTION
        return AgentStep(content=plan)
    
    
    def _action(self) -> AgentStep:
        driver_input = DriverInput(
            messages=self._memory,
            tools=self.tools,
            tool_choice=DriverToolChoice.AUTO,
            temperature=0.0,
        )
        response = self.driver.generate(input=driver_input)
        tool_calls = response.tool_calls

        # If no tools called, skip observation step and directly record response to memory
        if not tool_calls:
            self._next_step = NextStep.PLAN
            self._memory.append(AssistantMessage(role="assistant", content=response.content))
            return AgentStep(content=response.content)
        else:
            self._next_step = NextStep.OBSERVE
            self._next_actions = tool_calls
            return AgentStep(content=tool_calls)


    def _observe(self, actions: List[DriverToolCall]) -> Union[AgentStep, AgentResult]:
        tool_messages = []
        observations = ""
        for tool_call in actions:
            # Get function call info
            function_name = tool_call.name
            function_args = json.loads(tool_call.args)
            tool_call_id = tool_call.id

            # Iterate through provided tools to check if driver_response function call matches one
            no_match_flag = True
            for tool in self.tools:
                # If match, run tool function on arguments for result, and append to memory
                if tool.get_schema().name == function_name:
                    no_match_flag = False
                    function_result = str(tool.func(**function_args))

                    # If agent_finish called, return
                    if function_name == "agent_finish":
                        return AgentResult(content=function_result)
                    
                    # Otherwise, condense tool call result using LLM
                    condense_prompt = f"""
                    EXTRACT THE MOST RELAVANT INFO FROM THE FOLLOWING:

                    {function_result}
                    """
                    condense_message = SystemMessage(role="system", content=condense_prompt)
                    driver_input = DriverInput(
                        messages=self._memory + [condense_message],
                        temperature=0.0,
                    )
                    response = self.driver.generate(input=driver_input)
                    observation = response.content

                    # Put condensed result into tool message
                    tool_message = ToolMessage(
                        role="tool", 
                        content=observation,
                        tool_call_id=tool_call_id,
                    )
                    tool_messages.append(tool_message)

                    # Add to overall observations
                    observations += "\n\n" + observation
            
            # If driver_response function call matches none of the given tools
            if no_match_flag:
                raise Exception("Driver called function, function call does not match any of the provided tools.")
            
        # Once tool messages has been obtained from the results of function calls, add to memory
        self._memory.append(AssistantMessage(role="assistant", tool_calls=actions))
        self._memory += tool_messages
        self._next_step = NextStep.PLAN

        # Return string concatenated version of condensed tool call results
        return AgentStep(content=observations)