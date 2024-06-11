from typing import List, Optional
from pydantic import BaseModel, PrivateAttr, Field
from enum import Enum
import json

from compositeai.agents.base_agent import (
    AgentOutput, 
    AgentStep, 
    AgentResult, 
    BaseAgent,
)
from compositeai.drivers.base_driver import (
    DriverInput, 
    DriverToolChoice, 
    DriverMessage,
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolMessage,
)


class NextStep(Enum):
    PLAN = 'plan'
    ACTION = 'action'
    OBSERVE = 'observe'
    OUTPUT = 'output'


class Plan(BaseModel):
    steps: List[str] = Field("List of steps to take to complete task")


class StepCheck(BaseModel):
    complete: bool = Field(description="true if the current step is complete")


class PlanAgent(BaseAgent):
    _memory: List[DriverMessage] = PrivateAttr(default=[])
    _initial_plan: List[str] = PrivateAttr(default=[])
    _current_plan_index: int = PrivateAttr(default=0)
    _next_step: NextStep = PrivateAttr(default=NextStep.PLAN)


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
                return self._observe()
            case NextStep.OUTPUT:
                return self._output()
            

    def _plan(self) -> AgentStep:
        # Generate a plan formatted as list of steps 
        plan_prompt = f"""
        WRITE A BRIEF PLAN FOR WHAT YOU SHOULD DO AT THIS POINT IN TIME.

        The output should be formatted as a JSON instance that conforms to the JSON schema below.

        As an example, for the schema {{"properties": {{"foo": {{"title": "Foo", "description": "a list of strings", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["foo"]}}
        the object {{"foo": ["bar", "baz"]}} is a well-formatted instance of the schema. The object {{"properties": {{"foo": ["bar", "baz"]}}}} is not well-formatted.

        Here is the output schema:
        ```
        {Plan.model_json_schema()}
        ```
        """
        messages = self._memory + [SystemMessage(role="system", content=plan_prompt)]
        driver_input = DriverInput(
            messages=messages,
            temperature=0.0,
            response_format="json_object",
        )
        response = self.driver.generate(input=driver_input)

        # Parse response
        plan_dict = json.loads(response.content)
        plan_list = plan_dict["steps"]

        # Add to state and set next step as ACTION
        self._initial_plan = plan_list
        self._next_step = NextStep.ACTION

        # Format plan into single string and stream result as AgentStep
        plan_str = ""
        for i, step in enumerate(plan_list, 1):
            plan_str += f"{i}. {step}\n"
        return AgentStep(content=plan_str)
    
    
    def _action(self) -> AgentStep:
        # Get current step in plan
        current_plan_step = self._initial_plan[self._current_plan_index]

        # Generate action based on the step
        system_message = f"""
        WORK ON THE CURRENT STEP ONLY (DO NOT MOVE AHEAD):

        {current_plan_step}
        """
        driver_input = DriverInput(
            messages=self._memory + [SystemMessage(role="system", content=system_message)],
            tools=self.tools,
            tool_choice=DriverToolChoice.AUTO,
            temperature=0.0,
        )
        response = self.driver.generate(input=driver_input)
        tool_calls = response.tool_calls

        # If no tools called, 
        if not tool_calls:
            # Record response in memory
            self._memory.append(AssistantMessage(role="assistant", content=response.content))
            self._next_step = NextStep.OBSERVE
            return AgentStep(content=response.content)
        
        # If tools called, go to OBSERVE step and stream tool calls as AgentStep
        else:
            tool_messages = []
            observations = ""
            for tool_call in tool_calls:
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
            self._memory.append(AssistantMessage(role="assistant", tool_calls=tool_calls))
            self._memory += tool_messages
            self._next_step = NextStep.OBSERVE

            # Return string concatenated version of condensed tool call results
            tool_observe = f"""
            {tool_calls}

            {observations}
            """
            return AgentStep(content=tool_observe)


    def _observe(self) -> AgentStep:
        # Check if step has been completed
        current_plan_step = self._initial_plan[self._current_plan_index]
        step_check_prompt = f"""
        DO YOU BELIEVE THAT THE CURRENT STEP HAS BEEN COMPLETED?

        CURRENT STEP: {current_plan_step}

        The output should be formatted as a JSON instance that conforms to the JSON schema below.

        As an example, for the schema {{"properties": {{"foo": {{"title": "Foo", "description": "a list of strings", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["foo"]}}
        the object {{"foo": ["bar", "baz"]}} is a well-formatted instance of the schema. The object {{"properties": {{"foo": ["bar", "baz"]}}}} is not well-formatted.

        Here is the output schema:
        ```
        {StepCheck.model_json_schema()}
        ```
        """
        driver_input = DriverInput(
            messages=self._memory + [SystemMessage(role="system", content=step_check_prompt)],
            temperature=0.0,
            response_format="json_object"
        )
        completed = self.driver.generate(input=driver_input)
        completed = json.loads(completed.content)["complete"]

        # If current step is completed, move on to next step
        if completed:
            self._current_plan_index += 1

            # If there are no more steps left, go to output
            self._next_step = NextStep.ACTION
            if self._current_plan_index >= len(self._initial_plan):
                self._next_step = NextStep.OUTPUT
            
            return AgentStep(content=f"Completed Task: {current_plan_step}")
        else:
            return AgentStep(content=f"Continuing Task: {current_plan_step}")


    def _output(self) -> AgentResult:
        # Check if step has been completed
        result_prompt = f"""
        GIVEN YOUR PROGRESS, PRODUCE A FINAL RESULT THAT BEST ANSWERS THE ORIGINAL USER TASK.
        """
        driver_input = DriverInput(
            messages=self._memory + [SystemMessage(role="system", content=result_prompt)],
            temperature=0.0,
        )
        response = self.driver.generate(input=driver_input)
        return AgentResult(content=response.content)