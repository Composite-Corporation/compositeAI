from typing import List
from openai import OpenAI
from pydantic import validator, PrivateAttr
from dotenv import load_dotenv
import textwrap
import json

from compositeai.drivers.base_driver import (
    BaseDriver,
    DriverUsage,
    DriverToolCall,
    DriverPlan,
    DriverAction, 
    DriverObservation,
)
from compositeai.tools import BaseTool


class OpenAIDriver(BaseDriver):
    _client: OpenAI = PrivateAttr()
    

    def __init__(self, **data):
        super().__init__(**data)
        load_dotenv()
        self._client = OpenAI()


    @validator("model")
    def check_model(cls, v):
        _openai_supported_models = set([
            "gpt-4-turbo", 
            "gpt-4-turbo-2024-04-09", 
            "gpt-4-turbo-preview", 
            "gpt-4-0125-preview", 
            "gpt-4-1106-preview", 
            "gpt-4", 
            "gpt-4-0613", 
            "gpt-3.5-turbo", 
            "gpt-3.5-turbo-0125", 
            "gpt-3.5-turbo-1106", 
            "gpt-3.5-turbo-0613",
        ])
        if v not in _openai_supported_models:
            raise ValueError(f"Model must be one of {_openai_supported_models}.")
        return v
    

    def _plan(
        self, 
        system_prompt: str, 
        tools: List[BaseTool],
    ) -> DriverPlan:
        tools_json_desc = [tool.get_schema().json() for tool in tools]
        system_prompt = f"""
            {system_prompt}

            YOU CAN USE THE FOLLOWING TOOLS:
            {tools_json_desc}

            WRITE A BRIEF PLAN FOR WHAT YOU SHOULD DO AT THIS POINT IN TIME:
        """
        system_prompt = textwrap.dedent(system_prompt)
        messages = [{"role": "system", "content": system_prompt}]
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        content = response.choices[0].message.content
        usage = self._openai_usage_to_driver_usage(response.usage)
        return DriverPlan(content=content, usage=usage)
    
    
    def _action(
        self, 
        system_prompt: str, 
        tools: List[BaseTool],
    ) -> DriverAction:
        system_prompt = f"""
            {system_prompt}

            USE THE GIVEN TOOLS TO EXECUTE YOUR PLAN.
        """
        system_prompt = textwrap.dedent(system_prompt)
        messages = [{"role": "system", "content": system_prompt}]
        openai_functions = [self._basetool_to_openai_fc_schema(tool) for tool in tools]
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=openai_functions,
            tool_choice="required",
        )
        tool_calls = response.choices[0].message.tool_calls 
        driver_tool_calls = []
        for tool_call in tool_calls:
            driver_tool_call = DriverToolCall(id=tool_call.id, name=tool_call.function.name, args=tool_call.function.arguments)
            driver_tool_calls.append(driver_tool_call)
        usage = self._openai_usage_to_driver_usage(response.usage)
        return DriverAction(usage=usage, tool_calls=driver_tool_calls)


    def _observe(
        self, 
        system_prompt: str,
        tools: List[BaseTool],
        action: DriverAction
    ) -> DriverObservation:
        data = []
        for tool_call in action.tool_calls:
            # Get function call info
            function_name = tool_call.name
            function_args = json.loads(tool_call.args)

            # Iterate through provided tools to check if driver_response function call matches one
            no_match_flag = True
            for tool in tools:
                # If match, run tool function on arguments for result, and append to memory
                if tool.get_schema().name == function_name:
                    no_match_flag = False
                    function_result = str(tool.func(**function_args))
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
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        content = response.choices[0].message.content
        usage = self._openai_usage_to_driver_usage(response.usage)
        return DriverObservation(content=content, usage=usage)
    
    
    def _openai_usage_to_driver_usage(self, openai_usage_obj: object) -> DriverUsage:
        return DriverUsage(
            prompt_tokens=openai_usage_obj.prompt_tokens,
            completion_tokens=openai_usage_obj.completion_tokens,
            total_tokens=openai_usage_obj.total_tokens,
        )

    
    # Helper function to convert BaseTool to OpenAI function calling schema
    def _basetool_to_openai_fc_schema(self, tool: BaseTool) -> object:
        # Get schema of tool/function
        tool_schema = tool.get_schema()

        # OpenAI function calling tool format
        openai_fc = {
            "type": "function",
            "function": {
                "name": tool_schema.name,
                "description": tool_schema.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                    },
                    "required": []
                },
            }
        }

        # Populate formatted arguments for tool
        for arg in tool_schema.arguments:
            openai_fc["function"]["parameters"]["properties"][arg.name] = {"type": self._type_conversion_openai(arg.type)}
            if arg.required:
                openai_fc["function"]["parameters"]["required"].append(arg.name)
        
        return openai_fc
    
    
    # Helper function converting python types to OpenAI function calling type format
    def _type_conversion_openai(self, type: str) -> str:
        conversions = {
            "int": "integer",
            "str": "string",
            "float": "float",
            "bool": "boolean",
        }
        return conversions[type]
    