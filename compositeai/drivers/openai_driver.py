from typing import List
from openai import OpenAI
from pydantic import validator, PrivateAttr
from dotenv import load_dotenv

from compositeai.drivers.base_driver import (
    BaseDriver,
    DriverUsage,
    DriverToolCall,
    DriverMessage,
    DriverResponse,
    DriverInput,
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
            "gpt-4o"
        ])
        if v not in _openai_supported_models:
            raise ValueError(f"Model must be one of {_openai_supported_models}.")
        return v
    
    
    def generate(
        self,
        input: DriverInput,
    ) -> DriverResponse:
        messages = input.messages
        messages = self._messages_driver_to_openai(messages)
        max_tokens = input.max_tokens
        temperature = input.temperature
        tools = input.tools
        if tools:
            tools = self._fc_schema_basetools_to_openai(tools)
        tool_choice = input.tool_choice
        if tool_choice:
            tool_choice = tool_choice.value

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
        )

        content = response.choices[0].message.content
        tool_calls = response.choices[0].message.tool_calls 
        driver_tool_calls = None
        if tool_calls:
            driver_tool_calls = []
            for tool_call in tool_calls:
                driver_tool_call = DriverToolCall(id=tool_call.id, name=tool_call.function.name, args=tool_call.function.arguments)
                driver_tool_calls.append(driver_tool_call)
        usage = self._usage_openai_to_driver(response.usage)

        return DriverResponse(content=content, tool_calls=driver_tool_calls, usage=usage)


    def _messages_driver_to_openai(self, messages: List[DriverMessage]) -> List[object]:
        return [{"role": message.role.value, "content": message.content} for message in messages]
    

    # Helper function to convert BaseTool to OpenAI function calling schema
    def _fc_schema_basetools_to_openai(self, tools: List[BaseTool]) -> List[object]:
        openai_fcs = []
        for tool in tools:
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
            
            openai_fcs.append(openai_fc)
        return openai_fcs
    

    # Helper function converting python types to OpenAI function calling type format
    def _type_conversion_openai(self, type: str) -> str:
        conversions = {
            "int": "integer",
            "str": "string",
            "float": "float",
            "bool": "boolean",
        }
        return conversions[type]
    

    def _usage_openai_to_driver(self, openai_usage_obj: object) -> DriverUsage:
        return DriverUsage(
            prompt_tokens=openai_usage_obj.prompt_tokens,
            completion_tokens=openai_usage_obj.completion_tokens,
            total_tokens=openai_usage_obj.total_tokens,
        )
    