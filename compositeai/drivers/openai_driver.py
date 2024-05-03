from typing import Optional, List
import json
from openai import OpenAI

from compositeai.drivers.base_driver import (
    BaseDriver,
    DriverToolCall,
    DriverUsage,
    DriverResponse,
)
from compositeai.tools import BaseTool

class OpenAIDriver(BaseDriver):
    client = OpenAI()
    openai_supported_models = set([
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


    def __init__(self, model: str) -> None:
        if model not in self.openai_supported_models:
            raise ValueError("OpenAI model not supported: " + model)
        super().__init__(model)


    # Override BaseWrapper chat
    def _iterate(
        self, 
        messages: List,
        tools: Optional[List[BaseTool]],
    ) -> DriverResponse:
        # Convert tools to OpenAI function call schema
        openai_tools = self._toolschema_to_openai_tools(tools)

        # OpenAI API call response
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=openai_tools,
        ).choices[0].message

        # Extract response data
        role = response.role
        content = response.content
        driver_tool_call = None
        if response.tool_calls:
            tool_call = response.tool_calls[0]
            driver_tool_call = DriverToolCall(
                call_id=tool_call.id,
                tool_name=tool_call.name,
                arguments=json.loads(tool_call.function.arguments)
            )
        driver_usage = DriverUsage(
            prompt_tokens=response.usage.prompt_tokens, 
            completion_tokens=response.usage.completion_tokens, 
            total_tokens=response.usage.total_tokens
        )

        # Return DriverResponse object
        return DriverResponse(
            role=role,
            content=content,
            tool_call=driver_tool_call,
            usage=driver_usage,
        )


    # Helper function convert tool schema to OpenAI function calling schema
    def _toolschema_to_openai_tools(self, tools: Optional[List[BaseTool]]):
        if not tools:
            return None
        openai_tools = []
        for tool in tools:
            tool_schema = tool.get_schema()
            openai_tool = {
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

            for arg in tool_schema.arguments:
                openai_tool["function"]["parameters"]["properties"][arg.name] = {"type": self._type_conversion_openai(arg.type)}
                if arg.required:
                    openai_tool["function"]["parameters"]["required"].append(arg.name)

            openai_tools.append(openai_tool)

        return openai_tools
    
    def _type_conversion_openai(self, type: str) -> str:
        conversions = {
            "int": "integer",
            "str": "string",
            "float": "float",
            "bool": "boolean",
        }
        return conversions[type]
