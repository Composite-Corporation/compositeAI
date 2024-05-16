from typing import Optional, List
import json
from openai import OpenAI

from compositeai.drivers.base_driver import BaseDriver
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


    # Override BaseDriver
    def _iterate(
        self, 
        messages: List,
        tools: Optional[List[BaseTool]],
    ):
        # Convert tools to OpenAI function call schema
        openai_tools = self._toolschema_to_openai_tools(tools)

        # OpenAI API call response
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=openai_tools,
        )

        # Return response object
        return response


    # Helper function convert tool schema to OpenAI function calling schema
    def _toolschema_to_openai_tools(self, tools: Optional[List[BaseTool]]):
        # If no tools passed to driver, return None object
        if not tools:
            return None
        
        # Populate OpenAI function calling list of formatted tools
        openai_tools = []
        for tool in tools:
            # Get schema of tool/function
            tool_schema = tool.get_schema()

            # OpenAI function calling tool format
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

            # Populate formatted arguments for tool
            for arg in tool_schema.arguments:
                openai_tool["function"]["parameters"]["properties"][arg.name] = {"type": self._type_conversion_openai(arg.type)}
                if arg.required:
                    openai_tool["function"]["parameters"]["required"].append(arg.name)

            # Add final formatted function to list
            openai_tools.append(openai_tool)

        return openai_tools
    
    # Helper function converting python types to OpenAI function calling type format
    def _type_conversion_openai(self, type: str) -> str:
        conversions = {
            "int": "integer",
            "str": "string",
            "float": "float",
            "bool": "boolean",
        }
        return conversions[type]
