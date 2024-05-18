from typing import Optional, List
from openai import OpenAI
from pydantic import validator, PrivateAttr

from compositeai.drivers.base_driver import BaseDriver
from compositeai.tools import BaseTool

class OpenAIDriver(BaseDriver):
    # Class specific variables (not part of Pydantic model)
    _client: OpenAI = PrivateAttr()
    
    def __init__(self, **data):
        super().__init__(**data)
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

    # Override BaseDriver
    def _iterate(
        self, 
        messages: List,
        tools: Optional[List[BaseTool]],
    ):
        # Initialize empty list of OpenAI function call schemas
        openai_fcs = []

        # Convert given tools to OpenAI function calling format
        for tool in tools:
            tool_openai_fc = self._basetool_to_openai_fc_schema(tool)
            openai_fcs.append(tool_openai_fc)

        # OpenAI API call response
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=openai_fcs,
        )

        # Return response object
        return response

    
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
