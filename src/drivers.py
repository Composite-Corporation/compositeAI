from abc import ABC, abstractmethod
from typing import Optional, List, Dict
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import json

from src.tools import BaseTool

load_dotenv()

class DriverMessage(BaseModel):
    role: str
    content: str | None

class DriverUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class DriverToolCall(BaseModel):
    call_id: str
    tool_name: str
    arguments: Dict

class DriverResponse(BaseModel):
    message: DriverMessage
    usage: DriverUsage
    tool_call: DriverToolCall | None


class BaseDriver(ABC):
    def __init__(self, model: str) -> None:
       self.model = model

    @abstractmethod
    def _iterate(
        self, 
        messages: List,
        tools: Optional[List[BaseTool]],
    ) -> DriverResponse:
        raise NotImplementedError("BaseDriver should not be utilized.")
    

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
        openai_tools = self._toolschema_to_openai_tools(tools)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=openai_tools,
        ) 
        message = response.choices[0].message
        usage = response.usage
        driver_message = DriverMessage(role=message.role, content=message.content)
        driver_usage = DriverUsage(prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens, total_tokens=usage.total_tokens)
        driver_tool_call = None
        if message.tool_calls:
            tool_call = message.tool_calls[0]
            arguments = json.loads(tool_call.function.arguments)
            driver_tool_call = DriverToolCall(call_id=tool_call.id, tool_name=tool_call.function.name, arguments=arguments)
        return DriverResponse(message=driver_message, usage=driver_usage, tool_call=driver_tool_call)

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
