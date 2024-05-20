import inspect
import re
from typing import List, Any
from abc import abstractmethod
from pydantic import BaseModel, validator, Field


class ParamDesc(BaseModel):
    name: str
    type: str
    required: bool


class ToolSchema(BaseModel):
    name: str
    description: str
    arguments: List[ParamDesc]

    
class BaseTool(BaseModel):
    name: str = Field("Name of the tool")
    description: str = Field("Description of what the tool does")

    @validator("name")
    def check_name(cls, v):
        # Satisfy OpenAI function calling requirements
        if not re.fullmatch("^[a-zA-Z0-9_-]+$'.", v):
            raise ValueError("Tool names must match pattern: '^[a-zA-Z0-9_-]+$'.")
        return v

    @abstractmethod
    def func(self, *args: Any, **kwargs: Any) -> Any:
        """Function of tool to be implemented by subclass"""
        raise NotImplementedError("Function of tool must be implemented by subclass")

    def get_schema(self):
        """Get schema of defined function for tool"""
        signature = inspect.signature(self.func)
        arguments = []
        for name, param in signature.parameters.items():
            required = True if param.default is inspect.Parameter.empty else False
            arguments.append(ParamDesc(name=name, type=param.annotation.__name__, required=required))
        return ToolSchema(name=self.name, description=self.description, arguments=arguments)