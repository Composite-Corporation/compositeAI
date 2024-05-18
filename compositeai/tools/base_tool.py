import inspect
from typing import List, Callable
from pydantic import BaseModel, validator


class ParamDesc(BaseModel):
    name: str
    type: str
    required: bool


class ToolSchema(BaseModel):
    name: str
    description: str
    arguments: List[ParamDesc]

    
class BaseTool(BaseModel):
    func: Callable
    
    @validator("func")
    def check_func(cls, v):
        if not v.__name__:
            raise ValueError("Function of tool must have a name.")
        if not v.__doc__:
            raise ValueError("Function of tool must have a docstring.")
        
    def get_schema(self):
        # Get function details
        func_name = self.func.__name__
        func_doc = self.func.__doc__
        signature = inspect.signature(self.func)
        arguments = []
        for name, param in signature.parameters.items():
            required = True if param.default is inspect.Parameter.empty else False
            arguments.append(ParamDesc(name=name, type=param.annotation.__name__, required=required))
        
        return ToolSchema(name=func_name, description=func_doc, arguments=arguments)