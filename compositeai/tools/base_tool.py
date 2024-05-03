import inspect
from typing import List
from pydantic import BaseModel

class ParamDesc(BaseModel):
    name: str
    type: str
    required: bool

class ToolSchema(BaseModel):
    name: str
    description: str
    arguments: List[ParamDesc]

class BaseTool():
    def __init__(self, func: callable) -> None:
        # Get function details
        func_name = func.__name__
        func_doc = func.__doc__
        signature = inspect.signature(func)
        arguments = []
        for name, param in signature.parameters.items():
            required = True if param.default is inspect.Parameter.empty else False
            arguments.append(ParamDesc(name=name, type=param.annotation.__name__, required=required))

        # Validation 
        if not func_name:
            raise ValueError("Function of tool must have a name.")
        if not func_doc:
            raise ValueError("Function of tool must have a docstring.")

        # Set function description object
        self.tool = ToolSchema(name=func_name, description=func_doc, arguments=arguments)

    def get_schema(self):
        return self.tool