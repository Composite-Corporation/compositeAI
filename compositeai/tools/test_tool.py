from typing import Any

from compositeai.tools import BaseTool


class TestTool(BaseTool):
    name: str = "test_tool"
    description: str = "Used to mirror a list back to the user."


    def __init__(self, **data):
        super().__init__(**data)


    def func(self, sample: list) -> Any:
        return sample