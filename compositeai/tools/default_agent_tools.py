from compositeai.tools import BaseTool


class _AgentFinishTool(BaseTool):
    name: str = "agent_finish"
    description: str = "Return the final result once you believe you have completed the task at hand"

    def func(self, result: str) -> str:
        return result