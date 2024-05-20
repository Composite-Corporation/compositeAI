from typing import Callable

from compositeai.tools import BaseTool

def _agent_finish_tool(result: str):
    """
    Return the final result once you believe you have completed the task at hand.
    """
    return result

class _AgentFinishTool(BaseTool):
    func: Callable = _agent_finish_tool