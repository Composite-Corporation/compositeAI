<div align="center">

![Composite.ai Logo](./docs/logo-full.png)

**Composite.ai** was developed to make it as easy as possible for developers to build advanced AI agent systems while sacrificing as little customizability and versatility as possible. Using our framework, you should be able to construct AI agent teams that can complete complicated tasks and define rules for their interactions in just a few lines of code.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

</div>

## Getting Started

To get started with Composite.ai, follow these simple steps:

### Installation

```shell
pip install compositeai
```

### Set Up Your Agent

1. Choose the provider and model LLM that will power your agent's "driver".
2. Describe the role and purpose of your agent.
3. Choose built-in tools or define your own for the agent to use.
4. Have the agent begin a task that you describe.

```python
from compositeai.drivers import OpenAIDriver
from compositeai.tools import GoogleSerperApiTool, WebScrapeTool
from compositeai.agents import RAISEAgent


agent = RAISEAgent(
    driver=OpenAIDriver(model="gpt-4o"),
    description="You are a weatherman and can tell me the weather for any location.",
    tools=[GoogleSerperApiTool(), WebScrapeTool()],
)

task = """
What's the weather going to be today in Cambridge, MA?
"""

for chunk in agent.execute(task, stream=True):
    print(chunk)
    print("\n\n\n\n\n----------------------------------------------------------------------------------------------------------\n\n\n\n\n")

print("FINAL ANSWER:\n\n")
print(chunk.content)
```

## Key Features

### Currently Implemented

- **Role-Based Agent Design**: Customize agents with specific roles, goals, and tools.
- **Developer-Defined Agent Delegation**: The developer can specify exactly how teams of multiple AI agents should be interacting with each other to complete more complex tasks using a rule-based system.
- **Streamable Agent Thoughts and Actions**: For transparency within multi-agent systems, developers will be able to access records of agent planning, thoughts, memory, and actions taken.

### Planned
- **Works with multiple LLMs (including open source)**

## Examples

- [Private Investigator Agent](https://github.com/Composite-Corporation/compositeAI/blob/main/examples/private_investigator.py)
- [Financial News Analyst Agent](https://github.com/Composite-Corporation/compositeAI/blob/main/examples/stock_analysis.py)

## License

Composite.ai is released under the MIT License.