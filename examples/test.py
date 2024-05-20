from compositeai.drivers import OpenAIDriver
from compositeai.tools import GoogleSerperApiTool, WebScrapeTool
from compositeai.agents.agent import Agent

test_agent = Agent(
    driver=OpenAIDriver(model="gpt-4-turbo"),
    description="You are a web researcher that will find answers to my questions.",
    tools=[GoogleSerperApiTool(), WebScrapeTool()],
)

for chunk in test_agent.execute("What's the difference between the ACT paper vs digital test?", stream=True):
    print(chunk)
    print("\n")