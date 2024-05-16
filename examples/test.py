from compositeai.drivers import OpenAIDriver
from compositeai.tools import GoogleSerperApiTool, WebScrapeTool
from compositeai.agents.agent import Agent

test_agent = Agent(
    driver=OpenAIDriver(model="gpt-3.5-turbo-1106"),
    description="You are a private investigator that can research details about people off of the internet",
    tools=[GoogleSerperApiTool(), WebScrapeTool()],
)

test_agent.execute("Please find information on Jody Li, University of Washington")

