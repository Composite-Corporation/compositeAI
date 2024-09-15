from compositeai.drivers import OpenAIDriver
from compositeai.tools import GoogleSerperApiTool, WebScrapeTool
from compositeai.agents import PlanAgent


agent = PlanAgent(
    driver=OpenAIDriver(model="gpt-4o-mini", seed=1337),
    description="You are a private investigator that is good at finding information on people.",
    tools=[GoogleSerperApiTool(), WebScrapeTool()],
    max_iterations=100
)

task = """
Can you give me information on Jensen Huang?
Summarize his main achievements, and tell me about his past.
Cite your sources.
"""

for chunk in agent.execute(task, stream=True):
    print(chunk.content)
    print("\n\n\n\n\n----------------------------------------------------------------------------------------------------------\n\n\n\n\n")

print("FINAL ANSWER:\n\n")
print(chunk.content)