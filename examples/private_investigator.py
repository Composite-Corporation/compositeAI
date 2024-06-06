from compositeai.drivers import OpenAIDriver
from compositeai.tools import GoogleSerperApiTool, WebScrapeTool
from compositeai.agents import RAISEAgent


agent = RAISEAgent(
    driver=OpenAIDriver(model="gpt-4o"),
    description="You are a private investigator that is good at finding information on people.",
    tools=[GoogleSerperApiTool(), WebScrapeTool()],
)

task = """
I need to find info on Joseph Ros, a partner at Entrepreneur First.
What is his background? How much has he invested? What companies has he invested in?
Cite sources for each of your answers.
"""

for chunk in agent.execute(task, stream=True):
    print(chunk.content)
    print("\n\n\n\n\n----------------------------------------------------------------------------------------------------------\n\n\n\n\n")

print("FINAL ANSWER:\n\n")
print(chunk.content)