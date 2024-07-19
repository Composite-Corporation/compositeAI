from compositeai.drivers import OpenAIDriver
from compositeai.tools import GoogleSerperApiTool, WebScrapeTool
from compositeai.agents import PlanAgent


description = """
You are a financial news analysis AI tasked with synthesizing the latest news relevant to a specific investment portfolio.
"""

task = """
**Portfolio:** [Tesla, NVIDIA]

**Risks to Monitor:** [Supply chain, China, interest rates]

**Ignore:** [Elon Musk’s compensation package]

**Instructions:**
1. Scan financial news sources for relevant articles in May 2024, focusing on the listed risks.
2. Evaluate newsworthiness by considering immediate impact, risk magnitude, and long-term effects. Summarize each article in max 15 words with the source (name only) and date.
3. When you have searched enough, create a summary report with the most newsworthy articles.
"""

agent = PlanAgent(
    driver=OpenAIDriver(model="gpt-4o"),
    description=description,
    tools=[GoogleSerperApiTool(), WebScrapeTool()],
)

for chunk in agent.execute(task, stream=True):
    print(chunk.content)
    print("\n\n\n\n\n----------------------------------------------------------------------------------------------------------\n\n\n\n\n")

print("FINAL ANSWER:\n\n")
print(chunk.content)