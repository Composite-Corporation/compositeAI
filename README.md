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
```

### Example Response

```
To gather comprehensive information on Jensen Huang, I will:

1. Conduct a Google search to find reliable sources that provide detailed information about Jensen Huang's achievements and past.
2. Scrape relevant websites to extract detailed information.
3. Summarize the findings and cite the sources.

Let's start with the Google search.





----------------------------------------------------------------------------------------------------------





[DriverToolCall(id='call_h0krsJzheyfjXp5ZXtKTqvb2', name='google_search', args='{"query": "Jensen Huang biography"}'), DriverToolCall(id='call_7BGsOeDmI58lVT4uhioZXzPN', name='google_search', args='{"query": "Jensen Huang achievements"}')]





----------------------------------------------------------------------------------------------------------





### Jensen Huang: Biography and Achievements

#### Biography:
- **Full Name:** Jen-Hsun "Jensen" Huang
- **Date of Birth:** February 17, 1963
- **Place of Birth:** Tainan, Taiwan
- **Early Life:**
  - Moved to Thailand shortly after birth.
  - Due to the dangers caused by the Vietnam War, his family moved to the United States.
  - Worked as a busboy at Denny's during his early years.
- **Education:**
  - Studied at Oregon State University.
  - Received a master's degree in electrical engineering from Stanford University in 1992.

#### Career:
- **NVIDIA:**
  - **Co-founded NVIDIA:** In 1993, along with Chris Malachowsky and Curtis Priem.
  - **Role:** President, CEO, and a member of the board of directors since its inception.
  - **Achievements:**
    - Pioneered the development of the Graphics Processing Unit (GPU).
    - Led NVIDIA to become a leader in accelerated computing and AI technologies.
    - Under his leadership, NVIDIA has expanded into various fields including gaming, scientific computing, and self-driving cars.

#### Achievements and Recognitions:
- **Awards and Honors:**
  - Named the world's best CEO by Fortune, The Economist, and Brand Finance.
  - Listed as one of TIME magazine's 100 most influential people.
  - Received the Distinguished Lifetime Achievement Award by Asian American Engineer of the Year.
  - Awarded the Semiconductor Industry's top honor.
- **Innovations:**
  - Introduced the concept of "Huang's Law," which predicts the rapid advancement of GPUs.
  - Played a significant role in the development of AI and machine learning technologies.
- **Net Worth:**
  - Owns approximately 3% of NVIDIA.
  - Net worth surged to $90 billion due to the company's stock performance.

#### Sources:
1. [Wikipedia - Jensen Huang](https://en.wikipedia.org/wiki/Jensen_Huang)
2. [NVIDIA Newsroom - Jensen Huang](http://nvidianews.nvidia.com/bios/jensen-huang)
3. [Britannica - Jensen Huang](https://www.britannica.com/biography/Jensen-Huang)
4. [Forbes - Jensen Huang](https://www.forbes.com/profile/jensen-huang-1/)
5. [The Immigrant Learning Center - Jensen Huang](https://www.ilctr.org/entrepreneur-hof/jen-hsun-huang/)
6. [Stanford University School of Engineering - Jensen Huang](https://engineering.stanford.edu/about/heroes/2018-heroes/jensen-huang)
7. [NVIDIA Blog - Lifetime Achievement Award](https://blogs.nvidia.com/blog/ceo-jensen-huang-awarded-lifetime-achievement-asian-american-engineer/)
8. [Semiconductors.org - Jensen Huang](https://www.semiconductors.org/nvidia-founder-and-ceo-jensen-huang-to-receive-semiconductor-industrys-top-honor/)
9. [Medium - Jensen Huang's 30-Year CEO Journey](https://medium.com/@techheadings/how-jensen-huang-became-the-30-year-ceo-of-nvidia-469bd5406b46)





----------------------------------------------------------------------------------------------------------





The task of gathering comprehensive information on Jensen Huang has been completed. The information includes his biography, career achievements, and recognitions, along with citations from reliable sources.

### Plan:
1. Review the gathered information to ensure completeness and accuracy.
2. Summarize the findings in a concise manner.
3. Use the `agent_finish` tool to provide the final result.

Let's proceed with summarizing the findings and providing the final result.

### Summary of Findings:

#### Biography:
- **Full Name:** Jen-Hsun "Jensen" Huang
- **Date of Birth:** February 17, 1963
- **Place of Birth:** Tainan, Taiwan
- **Early Life:** Moved to Thailand shortly after birth, then to the United States due to the Vietnam War. Worked as a busboy at Denny's during his early years.
- **Education:** Studied at Oregon State University and received a master's degree in electrical engineering from Stanford University in 1992.

#### Career:
- **NVIDIA:**
  - **Co-founded NVIDIA:** In 1993, along with Chris Malachowsky and Curtis Priem.
  - **Role:** President, CEO, and a member of the board of directors since its inception.
  - **Achievements:** Pioneered the development of the Graphics Processing Unit (GPU), led NVIDIA to become a leader in accelerated computing and AI technologies, and expanded into various fields including gaming, scientific computing, and self-driving cars.

#### Achievements and Recognitions:
- **Awards and Honors:**
  - Named the world's best CEO by Fortune, The Economist, and Brand Finance.
  - Listed as one of TIME magazine's 100 most influential people.
  - Received the Distinguished Lifetime Achievement Award by Asian American Engineer of the Year.
  - Awarded the Semiconductor Industry's top honor.
- **Innovations:**
  - Introduced the concept of "Huang's Law," predicting the rapid advancement of GPUs.
  - Played a significant role in the development of AI and machine learning technologies.
- **Net Worth:**
  - Owns approximately 3% of NVIDIA.
  - Net worth surged to $90 billion due to the company's stock performance.

#### Sources:
1. [Wikipedia - Jensen Huang](https://en.wikipedia.org/wiki/Jensen_Huang)
2. [NVIDIA Newsroom - Jensen Huang](http://nvidianews.nvidia.com/bios/jensen-huang)
3. [Britannica - Jensen Huang](https://www.britannica.com/biography/Jensen-Huang)
4. [Forbes - Jensen Huang](https://www.forbes.com/profile/jensen-huang-1/)
5. [The Immigrant Learning Center - Jensen Huang](https://www.ilctr.org/entrepreneur-hof/jen-hsun-huang/)
6. [Stanford University School of Engineering - Jensen Huang](https://engineering.stanford.edu/about/heroes/2018-heroes/jensen-huang)
7. [NVIDIA Blog - Lifetime Achievement Award](https://blogs.nvidia.com/blog/ceo-jensen-huang-awarded-lifetime-achievement-asian-american-engineer/)
8. [Semiconductors.org - Jensen Huang](https://www.semiconductors.org/nvidia-founder-and-ceo-jensen-huang-to-receive-semiconductor-industrys-top-honor/)
9. [Medium - Jensen Huang's 30-Year CEO Journey](https://medium.com/@techheadings/how-jensen-huang-became-the-30-year-ceo-of-nvidia-469bd5406b46)

### Final Result:
I have gathered and summarized comprehensive information on Jensen Huang, including his biography, career achievements, and recognitions. The information is cited from reliable sources.

Using the `agent_finish` tool to provide the final result.







----------------------------------------------------------------------------------------------------------





[DriverToolCall(id='call_q8xzhvlJKjaVe73edtBrZntc', name='agent_finish', args='{"result":"I have gathered and summarized comprehensive information on Jensen Huang, including his biography, career achievements, and recognitions. The information is cited from reliable sources.\\n\\n### Summary of Findings:\\n\\n#### Biography:\\n- **Full Name:** Jen-Hsun \\"Jensen\\" Huang\\n- **Date of Birth:** February 17, 1963\\n- **Place of Birth:** Tainan, Taiwan\\n- **Early Life:** Moved to Thailand shortly after birth, then to the United States due to the Vietnam War. Worked as a busboy at Denny\'s during his early years.\\n- **Education:** Studied at Oregon State University and received a master\'s degree in electrical engineering from Stanford University in 1992.\\n\\n#### Career:\\n- **NVIDIA:**\\n  - **Co-founded NVIDIA:** In 1993, along with Chris Malachowsky and Curtis Priem.\\n  - **Role:** President, CEO, and a member of the board of directors since its inception.\\n  - **Achievements:** Pioneered the development of the Graphics Processing Unit (GPU), led NVIDIA to become a leader in accelerated computing and AI technologies, and expanded into various fields including gaming, scientific computing, and self-driving cars.\\n\\n#### Achievements and Recognitions:\\n- **Awards and Honors:**\\n  - Named the world\'s best CEO by Fortune, The Economist, and Brand Finance.\\n  - Listed as one of TIME magazine\'s 100 most influential people.\\n  - Received the Distinguished Lifetime Achievement Award by Asian American Engineer of the Year.\\n  - Awarded the Semiconductor Industry\'s top honor.\\n- **Innovations:**\\n  - Introduced the concept of \\"Huang\'s Law,\\" predicting the rapid advancement of GPUs.\\n  - Played a significant role in the development of AI and machine learning technologies.\\n- **Net Worth:**\\n  - Owns approximately 3% of NVIDIA.\\n  - Net worth surged to $90 billion due to the company\'s stock performance.\\n\\n#### Sources:\\n1. [Wikipedia - Jensen Huang](https://en.wikipedia.org/wiki/Jensen_Huang)\\n2. [NVIDIA Newsroom - Jensen Huang](http://nvidianews.nvidia.com/bios/jensen-huang)\\n3. [Britannica - Jensen Huang](https://www.britannica.com/biography/Jensen-Huang)\\n4. [Forbes - Jensen Huang](https://www.forbes.com/profile/jensen-huang-1/)\\n5. [The Immigrant Learning Center - Jensen Huang](https://www.ilctr.org/entrepreneur-hof/jen-hsun-huang/)\\n6. [Stanford University School of Engineering - Jensen Huang](https://engineering.stanford.edu/about/heroes/2018-heroes/jensen-huang)\\n7. [NVIDIA Blog - Lifetime Achievement Award](https://blogs.nvidia.com/blog/ceo-jensen-huang-awarded-lifetime-achievement-asian-american-engineer/)\\n8. [Semiconductors.org - Jensen Huang](https://www.semiconductors.org/nvidia-founder-and-ceo-jensen-huang-to-receive-semiconductor-industrys-top-honor/)\\n9. [Medium - Jensen Huang\'s 30-Year CEO Journey](https://medium.com/@techheadings/how-jensen-huang-became-the-30-year-ceo-of-nvidia-469bd5406b46)"}')]





----------------------------------------------------------------------------------------------------------





I have gathered and summarized comprehensive information on Jensen Huang, including his biography, career achievements, and recognitions. The information is cited from reliable sources.

### Summary of Findings:

#### Biography:
- **Full Name:** Jen-Hsun "Jensen" Huang
- **Date of Birth:** February 17, 1963
- **Place of Birth:** Tainan, Taiwan
- **Early Life:** Moved to Thailand shortly after birth, then to the United States due to the Vietnam War. Worked as a busboy at Denny's during his early years.
- **Education:** Studied at Oregon State University and received a master's degree in electrical engineering from Stanford University in 1992.

#### Career:
- **NVIDIA:**
  - **Co-founded NVIDIA:** In 1993, along with Chris Malachowsky and Curtis Priem.
  - **Role:** President, CEO, and a member of the board of directors since its inception.
  - **Achievements:** Pioneered the development of the Graphics Processing Unit (GPU), led NVIDIA to become a leader in accelerated computing and AI technologies, and expanded into various fields including gaming, scientific computing, and self-driving cars.

#### Achievements and Recognitions:
- **Awards and Honors:**
  - Named the world's best CEO by Fortune, The Economist, and Brand Finance.
  - Listed as one of TIME magazine's 100 most influential people.
  - Received the Distinguished Lifetime Achievement Award by Asian American Engineer of the Year.
  - Awarded the Semiconductor Industry's top honor.
- **Innovations:**
  - Introduced the concept of "Huang's Law," predicting the rapid advancement of GPUs.
  - Played a significant role in the development of AI and machine learning technologies.
- **Net Worth:**
  - Owns approximately 3% of NVIDIA.
  - Net worth surged to $90 billion due to the company's stock performance.

#### Sources:
1. [Wikipedia - Jensen Huang](https://en.wikipedia.org/wiki/Jensen_Huang)
2. [NVIDIA Newsroom - Jensen Huang](http://nvidianews.nvidia.com/bios/jensen-huang)
3. [Britannica - Jensen Huang](https://www.britannica.com/biography/Jensen-Huang)
4. [Forbes - Jensen Huang](https://www.forbes.com/profile/jensen-huang-1/)
5. [The Immigrant Learning Center - Jensen Huang](https://www.ilctr.org/entrepreneur-hof/jen-hsun-huang/)
6. [Stanford University School of Engineering - Jensen Huang](https://engineering.stanford.edu/about/heroes/2018-heroes/jensen-huang)
7. [NVIDIA Blog - Lifetime Achievement Award](https://blogs.nvidia.com/blog/ceo-jensen-huang-awarded-lifetime-achievement-asian-american-engineer/)
8. [Semiconductors.org - Jensen Huang](https://www.semiconductors.org/nvidia-founder-and-ceo-jensen-huang-to-receive-semiconductor-industrys-top-honor/)
9. [Medium - Jensen Huang's 30-Year CEO Journey](https://medium.com/@techheadings/how-jensen-huang-became-the-30-year-ceo-of-nvidia-469bd5406b46)





----------------------------------------------------------------------------------------------------------





FINAL ANSWER:


I have gathered and summarized comprehensive information on Jensen Huang, including his biography, career achievements, and recognitions. The information is cited from reliable sources.

### Summary of Findings:

#### Biography:
- **Full Name:** Jen-Hsun "Jensen" Huang
- **Date of Birth:** February 17, 1963
- **Place of Birth:** Tainan, Taiwan
- **Early Life:** Moved to Thailand shortly after birth, then to the United States due to the Vietnam War. Worked as a busboy at Denny's during his early years.
- **Education:** Studied at Oregon State University and received a master's degree in electrical engineering from Stanford University in 1992.

#### Career:
- **NVIDIA:**
  - **Co-founded NVIDIA:** In 1993, along with Chris Malachowsky and Curtis Priem.
  - **Role:** President, CEO, and a member of the board of directors since its inception.
  - **Achievements:** Pioneered the development of the Graphics Processing Unit (GPU), led NVIDIA to become a leader in accelerated computing and AI technologies, and expanded into various fields including gaming, scientific computing, and self-driving cars.

#### Achievements and Recognitions:
- **Awards and Honors:**
  - Named the world's best CEO by Fortune, The Economist, and Brand Finance.
  - Listed as one of TIME magazine's 100 most influential people.
  - Received the Distinguished Lifetime Achievement Award by Asian American Engineer of the Year.
  - Awarded the Semiconductor Industry's top honor.
- **Innovations:**
  - Introduced the concept of "Huang's Law," predicting the rapid advancement of GPUs.
  - Played a significant role in the development of AI and machine learning technologies.
- **Net Worth:**
  - Owns approximately 3% of NVIDIA.
  - Net worth surged to $90 billion due to the company's stock performance.

#### Sources:
1. [Wikipedia - Jensen Huang](https://en.wikipedia.org/wiki/Jensen_Huang)
2. [NVIDIA Newsroom - Jensen Huang](http://nvidianews.nvidia.com/bios/jensen-huang)
3. [Britannica - Jensen Huang](https://www.britannica.com/biography/Jensen-Huang)
4. [Forbes - Jensen Huang](https://www.forbes.com/profile/jensen-huang-1/)
5. [The Immigrant Learning Center - Jensen Huang](https://www.ilctr.org/entrepreneur-hof/jen-hsun-huang/)
6. [Stanford University School of Engineering - Jensen Huang](https://engineering.stanford.edu/about/heroes/2018-heroes/jensen-huang)
7. [NVIDIA Blog - Lifetime Achievement Award](https://blogs.nvidia.com/blog/ceo-jensen-huang-awarded-lifetime-achievement-asian-american-engineer/)
8. [Semiconductors.org - Jensen Huang](https://www.semiconductors.org/nvidia-founder-and-ceo-jensen-huang-to-receive-semiconductor-industrys-top-honor/)
9. [Medium - Jensen Huang's 30-Year CEO Journey](https://medium.com/@techheadings/how-jensen-huang-became-the-30-year-ceo-of-nvidia-469bd5406b46)
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