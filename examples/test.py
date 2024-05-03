from compositeai.drivers import OpenAIDriver
from compositeai.tools import GoogleSerperApiTool

test = OpenAIDriver(model="gpt-3.5-turbo-1106")
tool = GoogleSerperApiTool()
messages = [
    {"role": "system", "content": "You are a helpful assistant that will use tools at your disposal to complete the user's task."},
    {"role": "user", "content": "Please find contact information for as many political candidates running for office in Boston as you can."}
]

response = test._iterate(messages=messages, tools=[tool])
print(response)

