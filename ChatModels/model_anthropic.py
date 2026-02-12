from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model='claude-sonnet-4-5-20250929')
result = model.invoke("Could you help me writing 6 line poem on AI?")
print(result.content)
