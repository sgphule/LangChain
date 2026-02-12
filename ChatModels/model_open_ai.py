from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-5.2-pro', temperature=1.9)
result = model.invoke("Tell me capital of all european countries")
print(result.content[1]['text'])
