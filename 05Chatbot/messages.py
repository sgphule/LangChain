from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


load_dotenv()

model = ChatOpenAI()
messages = [
    SystemMessage(content="You are a MD Physician doctor"),
    HumanMessage(content="I am having toe pain what should I do?")
]

result = model.invoke(messages)

messages.append(AIMessage(content = result.content))

print(messages)
