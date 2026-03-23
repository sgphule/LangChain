from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

# tool creation
@tool
def multiply(a: int, b: int) -> int:
    """Given 2 numbers a and b, this function returns their mathematical product"""
    return a * b

result = multiply.invoke({"a": 5, "b": 8})
print("result: ", result)
print("multiply.name:", multiply.name)
print("multiply.description:", multiply.description)
print("multiply.args:", multiply.args)

# tool binding
llm = ChatOpenAI()
llm_with_tools = llm.bind_tools([multiply])

# tool calling
query = "I need multiplication of 5 with 12"
messages = HumanMessage(query)
messages = [messages]
print("Human Messages: ", messages)
ai_message = llm_with_tools.invoke(messages)
print("AI Message:", ai_message)
messages.append(ai_message)

# tool execution
tool_result = multiply.invoke(ai_message.tool_calls[0])
print("Tool Message:", tool_result)
messages.append(tool_result)
print("Whole history: ", messages)
final_result = llm_with_tools.invoke(messages).content
print("Final Result:", final_result)