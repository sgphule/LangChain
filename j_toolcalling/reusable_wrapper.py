from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
# ---------------------------------------------------------
# 1. Define tools
# ---------------------------------------------------------
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

# ---------------------------------------------------------
# 2. Reusable Wrapper
# ---------------------------------------------------------
def run_with_tools(llm, tools, user_input):
    messages = [HumanMessage(user_input)]
    llm_with_tools = llm.bind_tools(tools)

    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)

    if ai_msg.tool_calls:
        tool_call = ai_msg.tool_calls[0]
        tool = next(t for t in tools if t.name == tool_call["name"])
        result = tool.invoke(tool_call["args"])

        messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"]
            )
        )

    final = llm_with_tools.invoke(messages)
    return final.content

llm = ChatOpenAI(temperature=0)

# ---------------------------------------------------------
# 3. Usage
# ---------------------------------------------------------
print(run_with_tools(llm, [multiply], "Multiply 27 and 3"))
