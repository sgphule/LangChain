from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
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
def run_with_tools(llm, tools, user_input, system_prompt=None):
    messages = []

    if system_prompt:
        messages.append(SystemMessage(system_prompt))

    messages.append(HumanMessage(user_input))
    llm_with_tools = llm.bind_tools(tools)

    # Step 1: LLM decides whether to call a tool
    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)

    # Step 2: Execute all tool calls (if any)
    if ai_msg.tool_calls:
        for call in ai_msg.tool_calls:
            tool_name = call["name"]
            tool_args = call["args"]

            # Find the matching tool
            matching_tool = next(t for t in tools if t.name == tool_name)

            # Execute tool
            result = matching_tool.invoke(tool_args)

            # Add tool result message
            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=call["id"]
                )
            )

    # Step 3: Final LLM response
    final = llm_with_tools.invoke(messages)
    return final.content

# ---------------------------------------------------------
# 3. Usage
# ---------------------------------------------------------
llm = ChatOpenAI(temperature=0)

print(run_with_tools(
    llm,
    [multiply],
    "Multiply 17 and 3",
    system_prompt="You are a helpful assistant. Use tools when needed."
))
