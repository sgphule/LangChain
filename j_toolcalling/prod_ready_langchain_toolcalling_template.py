from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableWithTools

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------
# 1. Define tools
# ---------------------------------------------------------
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


TOOLS = [multiply]

# ---------------------------------------------------------
# 2. Create LLM with tools bound
# ---------------------------------------------------------
llm = ChatOpenAI(temperature=0)
llm_with_tools = llm.bind_tools(TOOLS)

# ---------------------------------------------------------
# 3. Create a prompt template
# ---------------------------------------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use tools when needed."),
    ("human", "{input}")
])

# ---------------------------------------------------------
# 4. Create an agent that automatically handles tool calls
# ---------------------------------------------------------
agent = RunnableWithTools(
    runnable=prompt | llm_with_tools,
    tools=TOOLS
)

# ---------------------------------------------------------
# 5. Run the agent
# ---------------------------------------------------------
query = "I need multiplication of 5 with 12"

result = agent.invoke({"input": query})

# ---------------------------------------------------------
# 6. Extract final answer
# ---------------------------------------------------------
final_message = result["messages"][-1]
print("\nFinal Answer:", final_message.content)