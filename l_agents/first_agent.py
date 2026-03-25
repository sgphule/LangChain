from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.agents import create_tool_calling_agent
from langchain_classic.agents import AgentExecutor
from dotenv import load_dotenv

load_dotenv()

search_tool = DuckDuckGoSearchRun()
tool_result = search_tool.invoke("What's the difference between DuckDuckGoSearchRun and Google Search?")
print(tool_result)

llm = ChatOpenAI()
llm_result = llm.invoke("How can I import create_react_agent, AgentExecutor in LangChain?")
print(llm_result)

# Correct prompt for tool-calling agents
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that uses tools when needed."),
    ("human", "{input}"),
    ("assistant", "{agent_scratchpad}")
])

agent = create_tool_calling_agent(
    llm = llm,
    tools = [search_tool],
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent = agent,
    tools = [search_tool],
    verbose= True
)

response = agent_executor.invoke({"input": "What should be the step by step learning flow for creating multi agent systems for a beginner?"})
print(response)

#> Entering new AgentExecutor chain...
# Invoking: `duckduckgo_search` with `{'query': 'step by step learning flow for creating multi agent systems for beginners'}`
# It ’ s becoming essentialforpractitioners to able to build AIagents, especiallymulti-agentsystems. ...Agentto make
# things easier, which ... In this comprehensive guide, we ’ ll take astep-by-stepapproach to exploring the world of
# multi-agentsystems, from the basics of single ... ... tasks is through a “divide-and-conquer” approach:createa
# specializedAgentforeach task and have them work together as aMulti-AgentSystem... If you already have a grasp of
# Reinforcement and Deep Q-Learningconcepts, feel free to jump directly to thestep-by-steptutorial. Forinstance,
# MindPal demonstrates how thesesystemsimprove customer supportbyautomating tasks like ticket routing and resolution.
# For beginners looking to create multi-agent systems, the step-by-step learning flow typically involves understanding
# the basics of single agents, learning about divide-and-conquer approaches to create specialized agents for specific
# tasks, and exploring concepts like Reinforcement and Deep Q-Learning. Once you have a grasp of these concepts, you
# can jump into a step-by-step tutorial to start building your multi-agent system. One example to consider is how these
# systems can improve customer support by automating tasks such as ticket routing and resolution.

# > Finished chain.
# {'input': 'What should be the step by step learning flow for creating multi agent systems for a beginner?',
# 'output': 'For beginners looking to create multi-agent systems, the step-by-step learning flow typically involves
# understanding the basics of single agents, learning about divide-and-conquer approaches to create specialized agents
# for specific tasks, and exploring concepts like Reinforcement and Deep Q-Learning. Once you have a grasp of these
# concepts, you can jump into a step-by-step tutorial to start building your multi-agent system. One example to consider
# is how these systems can improve customer support by automating tasks such as ticket routing and resolution.'}