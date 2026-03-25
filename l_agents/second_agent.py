from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
from langchain_classic import hub
from langchain_classic.agents import create_react_agent, AgentExecutor
from dotenv import load_dotenv

load_dotenv()

search_tool = DuckDuckGoSearchRun()
llm = ChatOpenAI()
prompt = hub.pull("hwchase17/react")

agent = create_react_agent(
    llm=llm,
    tools=[search_tool],
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool],
    verbose=True
)

response = agent_executor.invoke({"input": "Which frameworks do you suggest for learning to create multi agents systems?"})
print(response)

# > Entering new AgentExecutor chain...
# I should search for popular frameworks for creating multi-agent systems
# Action: duckduckgo_search
# Action Input: best frameworks for creating multi-agent systemsIf you want to build a basicmulti-agentsystemwithout
# relying on heavyframeworkslike LangGraph or CrewAI , you cancreateit from scratch using ... ... planning, reasoning,
# memory, tool use andmulti-step workflows, developers are shifting toward agentic AIframeworkswhich are
# structuredsystemsfor... ...agentframework, represents a low-code, centralizedframeworkdesigned to simplify the
# creation and orchestration ofmulti-agentsystemsforvarious ... ...agentframework, represents a low-code,
# centralizedframeworkdesigned to simplify the creation and orchestration ofmulti-agentsystemsforvarious ... Restaurant
# reservation :Forexample, an in-app AIagentin a restaurantsystemcan be designed to help users make online reservations,
# compare ...I need to find more specific information about the frameworks mentioned in the search results
# Action: duckduckgo_search
# Action Input: LangGraph framework for multi-agent systemsJun 6, 2025 ·In this guide, we’ll exploreLangGraph’s
# architecture, key components, and how to use it to orchestrate intelligent agents that communicate and collaborate on
# tasks. LangGraph is alibrary built on top of LangChainthat allows you to create stateful, multi-agent applications
# with Large Language Models (LLMs). It provides a structured way to compose chains into graphs, manage state across
# multiple LLM calls, and build complex workflows with conditional routing. May 20, 2025 ·This repository presents a
# modular,LangGraph-basedmulti-agentframeworkengineered for the development of sophisticated conversational AI
# applications. Its robustframeworkfor building stateful,multi-actorapplications with LLMs has transformed how we
# evaluate and optimize the performance of our AI guest-facing solutions. In this comprehensive guide, you’ll learn how
# to build production-readymulti-agentsystemsusingLangGraph. We’ll cover everything from basic concepts to advanced
# patterns, with practical code examples you can use in your projects.I have gathered enough information about various
# frameworks for creating multi-agent systems
# Final Answer: Some popular frameworks for creating multi-agent systems include LangGraph and CrewAI, among others.

# > Finished chain.
# {'input': 'Which frameworks do you suggest for learning to create multi agents systems?', 'output': 'Some popular
# frameworks for creating multi-agent systems include LangGraph and CrewAI, among others.'}
# Process finished with exit code 0