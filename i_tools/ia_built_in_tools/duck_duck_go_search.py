from langchain_community.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun()
result = search_tool.invoke("Bundesliga News")
print("result")
print(result)

# https://reference.langchain.com/python/langchain-community/tools