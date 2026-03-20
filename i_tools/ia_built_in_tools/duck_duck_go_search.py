from langchain_community.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun()
result = search_tool.invoke("Bundesliga News")
print("result")
print(result)

print("search_tool.name:", search_tool.name)
print("search_tool.description:", search_tool.description)
print("search_tool.args:", search_tool.args)
print("search_tool.args_schema.model_json_schema():", search_tool.args_schema.model_json_schema())
# https://reference.langchain.com/python/langchain-community/tools