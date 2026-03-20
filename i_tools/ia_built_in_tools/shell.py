from langchain_community.tools import ShellTool

shell_tool = ShellTool()
result = shell_tool.invoke("dir")
print("result")
print(result)

# https://reference.langchain.com/python/langchain-community/tools