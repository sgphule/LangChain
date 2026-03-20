from langchain_community.tools import ShellTool

shell_tool = ShellTool()
result = shell_tool.invoke("dir")
print("result")
print(result)

print("shell_tool.name:", shell_tool.name)
print("shell_tool.description:", shell_tool.description)
print("shell_tool.args:", shell_tool.args)
print("shell_tool.args_schema.model_json_schema():", shell_tool.args_schema.model_json_schema())

# https://reference.langchain.com/python/langchain-community/tools