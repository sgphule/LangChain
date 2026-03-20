from langchain_core.tools import tool, BaseTool
from pydantic import BaseModel, Field
from typing import List


class AddInput(BaseModel):
    a: int = Field(..., description="First number")
    b: int = Field(..., description="Second number")

@tool(args_schema=AddInput)
def add(a:int, b:int) -> int:
    """Add two numbers"""
    return a + b

class MultiplyInput(BaseModel):
    a: int = Field(..., description="First number")
    b: int = Field(..., description="Second number")

@tool(args_schema=MultiplyInput)
def multiply(a: int, b: int) -> int:
    """multiply two numbers"""
    return a * b

class MathToolkit:
    """A simple toolkit providing arithmetic tools."""
    def __init__(self):
        self._tools = [add, multiply]

    def register(self, tool):
        self._tools.append(tool)

    def get_tools(self):
        return self._tools


toolkit = MathToolkit()
tools = toolkit.get_tools()

for tool in tools:
    print(tool.name, "=>", tool.description)

