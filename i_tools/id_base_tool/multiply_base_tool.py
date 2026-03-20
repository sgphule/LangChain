from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type

class MultiplyInput(BaseModel):
    a: int = Field(..., ge=0, description= "The first number to be multiplied")
    b: int = Field(..., ge=0, description= "The second number to be multiplied")

class MultiplyTool(BaseTool):
    """Multiply two integers."""
    name: str = "multiply"
    description: str = "multiply two numbers"
    args_schema: Type[BaseModel] = MultiplyInput

    def _run(self, a: int, b: int) -> int:
        return a * b

    async def _arun(self, a: int, b: int) -> int:
        return a * b

multiply_tool = MultiplyTool()
result = multiply_tool.invoke({"a":3, "b": 12})
print("result:", result)
print("multiply_tool.name:", multiply_tool.name)
print("multiply_tool.description:", multiply_tool.description)

