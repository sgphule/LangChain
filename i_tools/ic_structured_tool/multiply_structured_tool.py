from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class MultiplyInput(BaseModel):
    a: int = Field(..., ge=0, description="The first number to be multiplied")
    b: int = Field(..., ge=0, description="The second number to be multiplied")

def multiply_func(a: int, b: int) -> int:
    """Multiply two integers and return the result."""
    return a*b

multiply_tool: StructuredTool = StructuredTool.from_function(
    func = multiply_func,
    name = "multiply",
    description="Multiply two numbers",
    args_schema=MultiplyInput,
    return_direct=True
)

result = multiply_tool.invoke({"a": 9, "b": 7})
print("result:", result)
print("multiply_tool.name:", multiply_tool.name)
print("multiply_tool.description:", multiply_tool.description)