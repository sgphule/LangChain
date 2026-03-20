from langchain_core.tools import tool

@tool
def multiply(a:int, b:int)->int:
    """Multiply two numbers"""
    return a * b

result = multiply.invoke({"a":5, "b":7})
print("result:", result)
print("multiply.name:", multiply.name)
print("multiply.description:", multiply.description)
print("multiply.args:", multiply.args)
print("multiply.args_schema.model_json_schema():", multiply.args_schema.model_json_schema())
