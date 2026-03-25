from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.tools import InjectedToolArg
from typing import Annotated
import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("EXCHANGE_API_KEY")
BASE_CURRENCY = 'EUR'
TARGET_CURRENCY = 'INR'
BASE_CURRENCY_VALUE = 1000
SUCCESS_CODE = 200

# tool creation
@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
    """Given base and target Currency, this function fetches currency conversion factor"""
    url = f"https://v6.exchangerate-api.com/v6/{API_KEY}/pair/{base_currency}/{target_currency}"
    response = requests.get(url)
    print("response.status_code:", response.status_code)

    if response.status_code == SUCCESS_CODE:
        factor = response.json()['conversion_rate']
    else:
        url = f"https://api.frankfurter.app/latest?from={base_currency}&to={target_currency}"
        response = requests.get(url)
        factor = response.json()["rates"][target_currency]
    return factor

conversion_factor_with_single_tool = get_conversion_factor.invoke({'base_currency':BASE_CURRENCY, 'target_currency': TARGET_CURRENCY})
print("conversion_factor_with_single_tool:", conversion_factor_with_single_tool)

@tool
def convert(base_currency_value: int, conversion_rate: Annotated[float, InjectedToolArg]) -> float:
    """Given Currency rate and a base currency value, this function calculates target currency value"""
    print("type(base_currency_value):", type(base_currency_value), "type(conversion_rate):", type(conversion_rate))
    print("base_currency_value:", base_currency_value, "conversion_rate:", conversion_rate)
    result = base_currency_value * float(conversion_rate)
    print("result: ", result)
    return result

final_answer_with_single_tool = convert.invoke({"base_currency_value": BASE_CURRENCY_VALUE, "conversion_rate": conversion_factor_with_single_tool})
print(f"Currency value of {BASE_CURRENCY_VALUE} {BASE_CURRENCY} in {TARGET_CURRENCY} is:", final_answer_with_single_tool)

# tool binding
llm = ChatOpenAI()
llm_with_tools = llm.bind_tools([get_conversion_factor, convert])

# tool calling
messages = [HumanMessage(f"What is the conversion factor between {BASE_CURRENCY} and {TARGET_CURRENCY}, and based on this can you convert "
                         f"{BASE_CURRENCY_VALUE} {BASE_CURRENCY} to {TARGET_CURRENCY}")]

print("Messages:", messages)

ai_message = llm_with_tools.invoke(messages)
print("ai_message:", ai_message)
messages.append(ai_message)

print("tool_calls:", ai_message.tool_calls)

for tool_call in ai_message.tool_calls:
    print("TOOL CALL:", tool_call)
    if tool_call["name"] == "get_conversion_factor":
        tool_message1 = get_conversion_factor.invoke(tool_call)
        print("tool_message1:", tool_message1)
        conversion_rate = tool_message1.content
        print("conversion_rate:", conversion_rate)
        messages.append(tool_message1)
    if tool_call["name"] == "convert":
        tool_call["args"]["conversion_rate"] = conversion_rate
        tool_message2 = convert.invoke(tool_call)
        messages.append(tool_message2)
        print("MESSAGES: ", messages)
print("Content:", llm_with_tools.invoke(messages).content)