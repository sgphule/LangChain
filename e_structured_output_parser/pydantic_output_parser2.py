from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# 1. Define your structured output using Pydantic
class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(gt=18, description="Age of the person")
    city: str = Field(description="City name the person belongs to")

parser = PydanticOutputParser(pydantic_object=Person)

# 2. LLM setup
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# 3. Prompt template with format instructions
template = PromptTemplate(
    template="Generate the name, age and city of a fictional person from {country} \n" 
        "{format_instructions}",
        input_variables=["country"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
)

# Without chain
# prompt = template.invoke ({"country":"Germany"})
# result = model.invoke(prompt)
# final_result = parser.parse(result.content)
# print(final_result)

# 4. LCEL chain
chain = template | model | parser

# 5. Invoke
result = chain.invoke({"country": "Sri Lanka"})
print(result)

