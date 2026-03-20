from langchain_core.output_parsers import JsonOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template = "Generate the name, age and city of any fictional actor \n {format_instruction}",
    input_variables=[],
    partial_variables={"format_instruction":parser.get_format_instructions()}
)

# prompt = template.format()
# print(prompt)
# result = model.invoke(prompt)
# final_result = parser.parse(result.content)

chain = template | model | parser

final_result = chain.invoke({})

print(final_result)
print(type(final_result))