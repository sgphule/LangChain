from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
llm = OpenAI()

prompt = PromptTemplate(
    template = "Suggest a good PhD topic in {field_of_study}",
    input_variables = ["field_of_study"]
)

field_of_study = input("Enter any field_of_study that you are interested in:")

formatted_prompt = prompt.format(field_of_study = field_of_study)

phd_topic = llm.invoke(formatted_prompt)

print("Generated Phd Topic is:", phd_topic)