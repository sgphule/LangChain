from langchain_anthropic import ChatAnthropic
from langchain_classic.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = ChatAnthropic(model='claude-sonnet-4-5-20250929')

prompt = PromptTemplate(
    template="Suggest a good Project topic in {field_of_study}",
    input_variables=["field_of_study"]
)

chain = LLMChain(llm=llm, prompt=prompt)

field_of_study = input("Enter any field_of_study that you are interested in:")
result = chain.invoke(field_of_study)

print("Generated Project topic:", result)