from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

prompt1 = PromptTemplate(
    template = "Generate a detailed report on facts about {topic}",
    input_variables = ["topic"]
)

prompt2 = PromptTemplate(
    template = "Generate a 10-15 lines summary from following report \n {report}",
    input_variables = ["report"]
)

model = ChatOpenAI()
parser = StrOutputParser()
chain = prompt1 | model | parser | prompt2 | model | parser
result = chain.invoke({"topic":"Unemployment in Germany Vs Unemployment in India"})

print(result)
chain.get_graph().print_ascii()