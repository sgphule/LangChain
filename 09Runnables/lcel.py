from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence

load_dotenv()

prompt1 = PromptTemplate(
    template = "List down {topic} in LangChain Open Source framework for implementing AI agents",
    input_variables = ["topic"]
)
model = ChatOpenAI()
parser = StrOutputParser()
lcel = prompt1 | model | parser
result = lcel.invoke({"topic":"Document loaders"})
print(result)
