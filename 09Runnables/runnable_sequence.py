from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence

load_dotenv()

prompt1 = PromptTemplate(
    template = "create a joke on {topic}",
    input_variables = ["topic"]
)

model = ChatOpenAI()
parser = StrOutputParser()

prompt2 = PromptTemplate(
    template="Explain the following joke {text}",
    input_variables=["text"]
)
chain1 = RunnableSequence(prompt1, model, parser)
output1 = chain1.invoke({"topic":"AI"})
chain2 = RunnableSequence(prompt1, model, parser, prompt2, model, parser)
output = chain2.invoke({"topic":"AI"})
print(output1)
print(output)
