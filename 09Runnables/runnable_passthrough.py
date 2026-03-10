from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough

load_dotenv()

passthrough = RunnablePassthrough()
print(passthrough.invoke({"country":"USA"}))

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
joke_gen_chain = RunnableSequence(prompt1, model, parser)
parallel_chain = RunnableParallel({
    "joke": RunnablePassthrough(),
    "explanation": RunnableSequence(prompt2, model, parser)
})
final_chain = RunnableSequence(joke_gen_chain, parallel_chain)
output = final_chain.invoke({"topic":"Soccer"})
print(output)
print("JOKE:", output["joke"])
print("JOKE EXPLAIN:", output["explanation"])
