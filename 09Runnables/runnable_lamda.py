from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda

load_dotenv()

def word_counter(text):
    return len(text.split())

runnable_word_counter = RunnableLambda(word_counter)
result = runnable_word_counter.invoke("Hi! Wie geht es dir?")
print("word counter:", result)

prompt1 = PromptTemplate(
    template = "create a joke on {topic}",
    input_variables = ["topic"]
)

model = ChatOpenAI()
parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt1, model, parser)
parallel_chain = RunnableParallel({
    "joke": RunnablePassthrough(),
    "word count": RunnableLambda(word_counter) # or RunnableLambda(lamda x:len(x.split()))
})
final_chain = RunnableSequence(joke_gen_chain, parallel_chain)
result = final_chain.invoke({"topic": "Generative AI"})
final_result = "{} \nword count: {}".format(result["joke"], result["word count"])
print("result:", result)
print(final_result)