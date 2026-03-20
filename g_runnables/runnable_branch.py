from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch

load_dotenv()
def is_word_count_more_than_threshold(text):
    threshold = 300
    return len(text.split())>threshold

prompt1 = PromptTemplate(
    template = "write a detailed report on {topic}",
    input_variables = ["topic"]
)
prompt2 = PromptTemplate(
    template = "summarize the following text \n {text}",
    input_variables = ["text"]
)
model = ChatOpenAI()
parser = StrOutputParser()
report_gen_chain = RunnableSequence(prompt1, model, parser)
branch_chain = RunnableBranch(
    (is_word_count_more_than_threshold, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
)
final_chain = RunnableSequence(report_gen_chain, branch_chain)
result = final_chain.invoke({"topic": "US Vs Iran"})
print(result)
#final_chain.get_graph().print_ascii()

