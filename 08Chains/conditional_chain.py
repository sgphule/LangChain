from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

class Feedback(BaseModel):
    sentiment: Literal["Positive", "Negative"] = Field(description="Provides the feedback's sentiment")
parser2 = PydanticOutputParser(pydantic_object=Feedback)

load_dotenv()
model1 = ChatAnthropic(model='claude-sonnet-4-5-20250929')
prompt1 = PromptTemplate(
    template = "Classify the sentiment of following feedback text into positive or negative\n {feedback} \n{format_instruction}",
    input_variables = ["feedback"],
    partial_variables = {"format_instruction": parser2.get_format_instructions()}
)
parser1 = StrOutputParser()

classifier_chain = prompt1 | model1 | parser2
# result = classifier_chain.invoke({"feedback":"This is not a smartphone"})
# print(result.sentiment)
prompt2 = PromptTemplate(
    template = "Write an appropriate response to this positive feedback \n {feedback}",
    input_variables = ["feedback"],
)
prompt3 = PromptTemplate(
    template = "Write an appropriate response to this negative feedback \n {feedback}",
    input_variables = ["feedback"],
)
branch_chain = RunnableBranch(
    (lambda x:x.sentiment=="Positive", prompt2| model1 | parser1),
    (lambda x:x.sentiment=="Negative", prompt3| model1 | parser1),
    RunnableLambda(lambda x:"Could not find a sentiment")
)

final_chain = classifier_chain | branch_chain
print(final_chain.invoke({"feedback":"This is a very good smartphone"}))
final_chain.get_graph().print_ascii()