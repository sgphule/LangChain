from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
model = ChatOpenAI()
prompt = PromptTemplate(
    template="Answer the following question \n {question} from the following text \n {text}",
    input_variables = ["question", "text"]
)
parser = StrOutputParser()

url = "https://zeissgroup.wd3.myworkdayjobs.com/External/job/Aalen/Product-Owner-digital-Refraction--f-m-x----limited-to-2-years_JR_1043951"
loader = WebBaseLoader(url)
docs = loader.load()
chain = prompt | model | parser
# result = chain.invoke({"question": "Does this role requires Ph.D.?", "text": docs[0].page_content})
print("docs:", docs)
#result = chain.invoke({"question": "Is this role for limited duration?", "text": docs[0].metadata["description"]})
result = chain.invoke({"question": "How much time requires to apply for this job vacancy?", "text": docs[0].metadata["description"]})
print("result:", result)
#print("len(docs):", len(docs))
#print("docs[0].page_content:", docs[0].page_content)