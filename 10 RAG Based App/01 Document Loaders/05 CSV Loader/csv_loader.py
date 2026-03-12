from langchain_community.document_loaders import CSVLoader
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
loader = CSVLoader(file_path="Fortune 500 Companies US.csv")
docs = loader.load()
print("len(docs):", len(docs))
print("type(docs[0].page_content):", type(docs[0].page_content))

first_five_records = ""
for i in range(5):
    first_five_records = first_five_records + docs[i].page_content
print("first 5 records:", first_five_records)
chain = prompt | model | parser
result = chain.invoke({"question": "Display all Profit values and "
                                   "Perform exact arithmetic only. Do not estimate. Do not skip steps. "
                                   "to find the average, max and min values of Profits ", "text": first_five_records})
print("result:", result)


