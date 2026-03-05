from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

loader = TextLoader("doc.txt")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
docs = text_splitter.split_documents(documents)

vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())

retriever = vectorstore.as_retriever()

query = "What are the key takeaways from the document?"
retrieved_docs = retriever.invoke(query)

retrieved_text = "\n".join([doc.page_content for doc in retrieved_docs])

llm = OpenAI()

prompt = f"Based on the following text answer the question:{query}\n\n{retrieved_text}"
answer = llm.invoke(prompt)

print("Answer:", answer)