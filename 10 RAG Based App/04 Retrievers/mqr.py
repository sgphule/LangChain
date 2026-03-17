from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_classic.retrievers import MultiQueryRetriever

from dotenv import load_dotenv

load_dotenv()

docs = [
    # 5 random-topic
    Document(
        page_content="The scientist carefully recorded the temperature changes during the experiment.",
        metadata={"source": "h6"}),
    Document(
        page_content="A gentle breeze moved through the forest as the sun began to rise.",
        metadata={"source": "h7"}),
    Document(
        page_content="The chef prepared a delicious meal using fresh ingredients from the market.",
        metadata={"source": "h8"}),
    Document(
        page_content="The engineer designed a new bridge that could withstand strong winds.",
        metadata={"source": "h9"}),
    Document(
        page_content="The artist painted a vibrant landscape inspired by her travels.",
        metadata={"source": "h10"}),
    # 5 factual soccer documents
    Document(
        page_content="A standard soccer match consists of two 45-minute halves.",
        metadata={"source": "h1"}),
    Document(
        page_content="The FIFA World Cup is held every four years and is the most-watched soccer tournament globally.",
        metadata={"source": "h2"}),
    Document(
        page_content="A soccer team fields eleven players, including one goalkeeper.",
        metadata={"source": "h3"}),
    Document(
        page_content="The offside rule prevents attacking players from gaining an unfair positional advantage.",
        metadata={"source": "h4"}),
    Document(
        page_content="A soccer field is typically between 100 and 110 meters long according to FIFA regulations.",
        metadata={"source": "h5"})
]

embedding_model = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents=docs, embedding=embedding_model)
similarity_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever = vectorstore.as_retriever(search_kwargs = {"k":5}),
    llm=ChatOpenAI(model="gpt-3.5-turbo")
)

query = "What are few facts about soccer game?"
similarity_results = similarity_retriever.invoke(query)
multiquery_results = multiquery_retriever.invoke(query)

for i, doc in enumerate(similarity_results):
    print(f"\n ========== Similarity Result {i+1} ==========")
    print(doc.page_content)

for i, doc in enumerate(multiquery_results):
    print(f"\n =|=|=|=|=|= Multiquery Result {i+1} =|=|=|=|=|=")
    print(doc.page_content)