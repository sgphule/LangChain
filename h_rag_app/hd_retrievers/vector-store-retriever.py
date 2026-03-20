from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

documents = [
    Document(page_content="Document Loaders - Load raw data from files, URLs, APIs, or external sources into LangChain as standardized Document objects."),
    Document(page_content="Text Splitters - Break large documents into smaller, semantically meaningful chunks optimized for embedding and retrieval."),
    Document(page_content="Vector Stores - Store and index embeddings so you can perform fast similarity search over your document chunks."),
    Document(page_content="Retrievers - Fetch the most relevant documents from a vector store (or other source) based on a user query.")
]

embedding_model = OpenAIEmbeddings()

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    collection_name="my_collection"
)

retriever = vectorstore.as_retriever(search_kwargs = {"k":2})

query = "What is the use of Text Splitter and Vector Store in a RAG pipeline"
results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"\n ================= Result {i+1} =================")
    print(doc.page_content)
