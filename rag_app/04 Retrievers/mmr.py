from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()
docs = [
    Document(page_content = "CrewAI / AutoGen - Best for workflow‑driven, role‑based multi‑agent teams. CrewAI feels like building a small company of AI workers."),
    Document(page_content="Qdrant is currently the strongest choice for multi‑agent systems because it balances everything that agents need"),
    Document(page_content = "AutoGen (Microsoft + CMU) - The strongest multi‑agent framework today. If your goal is true multi‑agent orchestration, AutoGen is the most mature and expressive option."),
    Document(page_content="Redis Vector Store - Redis is unbeatable when agents need instant memory access. If your agents talk frequently or coordinate rapidly, Redis feels like giving them a shared brain."),
    Document(page_content = "MultiQueryRetriever - This is the strongest choice for multi‑agent systems because it mirrors how agents think: multiple perspectives → merged evidence → stronger retrieval."),
    Document(page_content="ParentDocumentRetriever - This retriever returns the full parent document, even if only a chunk matched. Useful when agents need to build shared memory or long‑term understanding."),
    Document(page_content="SelfQueryRetriever - This retriever uses an LLM to convert natural language into structured search filters. If your agents need to filter by date, author, topic, or domain, this is the one."),
    Document(page_content="ContextualCompressionRetriever - This retriever compresses or filters documents before returning them. Agents stay efficient and avoid drowning in long context windows.")
]

embedding_model = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(
    documents=docs,
    embedding=embedding_model
)

retriever = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs = {"k": 3, "lamda_mult": 0.5}
)

query = "Which Retriever is best for long context?"
results = retriever.invoke(query)

for i,doc in enumerate(results):
    print(f"\n================ Result {i+1} ================")
    print(doc.page_content)

# Result for search_kwargs = {"k": 3, "lamda_mult": 1}
# ================ Result 1 ================
# MultiQueryRetriever - This is the strongest choice for multi‑agent systems because it mirrors how agents think: multiple perspectives → merged evidence → stronger retrieval.
#
# ================ Result 2 ================
# ContextualCompressionRetriever - This retriever compresses or filters documents before returning them. Agents stay efficient and avoid drowning in long context windows.
#
# ================ Result 3 ================
# ParentDocumentRetriever - This retriever returns the full parent document, even if only a chunk matched. Useful when agents need to build shared memory or long‑term understanding.