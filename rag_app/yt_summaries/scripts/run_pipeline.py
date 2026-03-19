from dotenv import load_dotenv

from rag_app.yt_summaries.app.ingestion.youtube_loader import load_transcript
from rag_app.yt_summaries.app.ingestion.text_splitter import split_text
from rag_app.yt_summaries.app.embeddings.embedding_model import get_embeddings
from rag_app.yt_summaries.app.vectorstore.faiss_store import create_vector_store
from rag_app.yt_summaries.app.retrieval.retriever import get_retriever
from rag_app.yt_summaries.app.chains.prompts import get_prompt
from rag_app.yt_summaries.app.chains.rag_chain import build_chain

load_dotenv()

video_id = "ZaPbP9DwBOE"

# Pipeline
transcript = load_transcript(video_id)
docs = split_text(transcript)

embeddings = get_embeddings()
vector_store = create_vector_store(docs, embeddings)

retriever = get_retriever(vector_store)
prompt = get_prompt()

chain = build_chain(retriever, prompt)

# Run
response = chain.invoke("Generate video summary")
print(response)

"""
The video provides an overview of recent advancements in AI, focusing on concepts like prompt engineering, embeddings, 
vector databases, and retrieval augmented generation (RAG).
 
It aims to educate viewers from a foundational level, explaining these concepts through a single project. 
The project involves building a production-ready search engine that transitions from a broken keyword search system 
with a 60% failure rate to one with a 95% success rate by utilizing embeddings, smart document chunking, and semantic 
search. 

The video also covers the process of RAG, breaking it down into three steps: retrieval, augmentation, and generation. 
Additionally, it explores building workflows with nodes and edges, demonstrating how data flows through various stages 
of content creation. 
The video concludes with an encouragement for viewers to experiment with deeper aspects of the technology.
"""