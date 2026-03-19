from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

load_dotenv()
# Indexing (Document Ingestion)
print("1.1. Indexing (Document Ingestion)")
video_id = "ZaPbP9DwBOE"
transcript = ""
try:
    ytt_api = YouTubeTranscriptApi()
    fetched_transcript = ytt_api.fetch(video_id)
    print(type(fetched_transcript))
    transcript = " ".join(snippet.text for snippet in fetched_transcript)
    print(transcript)

except TranscriptsDisabled:
    print("No captions available for this video.")


# Indexing (Text splitting)
print("1.2. Indexing (Text splitting):")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.create_documents([transcript])
print("len(chunks):", len(chunks))
print("chunks[0]:", chunks[0])

# Indexing (Embedding Generation and storing in vector store)
print("1.3. Indexing (Embedding Generation and storing in vector store)")
embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")
vector_store = FAISS.from_documents(chunks, embeddings)
print("vector_store.index_to_docstore_id:", vector_store.index_to_docstore_id)
print("vector_store.get_by_ids(['cb294557-df6b-492f-a55f-0355a318ba52']):", vector_store.get_by_ids(['cb294557-df6b-492f-a55f-0355a318ba52']))

# Retrieval
print("2. Retrieval")
retriever = vector_store.as_retriever(search_type= "similarity", search_kwargs={"k":4})
print(retriever)
result = retriever.invoke("How Businesses can unlock full value of its knowledge?")
print("result:", result)

# Augmentation
print("3. Augmentation")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)
prompt = PromptTemplate(
    template="""
    Answer only from the provided context.
    If context is insufficient, just say you don't know.
    {context}
    Question: {question}
    """,
    input_variables=["context", "question"]
)
question = "How Businesses can unlock full value of its knowledge?"
retrieved_docs = retriever.invoke(question)
print("retrieved_docs:", retrieved_docs)
context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
print("context_text:", context_text)
final_prompt = prompt.invoke({"context": context_text, "question": question})
print("final_prompt:", final_prompt)

# Generation
print("Generation")
final_answer = llm.invoke(final_prompt)
print("final_answer:", final_answer.content)


# Building a chain
"""
print("Building a chain")
def format_docs(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text

parallel_chain = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough
})
parallel_chain.invoke("How Businesses can unlock full value of its knowledge?")
parser = StrOutputParser()
main_chain = parallel_chain | prompt | llm | parser
main_chain.invoke("Generate video Summary.")
"""