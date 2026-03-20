from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from dotenv import load_dotenv

load_dotenv()

docs = [
    Document(
        page_content=(
            """Mount Everest is the tallest mountain above sea level on Earth.
            The process of erosion shapes landscapes over millions of years.
            Climbers from around the world attempt to reach its summit each year.
            The Himalayas continue to rise due to tectonic plate movement."""
        ),
        metadata={"source": "Doc1"}
    ),

    Document(
        page_content=(
            """Ancient Egypt developed one of the earliest known writing systems called hieroglyphics.
            The Nile River flooded annually, providing fertile soil for agriculture.
            Pyramids were constructed as monumental tombs for pharaohs.
            Solar energy is produced when sunlight is converted into electricity by photovoltaic cells."""
        ),
        metadata={"source": "Doc2"}
    ),

    Document(
        page_content=(
            """Volcanoes form when magma rises through cracks in the Earth's crust.
            Lava flows can create new land over time.
            The Pacific Ring of Fire contains most of the world's active volcanoes.
            Renewable energy sources help reduce greenhouse gas emissions."""
        ),
        metadata={"source": "Doc3"}
    ),

    Document(
        page_content=(
            """The Renaissance was a period of cultural revival in Europe beginning in the 14th century.
            Artists like Leonardo da Vinci and Michelangelo produced influential works.
            The printing press revolutionized the spread of knowledge.
            Gravity is the force that attracts objects toward the center of the Earth."""
        ),
        metadata={"source": "Doc4"}
    ),
]

embedding_model = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embedding_model)

base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

llm=ChatOpenAI(model="gpt-3.5-turbo")
compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_retriever= base_retriever,
    base_compressor=compressor
)

query = "How Solar energy is produced?"
compressed_results = compression_retriever.invoke(query)

for i, doc in enumerate(compressed_results):
    print(f"\n ========== Compressed Result {i+1} ==========")
    print(doc.page_content)