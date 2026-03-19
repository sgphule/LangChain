from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


text = """
LangChain is the easiest way to start building agents and applications powered by LLMs. With under 10 lines of code, you can connect to OpenAI, Anthropic, Google, and more. LangChain provides a pre-built agent architecture and model integrations to help you get started quickly and seamlessly incorporate LLMs into your agents and applications.
We recommend you use LangChain if you want to quickly build agents and autonomous applications. Use LangGraph, our low-level agent orchestration framework and runtime, when you have more advanced needs that require a combination of deterministic and agentic workflows, heavy customization, and carefully controlled latency.
LangChain agents are built on top of LangGraph in order to provide durable execution, streaming, human-in-the-loop, persistence, and more. (You do not need to know LangGraph for basic LangChain agent usage.)
"""
splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap = 0,
    separator = ""
)

result = splitter.split_text(text)

print(result)
for i in range(6):
    print("Split:", i, "->", result[i])

loader = PyPDFLoader("AI.pdf")
docs = loader.load()
result = splitter.split_documents(docs)
print("result")
for i in range(6):
    print("Split:", i, "->", result[i].page_content)

#Use chunk overlap to avoid missing context usually 10-20% of chunk overlap