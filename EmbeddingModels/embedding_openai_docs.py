from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=32)
documents = [
    "Berlin is the capital of Germany",
    "Rome is the capital of Italy",
    "Paris is the capital of France"
]
result = embedding.embed_documents(documents)

print(str(result))
