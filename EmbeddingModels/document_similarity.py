from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=200)

documents = [
    "Mithali Raj is India’s highest‑run scorer in women’s international cricket and a global icon of consistency.",
    "Harmanpreet Kaur, the current captain, is known for her explosive 171* in the 2017 World Cup semifinal.",
    "Smriti Mandhana is celebrated for her elegant left‑handed stroke play and match‑winning batting at the top.",
    "Jhulan Goswami, one of the fastest and most successful women pacers ever, led India’s bowling attack for two decades.",
    "Shafali Verma became famous as a teenager for her fearless power‑hitting and record‑breaking starts in T20Is."
]

query = 'tell me about Shafali Verma'
doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)
scores = cosine_similarity([query_embedding], doc_embeddings)[0]
index, score = sorted(list(enumerate(scores)), key=lambda x:x[1])[-1]

print("Query:", query)
print(documents[index])
print("Similarity score is: ", score)