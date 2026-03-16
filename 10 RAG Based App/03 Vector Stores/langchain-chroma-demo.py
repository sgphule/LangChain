from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

doc1 = Document(
    page_content="Rachin Ravindra is one of the most exciting young all‑rounders entering the IPL with huge expectations. "
                 "His explosive batting and handy left‑arm spin make him a perfect T20 asset. "
                 "CSK is expected to use him as a dynamic top‑order option.",
    metadata={"team":"Chennai Super Kings"}
)

doc2 = Document(
    page_content="A young Afghan mystery spinner, Noor Ahmad has already impressed in global leagues and now steps into a bigger role in the IPL. "
                 "His variations and control make him a strong middle‑overs weapon. "
                 "CSK sees him as a long‑term spin investment. ",
    metadata={"team":"Chennai Super Kings"}
)
doc3 = Document(
    page_content="Often called “Baby Malinga,” Pathirana is still new and rapidly rising. "
                 "His slingy action and deadly yorkers make him a specialist in death overs. "
                 "With more responsibility this season, he’s expected to become one of the breakout fast bowlers.",
    metadata={"team":"Chennai Super Kings"}
)

doc4 = Document(
    page_content="Khaleel is re‑entering the IPL spotlight with renewed form and rhythm. "
                 "As a left‑arm pacer, he brings variety and early‑wicket potential. "
                 "CSK is likely to use him in powerplay overs to exploit swing.",
    metadata={"team":"Chennai Super Kings"}
)
doc5 = Document(
    page_content="While not brand‑new, Gaikwad steps into a new era as CSK’s captain, making him a fresh strategic presence. "
                 "His calm leadership and consistent batting will shape the team’s new identity post‑Dhoni era.",
    metadata={"team":"Chennai Super Kings"}
)
doc6 = Document(
    page_content="Ben Stokes was one of the biggest stars for the Pune franchise. "
                 "He delivered match‑winning performances with both bat and ball, including a memorable century in 2017. "
                 "His all‑round impact played a major role in taking RPS to the IPL final that season.",
metadata={"team":"Rising Pune Supergiant"}
)

docs = [doc1, doc2, doc3, doc4, doc5, doc6]

vector_store = Chroma(
    embedding_function=OpenAIEmbeddings(),
    persist_directory="chroma_db4",
    collection_name="sample"
)

#vector_store.add_documents(docs)
#print(vector_store.get(include=["embeddings", "documents", "metadata"]))
print(vector_store._collection.get(include=["embeddings", "documents", "metadatas"]))
"""
{'ids': ['ea74fb03-f50f-4cec-ada8-53138181b0a9', 
        '4e230f40-e671-47e5-a24e-b9217ffe405b', 
        '232fa5ad-d940-4ad8-89bd-af9b087ec306', 
        '0d4dd692-7685-4992-9417-640e5d346775', 
        '57863857-f2f4-47da-ba95-0c8c6c88e510', 
        'ec0ccde9-2c11-4599-8b25-3ea12aaca505'], 
        'embeddings': array([[-0.01598285,  0.00993116,  0.03305514, ..., -0.0061148 ,
        -0.00872879,  0.01234253],
       [ 0.00059185, -0.00605186,  0.02440749, ..., -0.00512491,
         0.00758233, -0.01009643],
       [-0.02023063,  0.00527504,  0.02404849, ..., -0.00663372,
        -0.00488782,  0.00377711],
       [ 0.0037596 ,  0.00810373,  0.03831339, ...,  0.00817016,
         0.01323167, -0.00296749],
       [ 0.00275778,  0.00662664,  0.01739244, ..., -0.00849059,
        -0.00208119,  0.00943251],
       [ 0.01243577,  0.00637843,  0.02695866, ..., -0.01931926,
        -0.0150246 , -0.01879748]], shape=(6, 1536)), 
        'documents': [
            'Rachin Ravindra is one of the most exciting young all‑rounders entering the IPL with huge expectations. His explosive batting and handy left‑arm spin make him a perfect T20 asset. CSK is expected to use him as a dynamic top‑order option.', 
            'A young Afghan mystery spinner, Noor Ahmad has already impressed in global leagues and now steps into a bigger role in the IPL. His variations and control make him a strong middle‑overs weapon. CSK sees him as a long‑term spin investment. ', 
            'Often called “Baby Malinga,” Pathirana is still new and rapidly rising. His slingy action and deadly yorkers make him a specialist in death overs. With more responsibility this season, he’s expected to become one of the breakout fast bowlers.', 
            'Khaleel is re‑entering the IPL spotlight with renewed form and rhythm. As a left‑arm pacer, he brings variety and early‑wicket potential. CSK is likely to use him in powerplay overs to exploit swing.', 
            'While not brand‑new, Gaikwad steps into a new era as CSK’s captain, making him a fresh strategic presence. His calm leadership and consistent batting will shape the team’s new identity post‑Dhoni era.', 
            'Ben Stokes was one of the biggest stars for the Pune franchise. He delivered match‑winning performances with both bat and ball, including a memorable century in 2017. His all‑round impact played a major role in taking RPS to the IPL final that season.'], 
        'uris': None, 'included': ['embeddings', 'documents', 'metadatas'], 'data': None, 
        'metadatas': [{'team': 'Chennai Super Kings'}, 
                      {'team': 'Chennai Super Kings'}, 
                      {'team': 'Chennai Super Kings'}, 
                      {'team': 'Chennai Super Kings'}, 
                      {'team': 'Chennai Super Kings'}, 
                      {'team': 'Rising Pune Supergiant'}]}
"""
result = vector_store.similarity_search(query = "Who among these is a captain", k=1)
print("search result:", result)
"""
search result: [
Document(
id='57863857-f2f4-47da-ba95-0c8c6c88e510', 
metadata={'team': 'Chennai Super Kings'}, 
page_content='While not brand‑new, Gaikwad steps into a new era as CSK’s captain, making him a fresh strategic presence. 
              His calm leadership and consistent batting will shape the team’s new identity post‑Dhoni era.')]
"""
bowlers = vector_store.similarity_search(query="Who among these are bowlers", k=3)
print("Bowlers:", bowlers)
"""
Bowlers: [
Document(id='232fa5ad-d940-4ad8-89bd-af9b087ec306', metadata={'team': 'Chennai Super Kings'}, page_content='Often called “Baby Malinga,” Pathirana is still new and rapidly rising. His slingy action and deadly yorkers make him a specialist in death overs. With more responsibility this season, he’s expected to become one of the breakout fast bowlers.'), 
Document(id='4e230f40-e671-47e5-a24e-b9217ffe405b', metadata={'team': 'Chennai Super Kings'}, page_content='A young Afghan mystery spinner, Noor Ahmad has already impressed in global leagues and now steps into a bigger role in the IPL. His variations and control make him a strong middle‑overs weapon. CSK sees him as a long‑term spin investment. '), 
Document(id='ec0ccde9-2c11-4599-8b25-3ea12aaca505', metadata={'team': 'Rising Pune Supergiant'}, page_content='Ben Stokes was one of the biggest stars for the Pune franchise. He delivered match‑winning performances with both bat and ball, including a memorable century in 2017. His all‑round impact played a major role in taking RPS to the IPL final that season.')]
"""
#search_with_id = vector_store.similarity_search_by_vector(025cef51-acd5-4fe7-8713-5e7ccdd50abd)
search_with_score = vector_store.similarity_search_with_score(query="Who among these are bowlers",k=3)
print("bowler search with score:", search_with_score)
"""
bowler search with score: [
(Document(id='232fa5ad-d940-4ad8-89bd-af9b087ec306', metadata={'team': 'Chennai Super Kings'}, page_content='Often called “Baby Malinga,” Pathirana is still new and rapidly rising. His slingy action and deadly yorkers make him a specialist in death overs. With more responsibility this season, he’s expected to become one of the breakout fast bowlers.'), 0.36767229437828064), 
(Document(id='4e230f40-e671-47e5-a24e-b9217ffe405b', metadata={'team': 'Chennai Super Kings'}, page_content='A young Afghan mystery spinner, Noor Ahmad has already impressed in global leagues and now steps into a bigger role in the IPL. His variations and control make him a strong middle‑overs weapon. CSK sees him as a long‑term spin investment. '), 0.4152173101902008), 
(Document(id='ec0ccde9-2c11-4599-8b25-3ea12aaca505', metadata={'team': 'Rising Pune Supergiant'}, page_content='Ben Stokes was one of the biggest stars for the Pune franchise. He delivered match‑winning performances with both bat and ball, including a memorable century in 2017. His all‑round impact played a major role in taking RPS to the IPL final that season.'), 0.4215709865093231)]
"""
search_with_filter = vector_store.similarity_search(query="", filter={"team":"Rising Pune Supergiant"})
print("filter result:", search_with_filter)
"""
filter result: [
Document(
id='ec0ccde9-2c11-4599-8b25-3ea12aaca505', 
metadata={'team': 'Rising Pune Supergiant'}, 
page_content='Ben Stokes was one of the biggest stars for the Pune franchise. He delivered match‑winning performances with both bat and ball, including a memorable century in 2017. His all‑round impact played a major role in taking RPS to the IPL final that season.')]
"""
updated_doc6 = Document(
    page_content="Ben Stokes was one of the biggest stars for the Pune franchise. "
                 "He delivered match‑winning performances with both bat and ball, including a memorable century in 2017. "
                 "His all‑round impact played a major role in taking RPS to the IPL final that season."
                 "This Pune team doesn't exist anymore",
    metadata={"team":"Rising Pune Supergiant"}
)
vector_store.update_document(document_id='ec0ccde9-2c11-4599-8b25-3ea12aaca505', document=updated_doc6)
new_docs = vector_store.similarity_search(query="", filter={"team":"Rising Pune Supergiant"})
print("New docs:", new_docs)
"""
New docs: [
Document(id='ec0ccde9-2c11-4599-8b25-3ea12aaca505', metadata={'team': 'Rising Pune Supergiant'}, 
page_content="Ben Stokes was one of the biggest stars for the Pune franchise. 
He delivered match‑winning performances with both bat and ball, including a memorable century in 2017. 
His all‑round impact played a major role in taking RPS to the IPL final that season.
This Pune team doesn't exist anymore")]
"""
vector_store.delete(ids=['ec0ccde9-2c11-4599-8b25-3ea12aaca505'])
print(vector_store._collection.get(include=["embeddings", "documents", "metadatas"]))
"""
{'ids': [
            'ea74fb03-f50f-4cec-ada8-53138181b0a9', 
            '4e230f40-e671-47e5-a24e-b9217ffe405b', 
            '232fa5ad-d940-4ad8-89bd-af9b087ec306', 
            '0d4dd692-7685-4992-9417-640e5d346775', 
            '57863857-f2f4-47da-ba95-0c8c6c88e510'], 
'embeddings': array([[-0.01598285,  0.00993116,  0.03305514, ..., -0.0061148 ,
        -0.00872879,  0.01234253],
       [ 0.00059185, -0.00605186,  0.02440749, ..., -0.00512491,
         0.00758233, -0.01009643],
       [-0.02023063,  0.00527504,  0.02404849, ..., -0.00663372,
        -0.00488782,  0.00377711],
       [ 0.0037596 ,  0.00810373,  0.03831339, ...,  0.00817016,
         0.01323167, -0.00296749],
       [ 0.00275778,  0.00662664,  0.01739244, ..., -0.00849059,
        -0.00208119,  0.00943251]], shape=(5, 1536)), 
'documents': [
                'Rachin Ravindra is one of the most exciting young all‑rounders entering the IPL with huge expectations. His explosive batting and handy left‑arm spin make him a perfect T20 asset. CSK is expected to use him as a dynamic top‑order option.', 'A young Afghan mystery spinner, Noor Ahmad has already impressed in global leagues and now steps into a bigger role in the IPL. His variations and control make him a strong middle‑overs weapon. CSK sees him as a long‑term spin investment. ', 
                'Often called “Baby Malinga,” Pathirana is still new and rapidly rising. His slingy action and deadly yorkers make him a specialist in death overs. With more responsibility this season, he’s expected to become one of the breakout fast bowlers.', 'Khaleel is re‑entering the IPL spotlight with renewed form and rhythm. As a left‑arm pacer, he brings variety and early‑wicket potential. CSK is likely to use him in powerplay overs to exploit swing.', 
                'While not brand‑new, Gaikwad steps into a new era as CSK’s captain, making him a fresh strategic presence. His calm leadership and consistent batting will shape the team’s new identity post‑Dhoni era.'], 
'uris': None, 
'included': ['embeddings', 'documents', 'metadatas'], 
'data': None, 
'metadatas': [{'team': 'Chennai Super Kings'}, 
              {'team': 'Chennai Super Kings'}, 
              {'team': 'Chennai Super Kings'}, 
              {'team': 'Chennai Super Kings'}, 
              {'team': 'Chennai Super Kings'}]}
"""
