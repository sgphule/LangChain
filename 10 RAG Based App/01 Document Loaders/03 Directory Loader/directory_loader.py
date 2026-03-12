from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path = "data",
    glob = "*.pdf",
    loader_cls = PyPDFLoader
)

docs = loader.load()
print("Len(docs):", len(docs))

print("First book details")
print("docs[0].page_content:", docs[0].page_content)
print("docs[0].metadata:", docs[0].metadata)

print("Second book details")
print("docs[326].page_content:", docs[326].page_content)
print("docs[326].metadata:", docs[326].metadata)

# docs = loader.lazy_load()
# for doc in docs:
#    print(doc.metadata)

# Total Pages of first book: 326
# Total Pages of second book: 52
# Total pages of both books: 378

# glob pattern
# "**/*.txt" loads all txt files in all subfolders
# "*.pdf" loads all .pdf files in the root directory
# "data/*.csv" loads all .csv file in the data directory