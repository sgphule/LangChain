from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("AI.pdf")

docs = loader.load()
print("Length(docs):", len(docs))
print("Content:\n", docs[0].page_content)
print("Metadata:\n", docs[0].metadata)

# PyPDFLoader useful with textual data but doesn't work well with scanned PDFs
# For Tabular structure PDFPlumberLoader is useful
# For Scanned PDF/Images UnstructuredPDFLoader / AmazonTextractPDFLoader
# Layout or image data use PyMuPDFLoader
# For best structure extraction use UnstructuredPDFLoader

