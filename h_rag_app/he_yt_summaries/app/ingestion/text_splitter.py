from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_text(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    return splitter.create_documents([text])