from langchain_core.prompts import PromptTemplate

def get_prompt():
    return PromptTemplate(
        template="""
        Answer only from the provided context.
        If context is insufficient, just say you don't know.

        {context}
        Question: {question}
        """,
        input_variables=["context", "question"]
    )