from langchain_openai import ChatOpenAI
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_chain(retriever, prompt):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)

    parallel = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })

    return parallel | prompt | llm | StrOutputParser()