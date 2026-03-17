from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from ingest import get_vectorstore
from prompts import SYSTEM_PROMPT


def build_rag_chain():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    retriever = get_vectorstore().as_retriever(search_kwargs={"k": 5})
    prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)

    def format_docs(docs):
        return "\n\n---\n\n".join(
            f"[Source: {d.metadata.get('source', '?')}]\n{d.page_content}" for d in docs
        )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


if __name__ == "__main__":
    chain = build_rag_chain()

    while True:
        q = input("\nAsk: ").strip()
        if q.lower() == "exit":
            break

        print(chain.invoke(q))
