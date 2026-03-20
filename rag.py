from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from util import _to_str
from ingest import get_vectorstore
from prompts import RAG_PROMPT_TEMPLATE, HYDE_PROMPT

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def make_hypothetical_doc(question: str) -> str:
    """Generate a fake ideal answer (HyDE) to improve embedding similarity."""
    prompt = HYDE_PROMPT.format(question=question)
    content = llm.invoke(prompt).content

    return _to_str(content)


def build_rag_chain():
    vs = get_vectorstore()
    all_docs = vs.get()

    # Vector Retriever
    vector_retriever = vs.as_retriever(search_kwargs={"k": 5})

    # BM25 Retriever (only if documents exist)
    if all_docs["documents"]:
        bm25_retriever = BM25Retriever.from_texts(
            all_docs["documents"], metadatas=all_docs["metadatas"]
        )
        bm25_retriever.k = 5

        # Hybrid Retriever
        retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever], weights=[0.6, 0.4]
        )
    else:
        retriever = vector_retriever  # fall back to vector-only

    def retrieve_with_hyde(question: str):
        hypothetical = make_hypothetical_doc(question)
        return retriever.invoke(hypothetical)

    def format_docs(docs):
        return "\n\n---\n\n".join(
            f"[Source: {d.metadata.get('source', '?')}]\n{d.page_content}" for d in docs
        )

    chain = (
        {
            "context": RunnableLambda(retrieve_with_hyde) | format_docs,
            "question": RunnablePassthrough(),
        }
        | RAG_PROMPT_TEMPLATE
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
