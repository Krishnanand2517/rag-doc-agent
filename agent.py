from pydantic_ai import Agent, Tool
from pydantic_ai.models.openai import OpenAIResponsesModel

from util import _to_str
from rag import build_rag_chain, llm
from ingest import get_vectorstore
from prompts import build_system_prompt, REWRITE_QUERY_PROMPT, MULTI_QUERY_PROMPT


model = OpenAIResponsesModel("gpt-4o-mini")
rag_chain = build_rag_chain()

conversation_history = []


def rewrite_query(vague_query: str) -> str:
    """Turn a vague user question into a precise search query."""
    history_text = (
        "\n".join(f"{role}: {msg}" for role, msg in conversation_history[-6:])
        or "No prior conversation."
    )

    prompt = REWRITE_QUERY_PROMPT.format(
        history_text=history_text, vague_query=vague_query
    )
    content = llm.invoke(prompt).content

    return _to_str(content).strip()


def _list_documents_impl() -> str:
    vs = get_vectorstore()
    sources = set()
    data = vs.get()

    for meta in data["metadatas"]:
        sources.add(meta.get("source", "unknown"))

    return "\n".join(sources) or "No documents ingested yet."


@Tool
def search_knowledge_base(query: str) -> str:
    """Search the personal knowledge base. Accepts natural language questions."""
    precise_query = rewrite_query(query)
    return rag_chain.invoke(precise_query)


@Tool
def search_knowledge_base_multi(query: str) -> str:
    """Use this for broad or ambiguous questions — searches from multiple angles."""
    prompt = MULTI_QUERY_PROMPT.format(query=query)
    content = llm.invoke(prompt).content
    text = _to_str(content)

    variants = [v.strip() for v in text.split("\n") if v.strip()]

    seen, results = set(), []

    for v in variants[:3]:
        docs = rag_chain.invoke(v)
        if docs not in seen:
            seen.add(docs)
            results.append(docs)

    return "\n\n===\n\n".join(results)


@Tool
def list_documents() -> str:
    """List all documents in the knowledge base."""
    return _list_documents_impl()


agent = Agent(
    model=model,
    tools=[search_knowledge_base, list_documents],
    system_prompt=build_system_prompt(_list_documents_impl()),
)

if __name__ == "__main__":
    result = agent.run_sync("What documents do I have, and what are they about?")
    print(result.output)
