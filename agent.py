from pydantic_ai import Agent, Tool
from pydantic_ai.models.openai import OpenAIResponsesModel

from rag import build_rag_chain
from ingest import get_vectorstore
from prompts import SYSTEM_PROMPT


model = OpenAIResponsesModel("gpt-4o-mini")
rag_chain = build_rag_chain()


def _list_documents_impl() -> str:
    vs = get_vectorstore()
    sources = set()
    data = vs.get()

    for meta in data["metadatas"]:
        sources.add(meta.get("source", "unknown"))

    return "\n".join(sources) or "No documents ingested yet."


@Tool
def search_knowledge_base(query: str) -> str:
    """Search the personal knowledge base for relevant information."""
    return rag_chain.invoke(query)


@Tool
def list_documents() -> str:
    """List all documents in the knowledge base."""
    return _list_documents_impl()


agent = Agent(
    model=model,
    tools=[search_knowledge_base, list_documents],
    system_prompt=SYSTEM_PROMPT,
)

if __name__ == "__main__":
    result = agent.run_sync("What documents do I have, and what are they about?")
    print(result.output)
