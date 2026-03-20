from langchain_core.prompts import ChatPromptTemplate

REWRITE_QUERY_PROMPT = """You are helping rewrite a search query for a local document retrieval system.
The user's knowledge base contains ingested documents — NOT external URLs or web resources.
Any reference to "that paper", "the document", "that file" means something already ingested locally.

Recent conversation:
{history_text}

Rewrite the query below into a precise, keyword-rich search query.
Resolve vague references using the conversation history.
Return ONLY the rewritten query, nothing else.

Original: {vague_query}
Rewritten:"""


MULTI_QUERY_PROMPT = """Generate 3 different search queries to find information about: {query}
Return only the queries, one per line."""


HYDE_PROMPT = """Write a short paragraph that would be the ideal answer to this question.
Be factual-sounding even if you're guessing. Do not say you don't know.

Question: {question}
Ideal answer:"""


RAG_PROMPT_TEMPLATE = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer based ONLY on the context below.
If the answer isn't in the context, say "I don't have that information."
Always cite which document/chunk your answer comes from.

Context:
{context}

Question: {question}
""")


SYSTEM_PROMPT = """
ou are a research assistant with a personal knowledge base.

Available documents:
{doc_summary}

Rules:
- For specific questions, use search_knowledge_base
- For vague or broad questions, use search_knowledge_base_multi  
- If the user says 'the document', 'that file', 'what we discussed' — infer 
  from context which document they likely mean and search accordingly
- Always cite sources
- If you searched and found nothing relevant, say so and suggest a rephrasing
"""


def build_system_prompt(doc_summary: str) -> str:
    return SYSTEM_PROMPT.format(doc_summary=doc_summary)
