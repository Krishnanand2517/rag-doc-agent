import asyncio
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from rich import print
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TaskProgressColumn,
    TextColumn,
)
import typer


app = typer.Typer()

CHROMA_PATH = "./chroma_db"


def get_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)


async def embed_chunks_async(chunks, batch_size=20):
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

    batches = [chunks[i : i + batch_size] for i in range(0, len(chunks), batch_size)]
    texts = [[c.page_content for c in b] for b in batches]

    # Send all embedding requests at once
    results = await asyncio.gather(
        *[embeddings_model.aembed_documents(t) for t in texts]
    )

    embedded = []
    for batch, vectors in zip(batches, results):
        for chunk, vector in zip(batch, vectors):
            embedded.append((chunk, vector))

    return embedded


@app.command()
async def ingest(source: str, source_type: str = "pdf"):
    if source_type == "pdf":
        loader = PyPDFLoader(source)
    else:
        loader = WebBaseLoader(source)

    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(docs)

    vs = get_vectorstore()

    print(f"\n[bold]Embedding {source}...[/bold]")

    with Progress(
        SpinnerColumn(),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("{task.completed}/{task.total} chunks"),
    ) as progress:
        task = progress.add_task(f"Embedding {source}...", total=len(chunks))

        embedded = await embed_chunks_async(chunks)
        batch_size = 20

        for i in range(0, len(embedded), batch_size):
            batch = embedded[i : i + batch_size]
            vs._collection.add(
                ids=[str(hash(c.page_content)) for c, _ in batch],
                embeddings=[v for _, v in batch],
                documents=[c.page_content for c, _ in batch],
                metadatas=[c.metadata for c, _ in batch],
            )
            progress.advance(task, len(batch))

    print(f"[green]Done. {len(chunks)} chunks ingested from {source}[/green]")

    # Rebuild so BM25 includes the new documents
    import agent as agent_module

    agent_module.rag_chain = agent_module.build_rag_chain()


if __name__ == "__main__":
    app()
