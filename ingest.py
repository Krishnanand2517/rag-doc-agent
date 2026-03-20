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


@app.command()
def ingest(source: str, source_type: str = "pdf"):
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

        batch_size = 20
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            vs.add_documents(batch)
            progress.advance(task, len(batch))

    print(f"[green]Done. {len(chunks)} chunks ingested from {source}[/green]")

    # Rebuild so BM25 includes the new documents
    import agent as agent_module

    agent_module.rag_chain = agent_module.build_rag_chain()


if __name__ == "__main__":
    app()
