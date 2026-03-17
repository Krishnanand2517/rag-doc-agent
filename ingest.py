from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
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
    vs.add_documents(chunks)
    print(f"[green]Ingested {len(chunks)} chunks from {source}[/green]")


if __name__ == "__main__":
    app()
