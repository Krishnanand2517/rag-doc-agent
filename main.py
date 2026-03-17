from dotenv import load_dotenv
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.live import Live
import typer

load_dotenv()
from agent import agent  # noqa: E402


console = Console()
app = typer.Typer()


def stream_answer(question: str):
    with Live(Spinner("dots", text="Thinking..."), console=console, transient=True):
        result = agent.run_sync(question)

    console.print(
        Panel(
            Markdown(result.output),
            title="[bold purple]Agent[/bold purple]",
            border_style="purple",
            padding=(1, 2),
        )
    )


@app.command()
def chat():
    console.print(
        Panel(
            "[bold]RAG Agent - Personal Knowledge Base[/bold]\n"
            "[dim]Type your question, or use commands below:[/dim]\n\n"
            "  [cyan]/ingest <path>[/cyan]   Add a PDF or URL\n"
            "  [cyan]/docs[/cyan]            List ingested documents\n"
            "  [cyan]/clear[/cyan]           Clear the screen\n"
            "  [cyan]/exit[/cyan]            Quit",
            border_style="dim",
            padding=(1, 2),
        )
    )

    session = PromptSession(
        history=FileHistory(".rag_history"), auto_suggest=AutoSuggestFromHistory()
    )

    while True:
        try:
            user_input: str = session.prompt("\n[You] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not user_input:
            continue

        if user_input.startswith("/ingest "):
            source = user_input[8:].strip()
            source_type = "web" if source.startswith("http") else "pdf"

            from ingest import ingest

            ingest(source, source_type)

        elif user_input == "/docs":
            from agent import _list_documents_impl

            console.print(
                Panel(
                    _list_documents_impl(),
                    title="Ingested documents",
                    border_style="dim",
                )
            )

        elif user_input == "/clear":
            console.clear()

        elif user_input in ("/quit", "/exit"):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        else:
            stream_answer(user_input)


if __name__ == "__main__":
    app()
