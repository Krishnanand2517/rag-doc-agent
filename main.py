from dotenv import load_dotenv
import asyncio
from collections import Counter
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.live import Live
from rich.spinner import Spinner
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule
import typer

load_dotenv()
from agent import agent  # noqa: E402


console = Console()
app = typer.Typer()

# One persistent loop for the entire session
_loop = asyncio.new_event_loop()
asyncio.set_event_loop(_loop)


def stream_answer(question: str):
    async def _stream():
        content = ""
        tools_used = []
        seen = set()

        with Live(
            Panel(
                Spinner("dots", text="Thinking..."),
                title="Agent",
                border_style="purple",
            ),
            console=console,
            refresh_per_second=10,
        ) as live:
            async with agent.run_stream(question) as response:
                async for chunk in response.stream_text(delta=True):
                    content += chunk

                    # Check for new tool usage during streaming
                    for msg in response.all_messages():
                        for part in getattr(msg, "parts", []):
                            tool_name = getattr(part, "tool_name", None)

                            if tool_name and tool_name not in seen:
                                seen.add(tool_name)
                                tools_used.append(tool_name)

                    tool_display = ""

                    if tools_used:
                        counts = Counter(tools_used)

                        tool_display = "\n\nTools used: " + ", ".join(
                            f"{tool} ({count})" if count > 1 else tool
                            for tool, count in counts.items()
                        )

                    renderable = Group(
                        Markdown(content),
                        Rule(style="dim") if tool_display else Text(""),
                        Text(tool_display, style="dim") if tool_display else Text(""),
                    )

                    live.update(
                        Panel(
                            renderable,
                            title="Agent",
                            border_style="purple",
                            padding=(1, 2),
                        )
                    )

    try:
        # Reuse the same loop
        _loop.run_until_complete(_stream())
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")


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
