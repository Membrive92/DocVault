"""
Interactive CLI for the DocVault RAG pipeline.

Provides a REPL interface with rich formatting for querying documents,
viewing sources, and getting AI-generated answers with citations.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from src.rag import RAGPipeline, RAGResponse

logger = logging.getLogger(__name__)

console = Console()


class InteractiveCLI:
    """
    Interactive command-line interface for the RAG pipeline.

    Args:
        rag_pipeline: Pre-configured RAG pipeline instance.
            If None, creates a new one with default settings.

    Example:
        >>> cli = InteractiveCLI()
        >>> cli.run()
    """

    def __init__(self, rag_pipeline: Optional[RAGPipeline] = None) -> None:
        self.pipeline = rag_pipeline

    def _ensure_pipeline(self) -> None:
        """Initialize the pipeline lazily on first use."""
        if self.pipeline is None:
            console.print("[dim]Initializing RAG pipeline...[/dim]")
            self.pipeline = RAGPipeline()

    def print_banner(self) -> None:
        """Display the welcome banner with model info and commands."""
        self._ensure_pipeline()

        model_info = self.pipeline.llm.get_model_info()
        provider = model_info.get("provider", "unknown")
        model = model_info.get("model", "unknown")

        banner = (
            "[bold cyan]DocVault[/bold cyan] - Document Question Answering\n\n"
            f"[dim]Provider:[/dim] {provider}\n"
            f"[dim]Model:[/dim]    {model}\n\n"
            "[bold]Commands:[/bold]\n"
            "  [green]/sources[/green]  - Show indexed document info\n"
            "  [green]/help[/green]     - Show this help message\n"
            "  [green]/exit[/green]     - Exit the application\n\n"
            "[dim]Type your question and press Enter.[/dim]"
        )

        console.print(Panel(banner, title="DocVault CLI", border_style="cyan"))

    def run(self) -> None:
        """Run the interactive REPL loop."""
        try:
            self.print_banner()
        except Exception as e:
            console.print(f"[bold red]Failed to initialize pipeline:[/bold red] {e}")
            sys.exit(1)

        console.print()

        while True:
            try:
                user_input = Prompt.ask("[bold cyan]>[/bold cyan]")

                if not user_input.strip():
                    continue

                if user_input.strip().startswith("/"):
                    should_exit = self.handle_command(user_input.strip())
                    if should_exit:
                        break
                else:
                    self.execute_query(user_input.strip())

                console.print()

            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]Goodbye![/dim]")
                break

    def execute_query(self, query: str) -> None:
        """Execute a query and display the formatted response."""
        console.print("[dim]Searching and generating...[/dim]")

        try:
            response = self.pipeline.query(query_text=query)

            self._display_response(response)

        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")

    def _display_response(self, response: RAGResponse) -> None:
        """Display a RAG response with rich formatting."""
        # Answer panel
        console.print(
            Panel(
                Markdown(response.answer),
                title="Answer",
                border_style="green",
            )
        )

        # Sources
        if response.sources:
            console.print(f"\n[bold]Sources[/bold] ({response.retrieval_count} chunks):")
            for i, source in enumerate(response.sources, 1):
                score_color = "green" if source.similarity_score > 0.7 else "yellow"
                console.print(
                    f"  {i}. [dim]{source.source_file}[/dim] "
                    f"[{score_color}](score: {source.similarity_score:.2f})[/{score_color}]"
                )

        # Timing
        parts = []
        if response.retrieval_time_ms is not None:
            parts.append(f"retrieval: {response.retrieval_time_ms:.0f}ms")
        if response.generation_time_ms is not None:
            parts.append(f"generation: {response.generation_time_ms:.0f}ms")
        if parts:
            console.print(f"[dim]{' | '.join(parts)} | model: {response.model_used}[/dim]")

    def handle_command(self, command: str) -> bool:
        """
        Handle a CLI command.

        Returns:
            True if the CLI should exit, False otherwise.
        """
        cmd = command.lower()

        if cmd in ("/exit", "/quit"):
            console.print("[dim]Goodbye![/dim]")
            return True

        if cmd == "/sources":
            try:
                info = self.pipeline.get_indexed_sources()
                console.print(Panel(
                    f"[bold]Collection:[/bold] {info.get('collection_name', 'N/A')}\n"
                    f"[bold]Vectors:[/bold]    {info.get('vectors_count', 0)}\n"
                    f"[bold]Dimensions:[/bold] {info.get('vector_size', 'N/A')}\n"
                    f"[bold]Metric:[/bold]     {info.get('distance_metric', 'N/A')}\n"
                    f"[bold]Status:[/bold]     {info.get('status', 'N/A')}",
                    title="Indexed Sources",
                    border_style="blue",
                ))
            except Exception as e:
                console.print(f"[bold red]Error:[/bold red] {e}")
            return False

        if cmd == "/help":
            self.print_banner()
            return False

        console.print(f"[yellow]Unknown command:[/yellow] {command}")
        console.print("[dim]Type /help for available commands.[/dim]")
        return False


if __name__ == "__main__":
    cli = InteractiveCLI()
    cli.run()
