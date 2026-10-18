"""Command-line interface for Codemate with Rich visual terminal output & syntax error diagnostics."""

from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path
from typing import NoReturn

from .ask import SearchHit, ask_repository, retrieve
from .indexer import DEFAULT_MAX_FILE_BYTES, DEFAULT_MODEL, IndexResult, index_repository
from .tracker import TrackerError, find_repo_root, track_repository

try:
    from rich import box
    from rich.console import Console
    from rich.markup import escape
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.tree import Tree

    RICH_AVAILABLE = True
    console = Console()
    err_console = Console(stderr=True)
except ImportError:
    RICH_AVAILABLE = False
    console = None  # type: ignore[assignment]
    err_console = None  # type: ignore[assignment]


BANNER = r"""[bold cyan]
   ____          _                      _       
  / ___|___   __| | ___ _ __ ___   __ _| |_ ___ 
 | |   / _ \ / _` |/ _ \ '_ ` _ \ / _` | __/ _ \
 | |__| (_) | (_| |  __/ | | | | | (_| | ||  __/
  \____\___/ \__,_|\___|_| |_| |_|\__,_|\__\___|[/bold cyan] [bold green]v0.1.0[/bold green]
  [dim]Local Git Tracker • Line-Aware Chunker • SQLite Vector RAG • Grounded Citations[/dim]"""

KNOWN_COMMANDS = ["track", "index", "ask"]
KNOWN_FLAGS = ["--repo", "--model", "--max-file-bytes", "--top-k", "--provider", "--base-url", "--api-key-env", "--help"]


def _suggest_fix(raw_argv: list[str], message: str) -> str | None:
    """Compute a 'Did you mean?' command suggestion for typos and syntax errors."""
    for token in raw_argv:
        if token.startswith("-"):
            matches = difflib.get_close_matches(token.split("=")[0], KNOWN_FLAGS, n=1, cutoff=0.55)
            if matches and matches[0] != token:
                fixed = [matches[0] if t == token else t for t in raw_argv]
                return "codemate " + " ".join(fixed)
        elif token not in KNOWN_COMMANDS:
            matches = difflib.get_close_matches(token, KNOWN_COMMANDS, n=1, cutoff=0.5)
            if matches:
                fixed = [matches[0] if t == token else t for t in raw_argv]
                return "codemate " + " ".join(fixed)
    if "the following arguments are required: question" in message:
        return 'codemate ask "How does index_repository work?"'
    if "the following arguments are required: command" in message:
        return "codemate track   # or: codemate index | codemate ask \"<question>\""
    return None


def _render_error_panel(title: str, error_msg: str, suggestion: str | None = None) -> None:
    if RICH_AVAILABLE and err_console is not None:
        body = f"[bold red]✗ {escape(error_msg)}[/bold red]\n"
        if suggestion:
            body += f"\n[bold yellow]💡 Did you mean:[/bold yellow]\n  [bold green]$ {escape(suggestion)}[/bold green]\n"
        body += (
            "\n[dim]Valid syntax examples:\n"
            "  $ codemate track [--repo PATH]\n"
            "  $ codemate index [--model BAAI/bge-small-en-v1.5] [--max-file-bytes 1000000]\n"
            "  $ codemate ask \"<question>\" [--top-k 6] [--provider gemini|openai-compatible][/dim]"
        )
        err_console.print(
            Panel(
                body,
                title=f"[bold red]{escape(title)}[/bold red]",
                subtitle="[red]Exit Code: 2[/red]",
                border_style="red",
                box=box.ROUNDED,
            )
        )
    else:
        print(f"codemate: error: {error_msg}", file=sys.stderr)
        if suggestion:
            print(f"Did you mean: {suggestion}", file=sys.stderr)


class VisualArgumentParser(argparse.ArgumentParser):
    """Custom ArgumentParser that renders Rich visual syntax error panels with typo suggestions."""

    _raw_argv: list[str] = []

    def error(self, message: str) -> NoReturn:
        suggestion = _suggest_fix(self._raw_argv, message)
        _render_error_panel("CLI Syntax / Argument Error", message, suggestion)
        raise SystemExit(2)


def _score_bar(score: float, width: int = 16) -> str:
    filled = max(0, min(width, int(round(score * width))))
    bar = "█" * filled + "░" * (width - filled)
    color = "green" if score >= 0.75 else ("cyan" if score >= 0.55 else "yellow")
    return f"[{color}]{bar}[/{color}] [bold {color}]{score:.3f}[/bold {color}]"


def _parser() -> VisualArgumentParser:
    parser = VisualArgumentParser(prog="codemate", description="Track a Git codebase locally.")
    subparsers = parser.add_subparsers(dest="command", required=True, parser_class=VisualArgumentParser)
    track = subparsers.add_parser("track", help="snapshot current repository files into .index/")
    track.add_argument("--repo", type=Path, default=Path.cwd(), help="repository or subdirectory (default: current directory)")
    index = subparsers.add_parser("index", help="chunk and embed current text files locally")
    index.add_argument("--repo", type=Path, default=Path.cwd(), help="repository or subdirectory (default: current directory)")
    index.add_argument("--model", default=DEFAULT_MODEL, help=f"local embedding model (default: {DEFAULT_MODEL})")
    index.add_argument(
        "--max-file-bytes",
        type=int,
        default=DEFAULT_MAX_FILE_BYTES,
        metavar="BYTES",
        help=f"skip larger files for embedding (default: {DEFAULT_MAX_FILE_BYTES})",
    )
    ask = subparsers.add_parser("ask", help="ask a question about the indexed codebase")
    ask.add_argument("question", help="question about this repository")
    ask.add_argument("--repo", type=Path, default=Path.cwd(), help="repository or subdirectory (default: current directory)")
    ask.add_argument("--top-k", type=int, default=6, help="number of retrieved chunks to send (default: 6)")
    ask.add_argument("--model", help="chat model name (or set CODEMATE_MODEL in .env)")
    ask.add_argument(
        "--provider",
        choices=("gemini", "openai-compatible"),
        help="chat API provider (default: gemini; or set CODEMATE_PROVIDER in .env)",
    )
    ask.add_argument(
        "--base-url",
        help="provider API base URL (or set CODEMATE_BASE_URL in .env)",
    )
    ask.add_argument(
        "--api-key-env",
        default="GEMINI_API_KEY",
        metavar="ENV_VAR",
        help="environment variable name for the API key (default: GEMINI_API_KEY)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    VisualArgumentParser._raw_argv = raw_argv

    if RICH_AVAILABLE and console is not None:
        console.print(BANNER)

    parser = _parser()
    args = parser.parse_args(raw_argv)

    if args.command == "track":
        try:
            if RICH_AVAILABLE and console is not None:
                with Progress(
                    SpinnerColumn(style="cyan"),
                    TextColumn("[bold cyan]{task.description}"),
                    BarColumn(bar_width=26, complete_style="cyan", finished_style="green"),
                    TimeElapsedColumn(),
                    console=console,
                ) as progress:
                    t = progress.add_task("Scanning Git working-tree & computing SHA-256 manifest...", total=1)
                    result = track_repository(args.repo)
                    progress.advance(t)
            else:
                result = track_repository(args.repo)
        except TrackerError as exc:
            _render_error_panel("Repository Tracker Error", str(exc), "codemate track --repo .")
            return 2

        if RICH_AVAILABLE and console is not None:
            table = Table(title=f"Tracked Snapshot: {result.manifest_path}", box=box.ROUNDED, border_style="cyan")
            table.add_column("Status", style="bold")
            table.add_column("Files", justify="right", style="bold white")
            table.add_column("Sample Paths", style="dim")
            table.add_row("[green]+ Added[/green]", str(len(result.added)), ", ".join(result.added[:5]) or "—")
            table.add_row("[yellow]~ Changed[/yellow]", str(len(result.changed)), ", ".join(result.changed[:5]) or "—")
            table.add_row("[cyan]= Unchanged[/cyan]", str(len(result.unchanged)), ", ".join(result.unchanged[:5]) or "—")
            table.add_row("[red]- Removed[/red]", str(len(result.removed)), ", ".join(result.removed[:5]) or "—")
            table.add_row("[magenta]! Skipped[/magenta]", str(len(result.skipped)), ", ".join(result.skipped[:5]) or "—")
            console.print(table)
        else:
            print(f"Tracked snapshot: {result.manifest_path}")
            print(
                f"{len(result.added)} added, {len(result.changed)} changed, "
                f"{len(result.unchanged)} unchanged, {len(result.removed)} removed"
            )
            if result.skipped:
                print(f"{len(result.skipped)} skipped (directories, submodules, or unreadable paths)")
        return 0

    if args.command == "index":
        try:
            if RICH_AVAILABLE and console is not None:
                with Progress(
                    SpinnerColumn(style="cyan"),
                    TextColumn("[bold cyan]{task.description}"),
                    BarColumn(bar_width=26, complete_style="cyan", finished_style="green"),
                    TimeElapsedColumn(),
                    console=console,
                ) as progress:
                    t = progress.add_task(f"Chunking & embedding repository with {args.model}...", total=1)
                    result: IndexResult = index_repository(args.repo, args.model, args.max_file_bytes)
                    progress.advance(t)
            else:
                result = index_repository(args.repo, args.model, args.max_file_bytes)
        except TrackerError as exc:
            _render_error_panel("Indexing Error", str(exc), "codemate index --max-file-bytes 1000000")
            return 2

        if RICH_AVAILABLE and console is not None:
            table = Table(title="Local Vector Index (.index/vectors.sqlite3)", box=box.ROUNDED, border_style="green")
            table.add_column("Metric", style="bold cyan")
            table.add_column("Value", style="white")
            table.add_row("Embedding Model", f"[magenta]{escape(result.model_name)}[/magenta] (Local CPU)")
            table.add_row("SQLite Database", f"[cyan]{escape(str(result.database_path))}[/cyan]")
            table.add_row("Total Stored Chunks", f"[bold green]{result.chunks}[/bold green] chunks")
            table.add_row(
                "File Breakdown",
                f"[green]{result.added_files} added[/green] • [yellow]{result.updated_files} updated[/yellow] • "
                f"[cyan]{result.unchanged_files} unchanged[/cyan] • [red]{result.removed_files} removed[/red] • "
                f"[dim]{result.skipped_files} skipped[/dim]",
            )
            console.print(table)
        else:
            print(f"Indexed with {result.model_name}: {result.chunks} chunks in {result.database_path}")
            print(
                f"{result.added_files} added, {result.updated_files} updated, "
                f"{result.unchanged_files} unchanged, {result.removed_files} removed, "
                f"{result.skipped_files} skipped"
            )
        return 0

    if args.command == "ask":
        hits: list[SearchHit] = []
        embedding_model = DEFAULT_MODEL
        answer = ""
        llm_error: str | None = None
        try:
            answer, hits, embedding_model = ask_repository(
                args.question,
                args.repo,
                top_k=args.top_k,
                base_url=args.base_url,
                model=args.model,
                api_key_env=args.api_key_env,
                provider=args.provider,
            )
        except TrackerError as exc:
            try:
                root = find_repo_root(args.repo.resolve())
                hits, embedding_model = retrieve(args.question, root, top_k=args.top_k)
                llm_error = str(exc)
            except TrackerError as inner_exc:
                _render_error_panel(
                    "Query / Retrieval Validation Error",
                    str(inner_exc),
                    'codemate ask "How does index_repository work?" --top-k 5',
                )
                return 2

        if RICH_AVAILABLE and console is not None:
            tree = Tree(f"[bold cyan]RAG Retrieval Pipeline[/bold cyan] [dim](model: {escape(embedding_model)})[/dim]")
            for hit in hits:
                tree.add(
                    f"[bold white]\\[{hit.label}][/bold white] "
                    f"[cyan]{escape(hit.path)}:{hit.start_line}-{hit.end_line}[/cyan]  "
                    f"{_score_bar(hit.score)}"
                )
            console.print(tree)
            console.print()

            if answer:
                console.print(
                    Panel(
                        escape(answer),
                        title="[bold green]Grounded Answer[/bold green]",
                        border_style="green",
                    )
                )
            elif llm_error:
                top = hits[0]
                console.print(
                    Panel(
                        f"[bold yellow]Local Vector Retrieval Complete[/bold yellow] [dim]({escape(llm_error)})[/dim]\n\n"
                        f"Top matching code chunk is [bold cyan]\\[{top.label}] {escape(top.path)}:{top.start_line}-{top.end_line}[/bold cyan] "
                        f"with cosine similarity [bold green]{top.score:.3f}[/bold green].",
                        title="[bold cyan]Local RAG Result[/bold cyan]",
                        border_style="cyan",
                    )
                )

            for hit in hits[:3]:
                ext = Path(hit.path).suffix.lstrip(".") or "text"
                lang = {"py": "python", "md": "markdown", "toml": "toml", "html": "html", "js": "javascript"}.get(ext, "text")
                snippet = "\n".join(hit.text.splitlines()[:16])
                syntax = Syntax(snippet, lang, line_numbers=True, start_line=hit.start_line, theme="monokai")
                console.print(
                    Panel(
                        syntax,
                        title=f"[bold cyan]\\[{hit.label}] {escape(hit.path)}:{hit.start_line}-{hit.end_line}[/bold cyan] • score {hit.score:.3f}",
                        border_style="bright_black",
                    )
                )
        else:
            if answer:
                print(answer)
            elif llm_error:
                print(f"Note: {llm_error}")
            print(f"\nSources (retrieved locally with {embedding_model}):")
            for hit in hits:
                print(f"[{hit.label}] {hit.path}:{hit.start_line}-{hit.end_line} (score {hit.score:.3f})")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
