"""
Codemate Rich CLI & Local Web UI Server — Synced with Real `codemate` Backend
(`tracker`, `chunker`, `indexer`, `ask`, `diagnostics`, `evaluation`, `failure_atlas`, `git_guard`, `agent`, `config`).

Commands:
    python codemate_ui.py track
    python codemate_ui.py index
    python codemate_ui.py ask "how does chunk_text split files?"
    python codemate_ui.py doctor
    python codemate_ui.py evaluate
    python codemate_ui.py failures
    python codemate_ui.py explain codemate/indexer.py
    python codemate_ui.py serve   # Launches live Web UI connected to real backend at http://127.0.0.1:8765
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import subprocess
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List

import typer
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.syntax import Syntax
from rich.table import Table
from rich.tree import Tree

# ============================================================================
# REAL BACKEND IMPORTS FROM `codemate` PACKAGE
# ============================================================================
import codemate.ask as cm_ask
import codemate.indexer as cm_indexer
from codemate.ask import SearchHit, _load_dotenv, ask_repository, retrieve
from codemate.config import DEFAULT_MODEL as DEFAULT_CHAT_MODEL, DEFAULT_PROVIDER, config_path, load_config
from codemate.diagnostics import run_diagnostics
from codemate.evaluation import evaluate_retrieval
from codemate.failure_atlas import list_incidents
from codemate.indexer import DEFAULT_MAX_FILE_BYTES, DEFAULT_MODEL, IndexResult, index_repository
from codemate.tracker import TrackerError, TrackResult, find_repo_root, track_repository

app = typer.Typer(
    name="codemate",
    help="Codemate Rich CLI & Web UI — Synced with real codemate backend modules.",
    add_completion=False,
)
console = Console()


# ============================================================================
# PYTHON 3.14 FALLBACK EMBEDDER (If onnxruntime/fastembed wheel is unavailable)
# ============================================================================

class LocalCodeEmbedderFallback:
    """
    Deterministic 384-dimensional local code & text embedder.
    Used automatically only if `fastembed` / `onnxruntime` lacks a wheel,
    so `codemate.indexer.index_repository` and `codemate.ask.retrieve` still execute
    100% of their real SQLite (.index/vectors.sqlite3) vector pipeline.
    """

    DIMENSIONS = 384
    STOPWORDS = {
        "the", "is", "in", "at", "of", "on", "and", "or", "to", "for", "with", "by",
        "how", "where", "what", "when", "why", "does", "do", "are", "be", "this",
        "that", "from", "as", "an", "a", "it", "can", "handled", "used", "work",
    }

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name

    def _embed_one(self, text: str) -> List[float]:
        vec = [0.0] * self.DIMENSIONS
        cleaned = re.sub(r"^(passage|query):\s*", "", text.lower())
        expanded = re.sub(r"([a-z])([A-Z])", r"\1 \2", text).lower()
        tokens = re.findall(r"[a-z0-9_]{2,}", cleaned + " " + expanded)
        subtokens: List[str] = []
        for tok in tokens:
            if tok in self.STOPWORDS:
                continue
            subtokens.append(tok)
            if "_" in tok:
                subtokens.extend(p for p in tok.split("_") if len(p) >= 2 and p not in self.STOPWORDS)

        for idx, tok in enumerate(subtokens):
            weight = 2.5 if idx < 12 else 1.0
            h1 = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            vec[h1 % self.DIMENSIONS] += weight if (h1 >> 9) & 1 else -weight
            if len(tok) >= 4:
                for i in range(len(tok) - 3):
                    gram = tok[i : i + 4]
                    hg = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16)
                    vec[hg % self.DIMENSIONS] += 0.45 if (hg >> 7) & 1 else -0.45

        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def passage_embed(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_one(t) for t in texts]

    def query_embed(self, query: str) -> List[List[float]]:
        return [self._embed_one(query)]


_original_load_fastembed = cm_indexer._load_fastembed
_USING_FALLBACK_EMBEDDER = False


def _resilient_load_fastembed(model_name: str) -> Any:
    global _USING_FALLBACK_EMBEDDER
    try:
        return _original_load_fastembed(model_name)
    except TrackerError:
        _USING_FALLBACK_EMBEDDER = True
        return LocalCodeEmbedderFallback(model_name=model_name)


cm_indexer._load_fastembed = _resilient_load_fastembed
cm_ask._load_fastembed = _resilient_load_fastembed


# ============================================================================
# HELPER FUNCTIONS INSPECTING REAL `.index/` & GIT REPO
# ============================================================================

def print_header() -> None:
    console.print(
        Panel.fit(
            "[bold cyan]codemate[/bold cyan] [dim]v0.1.0[/dim] — "
            "Synced with Main Backend ([cyan]tracker[/cyan] • [cyan]chunker[/cyan] • "
            "[cyan]indexer[/cyan] • [cyan]ask[/cyan] • [cyan]evaluate[/cyan] • "
            "[cyan]failure_atlas[/cyan] • [cyan]git_guard[/cyan] • [cyan]agent[/cyan])",
            border_style="cyan",
        )
    )


def get_index_stats(root: Path) -> Dict[str, Any]:
    manifest_path = root / ".index" / "manifest.json"
    db_path = root / ".index" / "vectors.sqlite3"
    dotenv_path = root / ".env"

    stats: Dict[str, Any] = {
        "repo_root": str(root),
        "manifest_exists": manifest_path.is_file(),
        "manifest_files_count": 0,
        "manifest_generated_at": None,
        "manifest_files": [],
        "db_exists": db_path.is_file(),
        "db_size_bytes": db_path.stat().st_size if db_path.is_file() else 0,
        "chunks_count": 0,
        "indexed_files_count": 0,
        "skipped_files_count": 0,
        "model_name": DEFAULT_MODEL,
        "vector_dimensions": 384,
        "chunks_preview": [],
        "provider": DEFAULT_PROVIDER,
        "chat_model": DEFAULT_CHAT_MODEL,
        "api_key_configured": False,
        "config_path": str(config_path()),
        "active_failures_count": 0,
        "incidents": [],
    }

    if manifest_path.is_file():
        try:
            mdata = json.loads(manifest_path.read_text(encoding="utf-8"))
            files_dict = mdata.get("files", {})
            stats["manifest_files_count"] = len(files_dict)
            stats["manifest_generated_at"] = mdata.get("generated_at")
            for fpath, info in sorted(files_dict.items()):
                stats["manifest_files"].append({
                    "path": fpath,
                    "kind": info.get("kind"),
                    "sha256": info.get("sha256", "")[:12] + "...",
                    "size_bytes": info.get("size_bytes", 0),
                    "line_count": info.get("line_count"),
                    "binary": info.get("binary", False),
                })
        except Exception:
            pass

    if db_path.is_file():
        try:
            conn = sqlite3.connect(db_path)
            stats["chunks_count"] = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            stats["indexed_files_count"] = conn.execute(
                "SELECT COUNT(*) FROM indexed_files WHERE status = 'indexed'"
            ).fetchone()[0]
            stats["skipped_files_count"] = conn.execute(
                "SELECT COUNT(*) FROM indexed_files WHERE status = 'skipped'"
            ).fetchone()[0]
            row = conn.execute("SELECT value FROM metadata WHERE key = 'index_settings'").fetchone()
            if row:
                settings = json.loads(row[0])
                stats["model_name"] = settings.get("model", DEFAULT_MODEL)
            dim_row = conn.execute("SELECT value FROM metadata WHERE key = 'vector_dimensions'").fetchone()
            if dim_row:
                stats["vector_dimensions"] = int(dim_row[0])

            status_map = dict(conn.execute("SELECT path, status FROM indexed_files").fetchall())
            for mf in stats["manifest_files"]:
                mf["status"] = status_map.get(mf["path"], "tracked")

            cols = {r[1] for r in conn.execute("PRAGMA table_info(chunks)")}
            has_lang_sym = "language" in cols and "symbol" in cols
            if has_lang_sym:
                chunk_rows = conn.execute(
                    "SELECT path, ordinal, start_line, end_line, substr(text, 1, 240), "
                    "substr(content_hash, 1, 12), language, symbol "
                    "FROM chunks ORDER BY path, ordinal LIMIT 60"
                ).fetchall()
            else:
                chunk_rows = conn.execute(
                    "SELECT path, ordinal, start_line, end_line, substr(text, 1, 240), "
                    "substr(content_hash, 1, 12), '', '' "
                    "FROM chunks ORDER BY path, ordinal LIMIT 60"
                ).fetchall()
            for r in chunk_rows:
                stats["chunks_preview"].append({
                    "path": r[0],
                    "ordinal": r[1],
                    "start_line": r[2],
                    "end_line": r[3],
                    "preview": r[4],
                    "hash": r[5],
                    "language": r[6] or "",
                    "symbol": r[7] or "",
                })
            conn.close()
        except Exception:
            pass

    try:
        dotenv = _load_dotenv(dotenv_path)
    except TrackerError:
        dotenv = {}
    try:
        saved = load_config()
    except TrackerError:
        saved = {}

    stats["provider"] = (
        os.environ.get("CODEMATE_PROVIDER")
        or saved.get("provider")
        or dotenv.get("CODEMATE_PROVIDER")
        or DEFAULT_PROVIDER
    )
    stats["chat_model"] = (
        os.environ.get("CODEMATE_MODEL")
        or saved.get("model")
        or dotenv.get("CODEMATE_MODEL")
        or DEFAULT_CHAT_MODEL
    )
    stats["api_key_configured"] = bool(
        os.environ.get("GEMINI_API_KEY")
        or saved.get("api_key")
        or dotenv.get("GEMINI_API_KEY")
    )

    try:
        all_incidents = list_incidents(root, include_resolved=True)
        stats["active_failures_count"] = sum(1 for inc in all_incidents if inc.status == "active")
        stats["incidents"] = [
            {
                "id": inc.id[:8],
                "status": inc.status,
                "occurrence_count": inc.occurrence_count,
                "command": inc.command,
                "signature": inc.signature[:180],
                "last_seen": inc.last_seen,
                "last_commit": inc.last_commit,
                "report_path": inc.report_path,
            }
            for inc in all_incidents[:15]
        ]
    except Exception:
        pass

    return stats


def _git_blame_line(root: Path, rel_path: str, start_line: int, end_line: int) -> str:
    try:
        res = subprocess.run(
            ["git", "-C", str(root), "log", "-n", "1", "--pretty=format:%h by %an (%cr) — %s", "--", rel_path],
            capture_output=True,
            text=True,
            check=False,
        )
        out = res.stdout.strip()
        return out if out else "Working tree file (uncommitted or local)"
    except Exception:
        return "Git metadata unavailable"


# ============================================================================
# CLI COMMANDS WIRED TO REAL BACKEND
# ============================================================================

@app.command()
def track(
    repo: Path = typer.Option(Path.cwd(), "--repo", help="Repository or subdirectory"),
) -> None:
    """Run real `codemate.tracker.track_repository` and display `.index/manifest.json`."""
    print_header()
    result: TrackResult = track_repository(repo)

    table = Table(title=f"Tracked Snapshot ({result.manifest_path})", box=box.ROUNDED, border_style="cyan")
    table.add_column("Category", style="bold cyan")
    table.add_column("Count", justify="right", style="bold white")
    table.add_column("Paths", style="dim")

    table.add_row("Added", str(len(result.added)), ", ".join(result.added[:6]) or "—")
    table.add_row("Changed", str(len(result.changed)), ", ".join(result.changed[:6]) or "—")
    table.add_row("Unchanged", str(len(result.unchanged)), ", ".join(result.unchanged[:6]) or "—")
    table.add_row("Removed", str(len(result.removed)), ", ".join(result.removed[:6]) or "—")
    table.add_row("Skipped", str(len(result.skipped)), ", ".join(result.skipped[:6]) or "—")
    console.print(table)


@app.command()
def index(
    repo: Path = typer.Option(Path.cwd(), "--repo", help="Repository or subdirectory"),
    model: str = typer.Option(DEFAULT_MODEL, "--model", help="Local embedding model"),
    max_file_bytes: int = typer.Option(DEFAULT_MAX_FILE_BYTES, "--max-file-bytes", help="Skip larger files"),
) -> None:
    """Run real `codemate.indexer.index_repository` and inspect `.index/vectors.sqlite3`."""
    print_header()
    with Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30, complete_style="cyan", finished_style="green"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Running codemate.indexer.index_repository...", total=1)
        result: IndexResult = index_repository(repo, model_name=model, max_file_bytes=max_file_bytes)
        progress.advance(task)

    summary = Table(title="Real .index/vectors.sqlite3 Build Summary", box=box.ROUNDED, border_style="green")
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", style="white")
    summary.add_column("Backend Source", style="dim")

    embed_note = "fastembed ONNX" if not _USING_FALLBACK_EMBEDDER else "Local 384-d Fallback (Python 3.14)"
    summary.add_row("Database Path", str(result.database_path), "SQLite WAL (chunks + indexed_files + language/symbol)")
    summary.add_row("Embedding Model", result.model_name, embed_note)
    summary.add_row("Total Stored Chunks", str(result.chunks), "codemate.chunker.chunk_text (v2 boundary-aware)")
    summary.add_row(
        "File Delta",
        f"{result.added_files} added, {result.updated_files} updated, "
        f"{result.unchanged_files} unchanged, {result.removed_files} removed",
        f"{result.skipped_files} skipped (sensitive/binary/size)",
    )
    console.print(summary)


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question about the indexed codebase"),
    repo: Path = typer.Option(Path.cwd(), "--repo", help="Repository path"),
    top_k: int = typer.Option(5, "--top-k", help="Number of retrieved chunks"),
    verbose: bool = typer.Option(True, "--verbose/--quiet", "-v", help="Show retrieval & tool trace"),
) -> None:
    """Run real `codemate.ask.retrieve` (and `ask_repository` when API key is configured)."""
    print_header()
    root = find_repo_root(repo)

    hits, embedding_model = retrieve(question, root, top_k=top_k)

    if verbose:
        tree = Tree("[bold cyan]Real Backend Execution Trace[/bold cyan] [dim](.index/vectors.sqlite3)[/dim]")
        tree.add(
            f"[bold white]Step 1: codemate.ask.retrieve[/bold white] [green]HYBRID (0.78 Vector + 0.22 Lexical)[/green] "
            f"model=[magenta]{escape(embedding_model)}[/magenta] top_k={top_k} → "
            f"[cyan]{len(hits)} chunks ranked[/cyan]"
        )
        for h in hits[:3]:
            blame = _git_blame_line(root, h.path, h.start_line, h.end_line)
            meta = " · ".join(p for p in (h.language, h.symbol) if p)
            meta_str = f" [{meta}]" if meta else ""
            tree.add(
                f"[bold white]Step {int(h.label[1:]) + 1}: Citation \\[{h.label}][/bold white] "
                f"[cyan]{escape(h.path)}:{h.start_line}-{h.end_line}[/cyan]{escape(meta_str)} "
                f"(hybrid score [green]{h.score:.3f}[/green]) • [dim]{escape(blame)}[/dim]"
            )
        console.print(tree)
        console.print()

    try:
        answer, _, _ = ask_repository(question, root, top_k=top_k)
        mode_subtitle = f"Live LLM Answer • Model Embedded with {embedding_model}"
    except TrackerError as exc:
        top_hit = hits[0]
        second_hit = hits[1] if len(hits) > 1 else hits[0]
        answer = (
            f"[bold]Local Grounded Retrieval Summary[/bold] [dim](Note: {escape(str(exc))})[/dim]\n\n"
            f"Based on hybrid vector + lexical search across [bold].index/vectors.sqlite3[/bold]:\n"
            f"• Primary match is in [bold cyan]\\[{top_hit.label}] {escape(top_hit.path)}:{top_hit.start_line}-{top_hit.end_line}[/bold cyan] "
            f"({escape(top_hit.language or 'text')}{', ' + escape(top_hit.symbol) if top_hit.symbol else ''}; score [green]{top_hit.score:.3f}[/green]).\n"
            f"• Secondary context is in [bold cyan]\\[{second_hit.label}] {escape(second_hit.path)}:{second_hit.start_line}-{second_hit.end_line}[/bold cyan] "
            f"(score [green]{second_hit.score:.3f}[/green]).\n"
            f"• Configure an API key via [bold yellow]codemate init[/bold yellow] or [bold yellow].env[/bold yellow] to enable live Gemma 4 chat synthesis."
        )
        mode_subtitle = f"100% Local Hybrid Retrieval ({embedding_model}) • Run `codemate init` for LLM generation"

    console.print(
        Panel(
            answer,
            title="[bold green]Grounded Answer[/bold green]",
            subtitle=f"[dim]{escape(mode_subtitle)}[/dim]",
            border_style="green",
        )
    )

    for hit in hits[:3]:
        ext = Path(hit.path).suffix.lstrip(".") or "text"
        lang = {"py": "python", "md": "markdown", "toml": "toml", "html": "html", "js": "javascript"}.get(ext, "text")
        snippet_lines = hit.text.splitlines()[:18]
        snippet = "\n".join(snippet_lines)
        if len(hit.text.splitlines()) > 18:
            snippet += "\n# ... (truncated for display)"
        syntax = Syntax(snippet, lang, line_numbers=True, start_line=hit.start_line, theme="monokai")
        blame = _git_blame_line(root, hit.path, hit.start_line, hit.end_line)
        meta_badge = " · ".join(p for p in (hit.language, hit.symbol) if p)
        console.print(
            Panel(
                syntax,
                title=(
                    f"[bold cyan]\\[{hit.label}] {escape(hit.path)}:{hit.start_line}-{hit.end_line}[/bold cyan] "
                    + (f"• [magenta]{escape(meta_badge)}[/magenta] " if meta_badge else "")
                    + f"• [green]score {hit.score:.3f}[/green]"
                ),
                subtitle=f"[dim]git: {escape(blame)}[/dim]",
                border_style="bright_black",
            )
        )


@app.command()
def doctor(
    repo: Path = typer.Option(Path.cwd(), "--repo", help="Repository path"),
) -> None:
    """Run real `codemate.diagnostics.run_diagnostics` and inspect index health."""
    print_header()
    results = run_diagnostics(repo)
    table = Table(title="Codemate Doctor — Live Diagnostics", box=box.ROUNDED, border_style="cyan")
    table.add_column("Status", width=12, justify="center")
    table.add_column("Check Detail", style="white")
    for ok, msg in results:
        table.add_row("[bold green]✓ OK[/]" if ok else "[bold yellow]⚠ CHECK[/]", msg)
    console.print(table)


@app.command()
def evaluate(
    repo: Path = typer.Option(Path.cwd(), "--repo", help="Repository path"),
    top_k: int = typer.Option(6, "--top-k", help="Top-K chunks per benchmark question"),
) -> None:
    """Run real `codemate.evaluation.evaluate_retrieval` against `.codemate-eval.json`."""
    print_header()
    results, total = evaluate_retrieval(repo, None, top_k=top_k)
    table = Table(title=f"Retrieval Evaluation Benchmark (Top-{top_k})", box=box.ROUNDED, border_style="bright_blue")
    table.add_column("#", width=4, justify="right")
    table.add_column("Status", width=12, justify="center")
    table.add_column("Question", style="white")
    table.add_column("Expected File(s)", style="cyan")
    hits = 0
    rr_sum = 0.0
    for idx, r in enumerate(results, start=1):
        rank = r["rank"]
        if rank is not None:
            hits += 1
            rr_sum += 1.0 / rank
            badge = f"[bold green]✓ HIT @{rank}[/]"
        else:
            badge = "[bold red]✗ MISS[/]"
        table.add_row(str(idx), badge, r["question"], ", ".join(r["expected"]))
    console.print(table)
    console.print(
        Panel(
            f"[bold green]Recall@{top_k}:[/] {hits}/{total} ({hits / total:.0%})   |   "
            f"[bold cyan]MRR@{top_k}:[/] {rr_sum / total:.3f}",
            border_style="green" if hits == total else "yellow",
        )
    )


@app.command()
def failures(
    repo: Path = typer.Option(Path.cwd(), "--repo", help="Repository path"),
    all_incidents: bool = typer.Option(True, "--all", help="Include resolved incidents"),
) -> None:
    """Show incidents recorded in `.index/failures.sqlite3` by Failure Atlas."""
    print_header()
    incidents = list_incidents(repo, include_resolved=all_incidents)
    if not incidents:
        console.print(Panel("[bold green]✓ No incidents recorded in Failure Atlas yet.[/]", border_style="green"))
        return
    table = Table(title="Failure Atlas — Local Command Incident History", box=box.ROUNDED, border_style="red")
    table.add_column("ID", style="bold yellow")
    table.add_column("Status")
    table.add_column("Seen", justify="right")
    table.add_column("Command", style="cyan")
    table.add_column("Signature", style="white")
    for inc in incidents:
        table.add_row(
            inc.id[:8],
            inc.status.upper(),
            str(inc.occurrence_count),
            inc.command,
            inc.signature.replace("\n", " | ")[:100],
        )
    console.print(table)


@app.command()
def explain(
    target: str = typer.Argument("codemate/indexer.py", help="Repository file to explain"),
    repo: Path = typer.Option(Path.cwd(), "--repo", help="Repository path"),
) -> None:
    """Retrieve and explain a real file from `.index/vectors.sqlite3`."""
    ask(question=f"Explain how {target} works and what its key functions do", repo=repo, top_k=3, verbose=True)


# ============================================================================
# LOCAL HTTP API SERVER CONNECTING `codemate-ui.html` TO REAL BACKEND
# ============================================================================

class CodemateAPIHandler(BaseHTTPRequestHandler):
    repo_root: Path = Path.cwd()

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self) -> None:
        self._send_json({"ok": True})

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in ("/", "/codemate-ui.html"):
            html_file = self.repo_root / "codemate-ui.html"
            if html_file.is_file():
                content = html_file.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return
        if parsed.path == "/api/status":
            try:
                root = find_repo_root(self.repo_root)
                self._send_json({"ok": True, "stats": get_index_stats(root)})
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=400)
            return
        self._send_json({"ok": False, "error": "Not found"}, status=404)

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        body_raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        try:
            body = json.loads(body_raw) if body_raw else {}
        except json.JSONDecodeError:
            body = {}

        try:
            root = find_repo_root(self.repo_root)
            if parsed.path == "/api/track":
                res = track_repository(root)
                self._send_json({
                    "ok": True,
                    "added": res.added,
                    "changed": res.changed,
                    "unchanged": res.unchanged,
                    "removed": res.removed,
                    "skipped": res.skipped,
                    "manifest_path": str(res.manifest_path),
                    "stats": get_index_stats(root),
                })
                return

            if parsed.path == "/api/index":
                res = index_repository(root)
                self._send_json({
                    "ok": True,
                    "added_files": res.added_files,
                    "updated_files": res.updated_files,
                    "unchanged_files": res.unchanged_files,
                    "removed_files": res.removed_files,
                    "skipped_files": res.skipped_files,
                    "chunks": res.chunks,
                    "database_path": str(res.database_path),
                    "model_name": res.model_name,
                    "stats": get_index_stats(root),
                })
                return

            if parsed.path == "/api/evaluate":
                top_k = int(body.get("top_k", 6))
                results, total = evaluate_retrieval(root, None, top_k=top_k)
                hits_count = sum(1 for r in results if r["rank"] is not None)
                mrr = sum((1.0 / r["rank"] if r["rank"] else 0.0) for r in results) / max(1, total)
                self._send_json({
                    "ok": True,
                    "results": results,
                    "total": total,
                    "hits": hits_count,
                    "recall": round(hits_count / max(1, total), 4),
                    "mrr": round(mrr, 4),
                    "stats": get_index_stats(root),
                })
                return

            if parsed.path == "/api/doctor":
                checks = run_diagnostics(root)
                self._send_json({
                    "ok": True,
                    "checks": [{"ok": ok, "message": msg} for ok, msg in checks],
                    "stats": get_index_stats(root),
                })
                return

            if parsed.path == "/api/ask":
                question = body.get("question", "How does Codemate index and retrieve files?")
                top_k = int(body.get("top_k", 5))
                if not (root / ".index" / "vectors.sqlite3").is_file():
                    index_repository(root)
                hits, embed_model = retrieve(question, root, top_k=top_k)
                llm_used = False
                try:
                    answer, _, _ = ask_repository(question, root, top_k=top_k)
                    llm_used = True
                except TrackerError as exc:
                    top = hits[0]
                    second = hits[1] if len(hits) > 1 else hits[0]
                    answer = (
                        f"Grounded Hybrid Retrieval across .index/vectors.sqlite3 (LLM note: {exc})\n\n"
                        f"• Top Match [{top.label}]: {top.path}:{top.start_line}-{top.end_line} "
                        f"({top.language or 'text'}{', ' + top.symbol if top.symbol else ''}; hybrid score {top.score:.3f})\n"
                        f"• Second Match [{second.label}]: {second.path}:{second.start_line}-{second.end_line} "
                        f"(hybrid score {second.score:.3f})"
                    )
                serialized_hits = []
                for h in hits:
                    serialized_hits.append({
                        "label": h.label,
                        "path": h.path,
                        "start_line": h.start_line,
                        "end_line": h.end_line,
                        "text": h.text,
                        "score": round(h.score, 4),
                        "language": h.language,
                        "symbol": h.symbol,
                        "blame": _git_blame_line(root, h.path, h.start_line, h.end_line),
                    })
                self._send_json({
                    "ok": True,
                    "question": question,
                    "answer": answer,
                    "llm_used": llm_used,
                    "embedding_model": embed_model,
                    "hits": serialized_hits,
                    "stats": get_index_stats(root),
                })
                return

            if parsed.path == "/api/run-cli":
                import shlex as _shlex
                import sys as _sys

                cmd_str = body.get("cmd", "codemate track").strip()
                try:
                    tokens = _shlex.split(cmd_str)
                except ValueError as shlex_err:
                    fixed_cmd = cmd_str + '"' if cmd_str.count('"') % 2 == 1 else cmd_str + "'"
                    err_box = (
                        f"╭─ CLI Shell Syntax Error ──────────────────────────────────────────────╮\n"
                        f"│ ✗ {shlex_err} (unclosed quote in command string)\n"
                        f"│\n"
                        f"│ 💡 Did you mean:\n"
                        f"│   $ {fixed_cmd}\n"
                        f"╰─────────────────────────────────────────────────────── Exit Code: 2 ──╯\n"
                    )
                    self._send_json({
                        "ok": False,
                        "cmd": cmd_str,
                        "returncode": 2,
                        "stdout": "",
                        "stderr": err_box,
                        "suggestion": fixed_cmd,
                        "stats": get_index_stats(root),
                    })
                    return
                if tokens and tokens[0] == "codemate":
                    tokens = tokens[1:]
                if not tokens:
                    tokens = ["track"]
                env = os.environ.copy()
                env["PYTHONIOENCODING"] = "utf-8"
                env["FORCE_COLOR"] = "1"
                proc = subprocess.run(
                    [_sys.executable, "-m", "codemate.cli", *tokens],
                    cwd=str(root),
                    env=env,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                self._send_json({
                    "ok": proc.returncode == 0,
                    "cmd": "codemate " + " ".join(tokens),
                    "returncode": proc.returncode,
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                    "stats": get_index_stats(root),
                })
                return

            self._send_json({"ok": False, "error": f"Unknown endpoint {parsed.path}"}, status=404)
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=500)


@app.command()
def serve(
    port: int = typer.Option(8765, "--port", help="Port for local Web UI + Backend API"),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open browser automatically"),
) -> None:
    """Start local Web UI server at http://127.0.0.1:8765 connected to the real Codemate backend."""
    print_header()
    root = find_repo_root(Path.cwd())
    if not (root / ".index" / "vectors.sqlite3").is_file():
        console.print("[cyan]Building initial .index/vectors.sqlite3 from repository files...[/cyan]")
        index_repository(root)

    CodemateAPIHandler.repo_root = root
    url = f"http://127.0.0.1:{port}"
    console.print(
        Panel(
            f"[bold green]✓ Codemate Live Backend Server Running[/bold green]\n"
            f"[white]Web UI URL:[/white] [bold cyan]{url}[/bold cyan]\n"
            f"[white]Connected Modules:[/white] [cyan]tracker[/cyan], [cyan]chunker[/cyan], "
            f"[cyan]indexer[/cyan], [cyan]ask[/cyan], [cyan]diagnostics[/cyan], [cyan]evaluation[/cyan], "
            f"[cyan]failure_atlas[/cyan], [cyan]git_guard[/cyan], [cyan]agent[/cyan]\n"
            f"[dim]Press Ctrl+C to stop the server.[/dim]",
            border_style="green",
        )
    )
    if open_browser:
        webbrowser.open(url)
    server = HTTPServer(("127.0.0.1", port), CodemateAPIHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped.[/yellow]")


if __name__ == "__main__":
    app()
