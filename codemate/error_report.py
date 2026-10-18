"""Create a local Markdown report for a diagnosed command failure."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
import shlex

from .ask import SearchHit
from .tracker import TrackerError, find_repo_root


def _fence(content: str, language: str = "") -> str:
    longest = max((len(match.group(0)) for match in re.finditer(r"`+", content)), default=0)
    marker = "`" * max(3, longest + 1)
    return f"{marker}{language}\n{content.rstrip()}\n{marker}"


def create_error_report(
    repo: Path,
    command: list[str],
    exit_code: int,
    command_output: str,
    diagnosis: str,
    hits: list[SearchHit],
    embedding_model: str,
) -> Path:
    root = find_repo_root(repo.resolve())
    report_dir = root / "report"
    try:
        report_dir.mkdir(parents=True, exist_ok=True)
        created_at = datetime.now().astimezone()
        report_path = report_dir / f"diagnosis-{created_at.strftime('%Y%m%d-%H%M%S-%f')}.md"
        sources = "\n\n".join(
            f"### [{hit.label}] {hit.path}:{hit.start_line}-{hit.end_line}\n"
            f"Score: {hit.score:.3f}"
            f"{f' · {hit.language}' if hit.language else ''}"
            f"{f' · {hit.symbol}' if hit.symbol else ''}\n\n"
            f"{_fence(hit.text, 'text')}"
            for hit in hits
        )
        content = "\n".join(
            [
                "# Codemate error report",
                "",
                f"- Created: {created_at.isoformat(timespec='seconds')}",
                f"- Repository: `{root}`",
                f"- Exit code: `{exit_code}`",
                f"- Embedding model: `{embedding_model}`",
                "",
                "## Command",
                "",
                _fence(shlex.join(command), "sh"),
                "",
                "## Original command output",
                "",
                _fence(command_output, "text"),
                "",
                "## Codemate diagnosis",
                "",
                diagnosis.strip(),
                "",
                "## Retrieved source excerpts",
                "",
                sources or "No source excerpts were retrieved.",
                "",
            ]
        )
        with report_path.open("x", encoding="utf-8") as stream:
            stream.write(content)
    except OSError as exc:
        raise TrackerError(f"Could not write error report: {exc}") from exc
    return report_path
