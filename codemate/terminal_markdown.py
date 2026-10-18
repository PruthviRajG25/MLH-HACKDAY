"""Small, dependency-free Markdown formatter for terminal answers."""

from __future__ import annotations

import os
import re
import sys


_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def render_markdown(text: str, *, color: bool | None = None) -> str:
    """Render common Markdown blocks without exposing control sequences from model output."""
    if color is None:
        color = (
            sys.stdout.isatty()
            and "NO_COLOR" not in os.environ
            and os.environ.get("TERM", "") != "dumb"
        )
    text = _CONTROL.sub("", _ANSI.sub("", text)).replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    rendered: list[str] = []
    in_code = False
    table_rows: list[list[str]] = []

    def style(value: str, code: str) -> str:
        return f"\x1b[{code}m{value}\x1b[0m" if color and value else value

    def inline(value: str) -> str:
        value = _LINK.sub(lambda m: f"{m.group(1)} ({m.group(2)})", value)
        value = re.sub(r"`([^`]+)`", lambda m: style(m.group(1), "36"), value)
        value = re.sub(r"\*\*(.+?)\*\*|__(.+?)__", lambda m: style(m.group(1) or m.group(2), "1"), value)
        value = re.sub(r"(?<!\w)\*([^*\n]+)\*(?!\w)|(?<!\w)_([^_\n]+)_(?!\w)",
                       lambda m: style(m.group(1) or m.group(2), "3"), value)
        value = re.sub(r"~~(.+?)~~", lambda m: style(m.group(1), "9"), value)
        return value

    def flush_table() -> None:
        if not table_rows:
            return
        rows = [row for row in table_rows if not all(re.fullmatch(r"\s*:?-{3,}:?\s*", c) for c in row)]
        if not rows:
            table_rows.clear()
            return
        width_count = max(len(row) for row in rows)
        widths = [max((len(row[i]) if i < len(row) else 0) for row in rows) for i in range(width_count)]
        for row_index, row in enumerate(rows):
            cells = [row[i].strip() if i < len(row) else "" for i in range(width_count)]
            line = "  " + "  │  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells)).rstrip()
            rendered.append(style(line, "1") if row_index == 0 else line)
        table_rows.clear()

    for raw in lines:
        fence = re.match(r"^\s*(```+|~~~+)(.*)$", raw)
        if fence:
            flush_table()
            if not in_code:
                in_code = True
                code_language = fence.group(2).strip()
                if code_language:
                    rendered.append(style(f"  {code_language}", "2"))
            else:
                in_code = False
            continue
        if in_code:
            rendered.append(style(f"  {raw}" if raw else "", "36"))
            continue

        if raw.strip().startswith("|") and raw.strip().endswith("|"):
            cells = [cell.strip() for cell in raw.strip().strip("|").split("|")]
            table_rows.append(cells)
            continue
        flush_table()

        heading = re.match(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$", raw)
        if heading:
            rendered.append("")
            rendered.append(style(inline(heading.group(2)), "1;36"))
            rendered.append("")
            continue
        if re.fullmatch(r"\s*(?:-{3,}|\*{3,}|_{3,})\s*", raw):
            rendered.append(style("─" * 40, "2"))
            continue
        quote = re.match(r"^\s*>\s?(.*)$", raw)
        if quote:
            rendered.append(style("│ ", "36") + inline(quote.group(1)))
            continue
        bullet = re.match(r"^(\s*)[-*+]\s+(.*)$", raw)
        if bullet:
            indent = " " * min(len(bullet.group(1)), 8)
            rendered.append(f"{indent}• {inline(bullet.group(2))}")
            continue
        numbered = re.match(r"^(\s*)(\d{1,3})[.)]\s+(.*)$", raw)
        if numbered:
            indent = " " * min(len(numbered.group(1)), 8)
            rendered.append(f"{indent}{numbered.group(2)}. {inline(numbered.group(3))}")
            continue
        rendered.append(inline(raw))

    flush_table()
    while rendered and not rendered[-1]:
        rendered.pop()
    return "\n".join(rendered)
