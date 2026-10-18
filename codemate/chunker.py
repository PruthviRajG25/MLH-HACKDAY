"""Small, line-aware text chunking for repository files."""

from __future__ import annotations

import re
from dataclasses import dataclass


_DECLARATION_LINE = re.compile(
    r"(?m)^\s*(?:(?:async\s+)?(?:def|function|fn|func)\b|class\b|"
    r"(?:export\s+)(?:async\s+)?(?:function|class)\b)"
)


@dataclass(frozen=True)
class CodeChunk:
    ordinal: int
    start_line: int
    end_line: int
    text: str


def chunk_text(text: str, max_chars: int = 1800, overlap_chars: int = 240) -> list[CodeChunk]:
    """Split text into bounded overlapping pieces with source line locations.

    Boundaries prefer newlines, while long single lines are split at the size
    limit. Character offsets are used for overlap so chunking remains bounded
    even for minified files.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be greater than zero")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be between zero and max_chars - 1")
    if not text:
        return []

    chunks: list[CodeChunk] = []
    start = 0
    ordinal = 0
    length = len(text)
    while start < length:
        hard_end = min(start + max_chars, length)
        end = hard_end
        if hard_end < length:
            declaration = None
            for match in _DECLARATION_LINE.finditer(text, start + max_chars // 2, hard_end):
                declaration = match.start()
            if declaration is not None and declaration > start:
                end = declaration
            newline = text.rfind("\n", start, hard_end)
            if end == hard_end and newline >= start + max_chars // 2:
                end = newline + 1

        piece = text[start:end]
        start_line = text.count("\n", 0, start) + 1
        end_line = text.count("\n", 0, end)
        if end > 0 and text[end - 1] != "\n":
            end_line += 1
        chunks.append(CodeChunk(ordinal, start_line, max(start_line, end_line), piece))
        ordinal += 1
        if end >= length:
            break

        next_start = max(start + 1, end - overlap_chars)
        # Prefer to begin the next chunk at a line boundary when close to the
        # requested overlap point. Long lines remain safely split by character.
        boundary = text.find("\n", next_start, min(end, next_start + 120))
        if boundary >= 0:
            next_start = boundary + 1
        start = next_start
    return chunks
