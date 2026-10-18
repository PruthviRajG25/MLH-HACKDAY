"""Small retrieval-quality evaluation against repository-specific examples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .ask import retrieve
from .tracker import TrackerError, find_repo_root


def evaluate_retrieval(repo: Path, cases_file: Path | None = None, top_k: int = 6) -> tuple[list[dict[str, Any]], int]:
    root = find_repo_root(repo.resolve())
    source = cases_file or root / ".codemate-eval.json"
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrackerError(f"Could not read evaluation cases at {source}: {exc}") from exc
    cases = data.get("cases") if isinstance(data, dict) else None
    if not isinstance(cases, list) or not cases:
        raise TrackerError("Evaluation file must contain a non-empty 'cases' list.")

    results: list[dict[str, Any]] = []
    for number, case in enumerate(cases, start=1):
        if not isinstance(case, dict) or not isinstance(case.get("question"), str):
            raise TrackerError(f"Evaluation case {number} needs a string 'question'.")
        expected = case.get("expected_paths")
        if not isinstance(expected, list) or not expected or not all(isinstance(p, str) for p in expected):
            raise TrackerError(f"Evaluation case {number} needs a non-empty 'expected_paths' list.")
        try:
            hits, _ = retrieve(case["question"], root, top_k=top_k)
        except TrackerError as exc:
            results.append({"question": case["question"], "expected": expected, "paths": [], "rank": None, "error": str(exc)})
            continue
        paths = [hit.path for hit in hits]
        rank = next((i for i, path in enumerate(paths, start=1) if path in expected), None)
        results.append({"question": case["question"], "expected": expected, "paths": paths, "rank": rank})
    return results, len(cases)
