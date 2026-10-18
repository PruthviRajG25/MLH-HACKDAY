"""Command-line interface for Codemate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .ask import ask_repository
from .indexer import DEFAULT_MAX_FILE_BYTES, DEFAULT_MODEL, IndexResult, index_repository
from .tracker import TrackerError, track_repository


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="codemate", description="Track a Git codebase locally.")
    subparsers = parser.add_subparsers(dest="command", required=True)
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
    args = _parser().parse_args(argv)
    if args.command == "track":
        try:
            result = track_repository(args.repo)
        except TrackerError as exc:
            print(f"codemate: error: {exc}", file=sys.stderr)
            return 2
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
            result: IndexResult = index_repository(args.repo, args.model, args.max_file_bytes)
        except TrackerError as exc:
            print(f"codemate: error: {exc}", file=sys.stderr)
            return 2
        print(f"Indexed with {result.model_name}: {result.chunks} chunks in {result.database_path}")
        print(
            f"{result.added_files} added, {result.updated_files} updated, "
            f"{result.unchanged_files} unchanged, {result.removed_files} removed, "
            f"{result.skipped_files} skipped"
        )
        return 0
    if args.command == "ask":
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
            print(f"codemate: error: {exc}", file=sys.stderr)
            return 2
        print(answer)
        print(f"\nSources (retrieved locally with {embedding_model}):")
        for hit in hits:
            print(f"[{hit.label}] {hit.path}:{hit.start_line}-{hit.end_line} (score {hit.score:.3f})")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
