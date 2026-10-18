"""Chunk, embed, and persist the current text files in a repository."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .chunker import chunk_text
from .tracker import TrackerError, track_repository


DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_MAX_FILE_BYTES = 1_000_000
DEFAULT_MAX_CHARS = 1200
DEFAULT_OVERLAP_CHARS = 160
EMBED_BATCH_SIZE = 32


def _is_sensitive_path(relative: str) -> bool:
    parts = Path(relative).parts
    for part in parts:
        name = part.lower()
        if name == ".env" or name.startswith(".env."):
            return True
    filename = parts[-1].lower() if parts else ""
    return filename.endswith((".pem", ".key", ".p12", ".pfx")) or filename in {
        "id_rsa",
        "id_ed25519",
    }


@dataclass
class IndexResult:
    added_files: int
    updated_files: int
    unchanged_files: int
    removed_files: int
    skipped_files: int
    chunks: int
    database_path: Path
    model_name: str


def _connect(database_path: Path) -> sqlite3.Connection:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS indexed_files (
            path TEXT PRIMARY KEY,
            sha256 TEXT NOT NULL,
            status TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            path TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            start_line INTEGER NOT NULL,
            end_line INTEGER NOT NULL,
            text TEXT NOT NULL,
            embedding BLOB NOT NULL,
            dimensions INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS chunks_path_idx ON chunks(path);
        """
    )
    return connection


def _pack_vector(vector: Any) -> tuple[bytes, int]:
    values = vector.tolist() if hasattr(vector, "tolist") else list(vector)
    return struct.pack(f"<{len(values)}f", *values), len(values)


def _load_fastembed(model_name: str) -> Any:
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:
        raise TrackerError(
            "Local embedding support is not installed. Install it with "
            "`python -m pip install 'codemate[embeddings]'`."
        ) from exc
    try:
        return TextEmbedding(model_name=model_name)
    except Exception as exc:
        raise TrackerError(f"Could not load embedding model {model_name!r}: {exc}") from exc


def index_repository(
    path: Path,
    model_name: str = DEFAULT_MODEL,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> IndexResult:
    """Update the local chunk/vector index; only changed files are embedded."""
    if max_file_bytes < 1:
        raise TrackerError("Maximum indexed file size must be greater than zero.")
    try:
        chunks_config = {
            "max_chars": max_chars,
            "overlap_chars": overlap_chars,
            "max_file_bytes": max_file_bytes,
            "privacy_filter_version": 1,
        }
        # Validate chunk settings before changing the tracker or index.
        chunk_text("validation", max_chars, overlap_chars)
    except ValueError as exc:
        raise TrackerError(str(exc)) from exc

    tracking = track_repository(path)
    root = tracking.manifest_path.parent.parent
    try:
        manifest = json.loads(tracking.manifest_path.read_text(encoding="utf-8"))
        inventory: dict[str, dict[str, Any]] = manifest["files"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise TrackerError(f"Could not read tracker manifest: {exc}") from exc

    database_path = root / ".index" / "vectors.sqlite3"
    connection = _connect(database_path)
    settings = json.dumps(
        {"model": model_name, "chunks": chunks_config}, sort_keys=True, separators=(",", ":")
    )
    prior_settings = connection.execute(
        "SELECT value FROM metadata WHERE key = 'index_settings'"
    ).fetchone()
    reset = prior_settings is not None and prior_settings[0] != settings
    if reset:
        connection.execute("DELETE FROM chunks")
        connection.execute("DELETE FROM indexed_files")
        connection.commit()

    indexed = {
        row[0]: (row[1], row[2])
        for row in connection.execute("SELECT path, sha256, status FROM indexed_files")
    }
    inventory_paths = set(inventory)
    removed = set(indexed) - inventory_paths
    if removed:
        connection.executemany("DELETE FROM chunks WHERE path = ?", ((p,) for p in removed))
        connection.executemany("DELETE FROM indexed_files WHERE path = ?", ((p,) for p in removed))
        connection.commit()

    changed: list[tuple[str, dict[str, Any]]] = []
    unchanged_count = 0
    skipped_count = 0
    added_count = 0
    updated_count = 0
    for relative, record in sorted(inventory.items()):
        old = indexed.get(relative)
        if old is not None and old[0] == record.get("sha256"):
            unchanged_count += 1
            if old[1] == "skipped":
                skipped_count += 1
            continue
        if old is None:
            added_count += 1
        else:
            updated_count += 1
        changed.append((relative, record))

    embedder = None
    for relative, record in changed:
        file_path = root / relative
        status = "indexed"
        file_chunks = []
        try:
            if _is_sensitive_path(relative):
                status = "skipped"
            elif record.get("kind") != "file" or record.get("binary"):
                status = "skipped"
            elif record.get("size_bytes", 0) > max_file_bytes:
                status = "skipped"
            else:
                if file_path.is_symlink():
                    status = "skipped"
                    content = b""
                else:
                    try:
                        resolved_path = file_path.resolve(strict=True)
                        resolved_path.relative_to(root)
                        content = resolved_path.read_bytes()
                    except (OSError, ValueError) as exc:
                        raise TrackerError(f"Unsafe or unreadable repository path {relative}: {exc}") from exc
                if status == "skipped":
                    content = b""
                elif hashlib.sha256(content).hexdigest() != record.get("sha256"):
                    raise TrackerError(
                        f"{relative} changed while indexing; run `codemate index` again."
                    )
                if status != "skipped":
                    try:
                        text = content.decode("utf-8")
                    except UnicodeDecodeError:
                        status = "skipped"
                    else:
                        file_chunks = chunk_text(text, max_chars, overlap_chars)
        except OSError as exc:
            raise TrackerError(f"Could not read {relative}: {exc}") from exc

        vectors: list[Any] = []
        if file_chunks:
            if embedder is None:
                embedder = _load_fastembed(model_name)
            documents = [f"File: {relative}\nLines: {c.start_line}-{c.end_line}\n\n{c.text}" for c in file_chunks]
            try:
                for offset in range(0, len(documents), EMBED_BATCH_SIZE):
                    vectors.extend(
                        embedder.passage_embed(documents[offset : offset + EMBED_BATCH_SIZE])
                    )
            except Exception as exc:
                raise TrackerError(f"Embedding failed for {relative}: {exc}") from exc
            if len(vectors) != len(file_chunks):
                raise TrackerError(f"Embedding model returned an unexpected result for {relative}.")

        try:
            with connection:
                connection.execute("DELETE FROM chunks WHERE path = ?", (relative,))
                for chunk, vector in zip(file_chunks, vectors):
                    vector_blob, dimensions = _pack_vector(vector)
                    chunk_id = hashlib.sha256(
                        f"{relative}\0{record['sha256']}\0{chunk.ordinal}".encode(
                            "utf-8", errors="surrogateescape"
                        )
                    ).hexdigest()
                    connection.execute(
                        """INSERT INTO chunks
                           (id, path, content_hash, ordinal, start_line, end_line, text, embedding, dimensions)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            chunk_id,
                            relative,
                            record["sha256"],
                            chunk.ordinal,
                            chunk.start_line,
                            chunk.end_line,
                            chunk.text,
                            vector_blob,
                            dimensions,
                        ),
                    )
                connection.execute(
                    """INSERT INTO indexed_files(path, sha256, status) VALUES (?, ?, ?)
                       ON CONFLICT(path) DO UPDATE SET sha256=excluded.sha256, status=excluded.status""",
                    (relative, record["sha256"], status),
                )
                connection.execute(
                    "INSERT INTO metadata(key, value) VALUES ('index_settings', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (settings,),
                )
                if vectors:
                    connection.execute(
                        "INSERT INTO metadata(key, value) VALUES ('vector_dimensions', ?) "
                        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                        (str(dimensions),),
                    )
        except sqlite3.Error as exc:
            raise TrackerError(f"Could not save index data: {exc}") from exc
        if status == "skipped":
            skipped_count += 1

    if prior_settings is None or reset:
        with connection:
            connection.execute(
                "INSERT INTO metadata(key, value) VALUES ('index_settings', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (settings,),
            )
    stored_chunks = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    connection.close()
    return IndexResult(
        added_files=added_count,
        updated_files=updated_count,
        unchanged_files=unchanged_count,
        removed_files=len(removed),
        skipped_files=skipped_count,
        chunks=stored_chunks,
        database_path=database_path,
        model_name=model_name,
    )
