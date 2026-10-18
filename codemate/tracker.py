"""Track the current contents of a Git working tree in a local manifest."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


MANIFEST_NAME = "manifest.json"


class TrackerError(Exception):
    """An expected error while discovering or tracking a repository."""


@dataclass
class TrackResult:
    added: list[str]
    changed: list[str]
    unchanged: list[str]
    removed: list[str]
    skipped: list[str]
    manifest_path: Path


def find_repo_root(path: Path) -> Path:
    """Return the Git worktree root containing *path*."""
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise TrackerError("Git is required but was not found on PATH.") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or "not inside a Git repository"
        raise TrackerError(f"Could not find a Git repository: {detail}") from exc
    return Path(result.stdout.strip()).resolve()


def _git_inventory(root: Path) -> list[str]:
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode(errors="replace").strip()
        raise TrackerError(f"Git could not list repository files: {detail}") from exc
    # Git uses NUL separators so paths containing spaces or newlines stay intact.
    return sorted({entry.decode("utf-8", errors="surrogateescape")
                   for entry in result.stdout.split(b"\0") if entry})


def _safe_path(root: Path, relative: str) -> Path | None:
    """Resolve a Git path without following its final symlink."""
    rel = PurePosixPath(relative)
    if rel.is_absolute() or ".." in rel.parts or ".index" in rel.parts:
        return None
    candidate = root.joinpath(*rel.parts)
    try:
        # Resolve parents to ensure a directory symlink cannot escape the repo.
        candidate.parent.resolve(strict=True).relative_to(root)
    except (OSError, ValueError):
        return None
    if candidate.is_symlink():
        return candidate
    try:
        candidate.resolve(strict=True).relative_to(root)
    except (OSError, ValueError):
        return None
    if not candidate.is_file():
        return None
    return candidate


def _record(path: Path) -> dict[str, Any] | None:
    if path.is_symlink():
        try:
            target = os.readlink(path).encode("utf-8", errors="surrogateescape")
        except OSError:
            return None
        return {
            "kind": "symlink",
            "sha256": hashlib.sha256(target).hexdigest(),
            "size_bytes": len(target),
            "line_count": None,
            "binary": False,
        }

    digest = hashlib.sha256()
    size = 0
    newlines = 0
    has_nul = False
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
                newlines += chunk.count(b"\n")
                has_nul = has_nul or b"\0" in chunk
    except OSError:
        return None
    return {
        "kind": "file",
        "sha256": digest.hexdigest(),
        "size_bytes": size,
        "line_count": None if has_nul else newlines + (1 if size and not _ends_in_newline(path) else 0),
        "binary": has_nul,
    }


def _ends_in_newline(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                return False
            stream.seek(-1, os.SEEK_END)
            return stream.read(1) == b"\n"
    except OSError:
        return False


def _read_manifest(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema_version") != 1 or not isinstance(data.get("files"), dict):
            raise ValueError("unsupported manifest format")
        return data["files"]
    except (OSError, json.JSONDecodeError, AttributeError, ValueError) as exc:
        raise TrackerError(f"Cannot read {path}: {exc}") from exc


def _write_manifest(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix="manifest-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def track_repository(path: Path) -> TrackResult:
    """Snapshot eligible current working-tree files and compare to the prior run."""
    root = find_repo_root(path.resolve())
    manifest_path = root / ".index" / MANIFEST_NAME
    previous = _read_manifest(manifest_path)
    current: dict[str, dict[str, Any]] = {}
    skipped: list[str] = []

    for relative in _git_inventory(root):
        file_path = _safe_path(root, relative)
        if file_path is None:
            skipped.append(relative)
            continue
        record = _record(file_path)
        if record is None:
            skipped.append(relative)
            continue
        current[relative] = record

    added = sorted(current.keys() - previous.keys())
    removed = sorted(previous.keys() - current.keys())
    unchanged = sorted(
        name for name in current.keys() & previous.keys()
        if current[name]["sha256"] == previous[name].get("sha256")
    )
    changed = sorted((current.keys() & previous.keys()) - set(unchanged))

    _write_manifest(
        manifest_path,
        {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "files": current,
        },
    )
    return TrackResult(added, changed, unchanged, removed, sorted(set(skipped)), manifest_path)
