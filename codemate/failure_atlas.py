"""Local history of command failures and their diagnoses."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import os
import re
import shlex
import sqlite3
import subprocess
from pathlib import Path

from .tracker import TrackerError, find_repo_root


_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_HEX_ADDRESS = re.compile(r"\b0x[0-9a-fA-F]+\b")
_UUID = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F-]{27,}\b")
_DURATION = re.compile(r"\b\d+(?:\.\d+)?\s*(?:ms|msec|seconds?|secs?)\b", re.I)
_LINE_NUMBER = re.compile(r"(?<=:)\d+(?=[:)])")
_HIGH_SIGNAL = re.compile(
    r"(?:^\s*(?:FAILED|ERROR)\s+|\b(?:AssertionError|ImportError|ModuleNotFoundError|"
    r"SyntaxError|TypeError|ValueError|RuntimeError|KeyError|IndexError|NameError|"
    r"AttributeError|Exception|Error|panic|fatal error):|^\s*--- FAIL:|^\s*FAIL\s+|"
    r"^\s*●\s+)"
)


@dataclass(frozen=True)
class Incident:
    id: str
    status: str
    occurrence_count: int
    command: str
    signature: str
    last_seen: str
    last_commit: str | None
    diagnosis: str | None
    report_path: str | None


def _command_key(command: list[str]) -> str:
    return shlex.join(command)


def _signature(output: str, repo_root: Path) -> str:
    cleaned = _ANSI.sub("", output).replace("\r", "")
    meaningful = [line.strip() for line in cleaned.splitlines() if line.strip()]
    selected = [line for line in meaningful if _HIGH_SIGNAL.search(line)]
    if not selected:
        selected = meaningful[-8:]
    normalized: list[str] = []
    for line in selected[-12:]:
        line = line.replace(str(repo_root), "<repo>")
        line = _HEX_ADDRESS.sub("<address>", line)
        line = _UUID.sub("<id>", line)
        line = _DURATION.sub("<duration>", line)
        line = _LINE_NUMBER.sub("<line>", line)
        normalized.append(" ".join(line.split()))
    return "\n".join(normalized)[:3000] or "Command failed without output"


def _commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _connect(root: Path) -> sqlite3.Connection:
    database = root / ".index" / "failures.sqlite3"
    try:
        database.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(database, timeout=10)
        os.chmod(database, 0o600)
        connection.row_factory = sqlite3.Row
        connection.execute(
            """CREATE TABLE IF NOT EXISTS incidents (
                id TEXT PRIMARY KEY,
                command_key TEXT NOT NULL,
                command TEXT NOT NULL,
                signature TEXT NOT NULL,
                exit_code INTEGER NOT NULL,
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                occurrence_count INTEGER NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('active', 'resolved')),
                last_commit TEXT,
                diagnosis TEXT,
                report_path TEXT
            )"""
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS incidents_last_seen ON incidents(last_seen DESC)"
        )
        connection.commit()
        return connection
    except (OSError, sqlite3.Error) as exc:
        raise TrackerError(f"Could not open local Failure Atlas: {exc}") from exc


@contextmanager
def _database(root: Path):
    connection = _connect(root)
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def record_failure(
    repo: Path, command: list[str], exit_code: int, output: str
) -> Incident:
    """Record a failure and return its new, recurring, or reopened incident."""
    root = find_repo_root(repo.resolve())
    key = _command_key(command)
    signature = _signature(output, root)
    incident_id = hashlib.sha256(f"{key}\0{signature}".encode()).hexdigest()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    commit = _commit(root)
    try:
        with _database(root) as connection:
            previous = connection.execute(
                "SELECT * FROM incidents WHERE id = ?", (incident_id,)
            ).fetchone()
            if previous:
                count = previous["occurrence_count"] + 1
                connection.execute(
                    """UPDATE incidents SET last_seen=?, occurrence_count=?, status='active',
                       exit_code=?, last_commit=? WHERE id=?""",
                    (now, count, exit_code, commit, incident_id),
                )
                incident = Incident(
                    id=incident_id,
                    status="reopened" if previous["status"] == "resolved" else "recurring",
                    occurrence_count=count,
                    command=previous["command"],
                    signature=signature,
                    last_seen=now,
                    last_commit=commit,
                    diagnosis=previous["diagnosis"],
                    report_path=previous["report_path"],
                )
                return incident
            connection.execute(
                """INSERT INTO incidents
                   (id, command_key, command, signature, exit_code, first_seen, last_seen,
                    occurrence_count, status, last_commit)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 1, 'active', ?)""",
                (incident_id, key, key, signature, exit_code, now, now, commit),
            )
        return Incident(incident_id, "new", 1, key, signature, now, commit, None, None)
    except sqlite3.Error as exc:
        raise TrackerError(f"Could not record failure in the local Failure Atlas: {exc}") from exc


def save_diagnosis(repo: Path, incident_id: str, diagnosis: str) -> None:
    root = find_repo_root(repo.resolve())
    try:
        with _database(root) as connection:
            connection.execute(
                "UPDATE incidents SET diagnosis=? WHERE id=?",
                (diagnosis.strip(), incident_id),
            )
    except sqlite3.Error as exc:
        raise TrackerError(f"Could not save diagnosis in the local Failure Atlas: {exc}") from exc


def save_report_path(repo: Path, incident_id: str, report_path: Path) -> None:
    root = find_repo_root(repo.resolve())
    try:
        relative = report_path.resolve().relative_to(root).as_posix()
        with _database(root) as connection:
            connection.execute(
                "UPDATE incidents SET report_path=? WHERE id=?", (relative, incident_id)
            )
    except (OSError, ValueError, sqlite3.Error) as exc:
        raise TrackerError(f"Could not attach the report to the local incident: {exc}") from exc


def record_success(repo: Path, command: list[str]) -> int:
    """Resolve active failures only for the exact command that just passed."""
    root = find_repo_root(repo.resolve())
    if not (root / ".index" / "failures.sqlite3").is_file():
        return 0
    try:
        with _database(root) as connection:
            cursor = connection.execute(
                "UPDATE incidents SET status='resolved' "
                "WHERE command_key=? AND status='active'",
                (_command_key(command),),
            )
            return cursor.rowcount
    except sqlite3.Error as exc:
        raise TrackerError(f"Could not update the local Failure Atlas: {exc}") from exc


def list_incidents(repo: Path, include_resolved: bool = False) -> list[Incident]:
    root = find_repo_root(repo.resolve())
    database = root / ".index" / "failures.sqlite3"
    if not database.is_file():
        return []
    try:
        with _database(root) as connection:
            where = "" if include_resolved else "WHERE status='active'"
            rows = connection.execute(
                f"SELECT * FROM incidents {where} ORDER BY last_seen DESC"
            ).fetchall()
        return [
            Incident(
                id=row["id"],
                status=row["status"],
                occurrence_count=row["occurrence_count"],
                command=row["command"],
                signature=row["signature"],
                last_seen=row["last_seen"],
                last_commit=row["last_commit"],
                diagnosis=row["diagnosis"],
                report_path=row["report_path"],
            )
            for row in rows
        ]
    except sqlite3.Error as exc:
        raise TrackerError(f"Could not read the local Failure Atlas: {exc}") from exc
