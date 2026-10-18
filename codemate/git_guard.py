"""Preflight the configured upstream for Codemate's push and pull wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import subprocess

from .tracker import TrackerError, find_repo_root


_OID = re.compile(r"^[0-9a-fA-F]{40,64}$")


@dataclass(frozen=True)
class GitPreflight:
    checked: bool
    note: str = ""
    target_ref: str = ""
    local_ahead: int = 0
    remote_ahead: int = 0
    conflicts: tuple[str, ...] = ()
    preview: str = ""


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )


def _output(root: Path, *args: str) -> str:
    result = _git(root, *args)
    if result.returncode:
        return ""
    return result.stdout.strip()


def _tool_version(root: Path) -> tuple[int, int] | None:
    version = _output(root, "--version")
    match = re.search(r"\b(\d+)\.(\d+)", version)
    return (int(match.group(1)), int(match.group(2))) if match else None


def _remote_for(root: Path, operation: str) -> str:
    branch = _output(root, "symbolic-ref", "--quiet", "--short", "HEAD")
    if not branch:
        return ""
    if operation == "push":
        for key in (f"branch.{branch}.pushRemote", "remote.pushDefault", f"branch.{branch}.remote"):
            value = _output(root, "config", "--get", key)
            if value:
                return value
    return _output(root, "config", "--get", f"branch.{branch}.remote")


def _target_ref(root: Path, operation: str) -> str:
    symbolic = "@{push}" if operation == "push" else "@{upstream}"
    return _output(root, "rev-parse", "--symbolic-full-name", symbolic)


def _conflict_paths_and_oid(stdout: str) -> tuple[str, tuple[str, ...]]:
    fields = stdout.split("\0")
    header = fields[0].splitlines()[0].strip() if fields and fields[0].splitlines() else ""
    oid = header if _OID.fullmatch(header) else ""
    paths: list[str] = []
    if oid:
        for field in fields[1:]:
            if not field:
                break
            # The tree OID is the first section. Paths before the empty separator are
            # emitted by --name-only; message sections come after that separator.
            if "\n" in field and not paths:
                field = field.split("\n", 1)[0]
            paths.append(field)
    return oid, tuple(paths)


def _conflict_preview(root: Path, tree_oid: str, paths: tuple[str, ...]) -> str:
    sections = []
    for path in paths[:8]:
        result = subprocess.run(
            ["git", "-C", str(root), "cat-file", "blob", f"{tree_oid}:{path}"],
            capture_output=True,
            check=False,
        )
        if result.returncode or b"\0" in result.stdout:
            continue
        text = result.stdout.decode("utf-8", errors="replace")
        sections.append(f"Conflict preview for {path}:\n{text[:6000]}")
    return "\n\n".join(sections)[:24_000]


def preflight_git(path: Path, operation: str, git_args: list[str]) -> GitPreflight:
    """Compare the current commit with its configured push/upstream target."""
    root = find_repo_root(path.resolve())
    if git_args:
        return GitPreflight(False, "Custom Git arguments use a target Codemate cannot safely infer; forwarding directly to Git.")
    for marker in ("MERGE_HEAD", "CHERRY_PICK_HEAD", "REVERT_HEAD", "REBASE_HEAD"):
        if _output(root, "rev-parse", "--git-path", marker):
            marker_path = Path(_output(root, "rev-parse", "--git-path", marker))
            if not marker_path.is_absolute():
                marker_path = root / marker_path
            if marker_path.exists():
                return GitPreflight(False, "A merge, rebase, cherry-pick, or revert is already in progress; forwarding directly to Git.")

    target = _target_ref(root, operation)
    if not target:
        return GitPreflight(False, "No configured push/upstream branch was found; forwarding directly to Git.")
    remote = _remote_for(root, operation)
    if remote and remote != ".":
        fetched = _git(root, "fetch", "--no-tags", "--quiet", remote)
        if fetched.returncode:
            detail = fetched.stderr.strip() or fetched.stdout.strip() or "fetch failed"
            return GitPreflight(False, f"Could not refresh {remote!r} before the Git check: {detail}")
        # Re-resolve after fetch in case the symbolic push/upstream ref changed.
        target = _target_ref(root, operation) or target
    if not _output(root, "rev-parse", "--verify", target):
        return GitPreflight(False, f"The configured target {target!r} is not available; forwarding directly to Git.")

    version = _tool_version(root)
    if version is None or version < (2, 38):
        return GitPreflight(False, "Git 2.38 or newer is needed for a reliable merge preview; forwarding directly to Git.", target)
    counts = _output(root, "rev-list", "--left-right", "--count", f"HEAD...{target}")
    try:
        local_ahead, remote_ahead = (int(value) for value in counts.split())
    except (TypeError, ValueError):
        return GitPreflight(False, f"Could not compare HEAD with {target}; forwarding directly to Git.", target)

    if not local_ahead or not remote_ahead:
        if operation == "push" and remote_ahead:
            return GitPreflight(
                True,
                f"The remote target is {remote_ahead} commit(s) ahead. Git will likely reject this push; pull or rebase before pushing.",
                target,
                local_ahead,
                remote_ahead,
            )
        return GitPreflight(True, "", target, local_ahead, remote_ahead)

    preview = _git(root, "merge-tree", "--write-tree", "--name-only", "-z", "--messages", "HEAD", target)
    if preview.returncode == 0:
        return GitPreflight(True, "", target, local_ahead, remote_ahead)
    if preview.returncode != 1:
        detail = preview.stderr.strip() or "Git could not calculate a merge preview"
        return GitPreflight(False, f"Merge preview unavailable: {detail}; forwarding directly to Git.", target)
    tree_oid, conflicts = _conflict_paths_and_oid(preview.stdout)
    if not tree_oid or not conflicts:
        return GitPreflight(False, "Git reported a merge-preview issue but did not provide conflict paths; forwarding directly to Git.", target)
    conflict_preview = _conflict_preview(root, tree_oid, conflicts)
    note = (
        f"Preflight found {len(conflicts)} likely merge conflict(s) between HEAD and {target}: "
        + ", ".join(conflicts)
    )
    if operation == "push":
        note += ". A push cannot merge remote changes; it will likely be rejected until the branches are integrated."
    return GitPreflight(True, note, target, local_ahead, remote_ahead, conflicts, conflict_preview)
