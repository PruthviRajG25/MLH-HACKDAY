"""Command-line interface for Codemate."""

from __future__ import annotations

import argparse
import getpass
import shlex
import subprocess
import sys
from pathlib import Path
from typing import TextIO

from .ask import ask_repository
from .agent import apply_changes, repair_repository, render_diff
from .config import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL as DEFAULT_CHAT_MODEL,
    DEFAULT_PROVIDER,
    FIELDS,
    config_path,
    display_config,
    load_config,
    save_config,
    storage_key,
    validate,
)
from .diagnostics import run_diagnostics
from .error_report import create_error_report
from .evaluation import evaluate_retrieval
from .failure_atlas import (
    Incident,
    list_incidents,
    record_failure,
    record_success,
    save_diagnosis,
    save_report_path,
)
from .git_guard import GitPreflight, preflight_git
from .indexer import DEFAULT_MAX_FILE_BYTES, DEFAULT_MODEL as DEFAULT_EMBEDDING_MODEL, IndexResult, index_repository
from .tracker import TrackerError, find_repo_root, track_repository
from .terminal_markdown import render_markdown


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="codemate", description="Track a Git codebase locally.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init", help="configure the AI provider for this user")
    doctor = subparsers.add_parser("doctor", help="check AI configuration, API access, and the local index")
    doctor.add_argument("--repo", type=Path, default=Path.cwd(), help="repository (default: current directory)")
    evaluate = subparsers.add_parser("evaluate", help="measure retrieval against expected source files")
    evaluate.add_argument("--repo", type=Path, default=Path.cwd(), help="repository (default: current directory)")
    evaluate.add_argument("--file", type=Path, help="evaluation JSON (default: .codemate-eval.json in repository)")
    evaluate.add_argument("--top-k", type=int, default=6, help="number of retrieved chunks per question")
    config = subparsers.add_parser("config", help="view or change saved AI settings")
    config_subparsers = config.add_subparsers(dest="config_action")
    config_subparsers.add_parser("list", help="show saved AI settings")
    set_config = config_subparsers.add_parser("set", help="change one saved AI setting")
    set_config.add_argument("field", choices=FIELDS)
    set_config.add_argument("value", nargs="?", help="omit for a hidden API key prompt")
    unset_config = config_subparsers.add_parser("unset", help="remove a saved setting")
    unset_config.add_argument("field", choices=FIELDS)
    track = subparsers.add_parser("track", help="snapshot current repository files into .index/")
    track.add_argument("--repo", type=Path, default=Path.cwd(), help="repository or subdirectory (default: current directory)")
    index = subparsers.add_parser("index", help="chunk and embed current text files locally")
    index.add_argument("--repo", type=Path, default=Path.cwd(), help="repository or subdirectory (default: current directory)")
    index.add_argument(
        "--model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"local embedding model (default: {DEFAULT_EMBEDDING_MODEL})",
    )
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
    ask.add_argument("--model", help="chat model name (overrides saved configuration)")
    ask.add_argument(
        "--provider",
        choices=("gemini", "openai-compatible"),
        help="chat API provider (overrides saved configuration)",
    )
    ask.add_argument(
        "--base-url",
        help="provider API base URL (overrides saved configuration)",
    )
    ask.add_argument(
        "--api-key-env",
        default="GEMINI_API_KEY",
        metavar="ENV_VAR",
        help="environment variable name for the API key (default: GEMINI_API_KEY)",
    )
    run = subparsers.add_parser(
        "run",
        help="run a repository command and diagnose failures with the indexed codebase",
    )
    run.add_argument("--repo", type=Path, default=Path.cwd(), help="working directory (default: current directory)")
    run.add_argument("run_command", nargs=argparse.REMAINDER, help="command and arguments to run")
    git_command = subparsers.add_parser("git", help="preflight and run Git push/pull")
    git_command.add_argument("--repo", type=Path, default=Path.cwd(), help="repository (default: current directory)")
    git_command.add_argument("git_operation", choices=("push", "pull"))
    git_command.add_argument("git_args", nargs=argparse.REMAINDER, help="arguments forwarded to Git")
    failures = subparsers.add_parser("failures", help="show the repository's local failure history")
    failures.add_argument("--repo", type=Path, default=Path.cwd(), help="repository (default: current directory)")
    failures.add_argument("--all", action="store_true", help="include resolved incidents")
    return parser


def _prompt_value(label: str, default: str) -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{label}{suffix}: ").strip()
    return value or default


def _write_command_output(stream: TextIO, value: str) -> None:
    if not value:
        return
    stream.write(value)
    if not value.endswith("\n"):
        stream.write("\n")


def _diagnose_failed_command(
    command: list[str], cwd: Path, stdout: str, stderr: str, exit_code: int,
    incident: Incident | None = None, *, allow_repair: bool = True,
    allow_rerun: bool = True,
) -> int | None:
    limit = 12_000
    combined_output = (
        f"stdout:\n{stdout}\n\nstderr:\n{stderr}"
        if stdout
        else f"stderr:\n{stderr}"
    )
    if len(combined_output) > limit:
        combined_output = "[Earlier command output omitted. Showing the final portion.]\n" + combined_output[-limit:]
    question = (
        f"Diagnose this failed repository command. Command: {shlex.join(command)}. "
        f"Exit code: {exit_code}. Use the error output to locate likely causes in the retrieved code. "
        "Explain the likely cause, cite repository evidence, and suggest the next debugging step. "
        "If the evidence is insufficient, say so.\n\nCommand output:\n"
        f"{combined_output}"
    )
    if incident and incident.diagnosis:
        question += (
            "\n\nFailure Atlas history: this failure matches an earlier incident "
            f"seen {incident.occurrence_count - 1} time(s). Its previous diagnosis was:\n"
            f"{incident.diagnosis}\nTreat that diagnosis as historical context, not proof; "
            "re-check it against the current output and retrieved code."
        )
    print("\nCodemate diagnosis:")
    try:
        answer, hits, embedding_model = ask_repository(question, cwd)
    except TrackerError as exc:
        print(f"Could not diagnose this failure: {exc}", file=sys.stderr)
        return
    print(render_markdown(answer))
    if incident:
        try:
            save_diagnosis(cwd, incident.id, answer)
        except TrackerError as exc:
            print(f"Could not save diagnosis to Failure Atlas: {exc}", file=sys.stderr)
    print(f"\nSources (retrieved locally with {embedding_model}):")
    for hit in hits:
        detail = f"; {hit.language}" if hit.language else ""
        if hit.symbol:
            detail += f"; {hit.symbol}"
        print(f"[{hit.label}] {hit.path}:{hit.start_line}-{hit.end_line} (score {hit.score:.3f}{detail})")
    if not sys.stdin.isatty():
        return None
    try:
        choice = input("\nCreate an error report under report/? [y/N]: ").strip().lower()
    except EOFError:
        choice = ""
    if choice in {"y", "yes"}:
        try:
            report_path = create_error_report(
                cwd,
                command,
                exit_code,
                combined_output,
                answer,
                hits,
                embedding_model,
            )
        except TrackerError as exc:
            print(f"Could not create error report: {exc}", file=sys.stderr)
        else:
            print(f"Error report saved to {report_path}")
            if incident:
                try:
                    save_report_path(cwd, incident.id, report_path)
                except TrackerError as exc:
                    print(f"Could not link report to Failure Atlas: {exc}", file=sys.stderr)
    if not allow_repair:
        return None
    repaired_status = _offer_repair(
        command, cwd, stdout, stderr, exit_code, answer, incident,
        allow_rerun=allow_rerun,
    )
    if repaired_status is not None:
        return repaired_status
    return None


def _offer_repair(
    command: list[str], cwd: Path, stdout: str, stderr: str, exit_code: int,
    diagnosis: str, incident: Incident | None, *, allow_rerun: bool = True,
) -> int | None:
    if not sys.stdin.isatty():
        return None
    try:
        choice = input("\nAsk Codemate to inspect files and propose a code fix? [y/N]: ").strip().lower()
    except EOFError:
        return None
    if choice not in {"y", "yes"}:
        return None

    current_stdout, current_stderr = stdout, stderr
    current_exit = exit_code
    current_diagnosis = diagnosis
    current_incident = incident
    for attempt in range(1, 4):
        failure_output = f"stdout:\n{current_stdout}\n\nstderr:\n{current_stderr}"
        try:
            result = repair_repository(
                "Propose the smallest safe code change that addresses this failure. Inspect the relevant "
                "source files, explain your reasoning briefly, and stage a change only when repository "
                "evidence supports it.",
                cwd,
                diagnosis=current_diagnosis,
                command=command,
                output=failure_output[-12_000:],
            )
        except TrackerError as exc:
            print(f"Could not start the repair agent: {exc}", file=sys.stderr)
            return None
        print("\nCodemate repair analysis:")
        print(render_markdown(result.message))
        if not result.changes:
            print("The agent did not stage a code change.")
            return None
        try:
            diff = render_diff(result.changes)
        except TrackerError as exc:
            print(f"Could not show the proposed changes: {exc}", file=sys.stderr)
            return None
        print("\nProposed changes (not applied):\n")
        print(diff)
        try:
            approval = input("Apply these exact changes? [y/N]: ").strip().lower()
        except EOFError:
            return None
        if approval not in {"y", "yes"}:
            print("Changes discarded.")
            return None
        try:
            changed_paths = apply_changes(cwd, result.changes)
        except TrackerError as exc:
            print(f"Could not apply the reviewed changes: {exc}", file=sys.stderr)
            return None
        print(f"Applied: {', '.join(changed_paths)}")
        if not allow_rerun:
            print("Codemate left Git in control. Review the working tree, stage resolved files, then continue the merge or rebase as appropriate.")
            return None
        try:
            rerun = input("Run the original command to check the fix? [Y/n]: ").strip().lower()
        except EOFError:
            return None
        if rerun in {"n", "no"}:
            return None

        try:
            process = subprocess.run(
                command,
                cwd=cwd,
                capture_output=True,
                text=True,
                errors="replace",
                check=False,
            )
            current_stdout, current_stderr, current_exit = process.stdout, process.stderr, process.returncode
        except OSError as exc:
            current_stdout, current_stderr, current_exit = "", str(exc), 127
        _write_command_output(sys.stdout, current_stdout)
        _write_command_output(sys.stderr, current_stderr)
        if current_exit == 0:
            if current_incident:
                try:
                    record_success(cwd, command)
                except TrackerError:
                    pass
            print("Codemate verification: the command passed after the approved changes.")
            return 0
        if not current_stderr and not current_stdout:
            current_stderr = f"Command exited with code {current_exit}."
            print(current_stderr, file=sys.stderr)
        if attempt == 3:
            print("Repair limit reached after three reviewed attempts.")
            return current_exit if current_exit > 0 else 128 + abs(current_exit)
        try:
            repo_root = find_repo_root(cwd.resolve())
            if (repo_root / ".index" / "vectors.sqlite3").is_file():
                index_repository(repo_root)
            question = (
                f"Diagnose the still-failing command {shlex.join(command)} (exit code {current_exit}) "
                "after an approved repair. Re-check the current source rather than assuming the previous "
                "diagnosis is correct.\n\nCommand output:\n"
                f"stdout:\n{current_stdout}\n\nstderr:\n{current_stderr}"
            )
            current_diagnosis, hits, embedding_model = ask_repository(question, cwd)
            print("\nUpdated Codemate diagnosis:")
            print(render_markdown(current_diagnosis))
            print(f"\nSources (retrieved locally with {embedding_model}):")
            for hit in hits:
                print(f"[{hit.label}] {hit.path}:{hit.start_line}-{hit.end_line} (score {hit.score:.3f})")
            current_incident = record_failure(cwd, command, current_exit, f"stdout:\n{current_stdout}\n\nstderr:\n{current_stderr}")
            if current_incident:
                save_diagnosis(cwd, current_incident.id, current_diagnosis)
        except TrackerError as exc:
            print(f"Could not analyze the repeated failure: {exc}", file=sys.stderr)
            return current_exit if current_exit > 0 else 128 + abs(current_exit)
        try:
            continue_choice = input("Ask the agent to inspect this new failure and propose another fix? [y/N]: ").strip().lower()
        except EOFError:
            return current_exit if current_exit > 0 else 128 + abs(current_exit)
        if continue_choice not in {"y", "yes"}:
            return current_exit if current_exit > 0 else 128 + abs(current_exit)
    return current_exit if current_exit > 0 else 128 + abs(current_exit)


def _analyze_git_preflight(preflight: GitPreflight, cwd: Path, operation: str) -> None:
    if not preflight.note:
        return
    print(f"Codemate Git preflight: {preflight.note}", file=sys.stderr)
    if not preflight.conflicts:
        return
    question = (
        f"Analyze this Git {operation} preflight. Git preview found likely conflicts in: "
        f"{', '.join(preflight.conflicts)}. Explain what the conflict means and suggest a safe way "
        "to resolve it. Use only the supplied preview and retrieved repository excerpts; distinguish "
        "facts from hypotheses and do not propose changing files automatically.\n\n"
        f"Merge preview:\n{preflight.preview or '[No textual preview available.]'}"
    )
    try:
        answer, hits, model = ask_repository(question, cwd)
    except TrackerError as exc:
        print(f"Could not add a repository-specific analysis: {exc}", file=sys.stderr)
        return
    print("\nCodemate conflict analysis:")
    print(render_markdown(answer))
    print(f"\nSources (retrieved locally with {model}):")
    for hit in hits:
        print(f"[{hit.label}] {hit.path}:{hit.start_line}-{hit.end_line} (score {hit.score:.3f})")


def _run_guarded_git(cwd: Path, operation: str, git_args: list[str]) -> int:
    try:
        preflight = preflight_git(cwd, operation, git_args)
    except (OSError, TrackerError) as exc:
        preflight = GitPreflight(False, f"Preflight unavailable ({exc}); forwarding directly to Git.")
    _analyze_git_preflight(preflight, cwd, operation)
    command = ["git", operation, *git_args]
    try:
        process = subprocess.run(command, cwd=cwd, capture_output=True, text=True, errors="replace", check=False)
        stdout, stderr, code = process.stdout, process.stderr, process.returncode
    except OSError as exc:
        stdout, stderr, code = "", str(exc), 127
    _write_command_output(sys.stdout, stdout)
    _write_command_output(sys.stderr, stderr)
    if code == 0:
        return 0
    if not stderr and not stdout:
        stderr = f"Git {operation} exited with code {code}."
        print(stderr, file=sys.stderr)
    try:
        incident = record_failure(cwd, command, code, f"stdout:\n{stdout}\n\nstderr:\n{stderr}")
    except TrackerError:
        incident = None
    failure_text = f"{stdout}\n{stderr}".casefold()
    conflict_failure = bool(preflight.conflicts) or any(
        marker in failure_text
        for marker in ("conflict", "could not apply", "automatic merge failed", "fix conflicts")
    )
    # Push failures concern remote history or transport; source edits cannot fix them.
    # Only pull failures that look like merge conflicts offer code repair; Git is never
    # rerun automatically after such a proposal.
    _diagnose_failed_command(
        command, cwd, stdout, stderr, code, incident,
        allow_repair=(operation == "pull" and conflict_failure), allow_rerun=False,
    )
    return code if code > 0 else 128 + abs(code)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "failures":
        try:
            incidents = list_incidents(args.repo, include_resolved=args.all)
        except TrackerError as exc:
            print(f"codemate: error: {exc}", file=sys.stderr)
            return 2
        if not incidents:
            print("No active incidents. Use `codemate failures --all` to include resolved history.")
            return 0
        for incident in incidents:
            short_id = incident.id[:8]
            commit = f" · commit {incident.last_commit}" if incident.last_commit else ""
            print(
                f"{incident.status.upper():8} {short_id} · seen {incident.occurrence_count} "
                f"time(s) · {incident.last_seen}{commit}"
            )
            print(f"  Command: {incident.command}")
            signature = incident.signature.replace("\n", " | ")
            print(f"  Failure: {signature[:220]}")
            if incident.diagnosis:
                first_line = next((line.strip() for line in incident.diagnosis.splitlines() if line.strip()), "")
                if first_line:
                    print(f"  Last diagnosis: {first_line[:220]}")
            if incident.report_path:
                print(f"  Report: {incident.report_path}")
        return 0
    if args.command == "doctor":
        try:
            results = run_diagnostics(args.repo)
        except TrackerError as exc:
            print(f"codemate: error: {exc}", file=sys.stderr)
            return 2
        for ok, message in results:
            print(f"{'OK' if ok else 'CHECK'}  {message}")
        return 0 if all(ok for ok, _ in results) else 1
    if args.command == "evaluate":
        try:
            results, total = evaluate_retrieval(args.repo, args.file, args.top_k)
        except TrackerError as exc:
            print(f"codemate: error: {exc}", file=sys.stderr)
            return 2
        hits = 0
        reciprocal_ranks = []
        for number, result in enumerate(results, start=1):
            rank = result["rank"]
            if rank is not None:
                hits += 1
                reciprocal_ranks.append(1 / rank)
                status = f"hit at {rank}"
            else:
                reciprocal_ranks.append(0.0)
                status = "miss"
            print(f"{number}. {status}: {result['question']}")
            if rank is None:
                print(f"   expected one of: {', '.join(result['expected'])}")
                if result.get("error"):
                    print(f"   {result['error']}")
        print(f"Recall@{args.top_k}: {hits}/{total} ({hits / total:.0%})")
        print(f"MRR@{args.top_k}: {sum(reciprocal_ranks) / total:.3f}")
        return 0 if hits == total else 1
    if args.command == "run":
        command = args.run_command
        if command and command[0] == "--":
            command = command[1:]
        if not command:
            print("codemate: error: provide a command to run, for example `codemate run pytest`", file=sys.stderr)
            return 2
        if len(command) >= 2 and command[0] == "git" and command[1] in {"push", "pull"}:
            return _run_guarded_git(args.repo, command[1], command[2:])
        try:
            result = subprocess.run(
                command,
                cwd=args.repo,
                capture_output=True,
                text=True,
                errors="replace",
                check=False,
            )
            stdout, stderr, exit_code = result.stdout, result.stderr, result.returncode
        except OSError as exc:
            stdout, stderr, exit_code = "", str(exc), 127
        _write_command_output(sys.stdout, stdout)
        if exit_code == 0:
            _write_command_output(sys.stderr, stderr)
            try:
                record_success(args.repo, command)
            except TrackerError:
                pass
            return 0
        if stderr:
            _write_command_output(sys.stderr, stderr)
        else:
            print(f"Command exited with code {exit_code}.", file=sys.stderr)
            stderr = f"Command exited with code {exit_code}."
        sys.stdout.flush()
        sys.stderr.flush()
        try:
            repo_root = find_repo_root(args.repo.resolve())
            if (repo_root / ".index" / "vectors.sqlite3").is_file():
                updated = index_repository(repo_root)
                if updated.added_files or updated.updated_files or updated.removed_files:
                    print("Updated the local code index before diagnosis.", file=sys.stderr)
        except TrackerError as exc:
            print(f"Could not refresh the local code index: {exc}", file=sys.stderr)
        incident = None
        try:
            incident = record_failure(
                args.repo, command, exit_code,
                f"stdout:\n{stdout}\n\nstderr:\n{stderr}",
            )
            print(
                f"Failure Atlas: {incident.status} incident {incident.id[:8]} "
                f"(seen {incident.occurrence_count} time(s))."
            )
        except TrackerError as exc:
            print(f"Could not update Failure Atlas: {exc}", file=sys.stderr)
        repair_status = _diagnose_failed_command(command, args.repo, stdout, stderr, exit_code, incident)
        if repair_status is not None:
            return repair_status
        return exit_code if exit_code > 0 else 128 + abs(exit_code)
    if args.command == "git":
        return _run_guarded_git(args.repo, args.git_operation, args.git_args)
    if args.command == "init":
        try:
            current = load_config()
            previous_provider = current.get("provider", DEFAULT_PROVIDER)
            provider = _prompt_value(
                "Provider [gemini/openai-compatible]", previous_provider
            )
            validate("provider", provider)
            same_provider = provider == previous_provider
            default_model = (
                current.get("model", "") if same_provider
                else DEFAULT_CHAT_MODEL if provider == "gemini" else ""
            )
            model = _prompt_value("Model", default_model)
            validate("model", model)
            default_url = current.get("base_url", "") if same_provider else ""
            if not default_url:
                default_url = DEFAULT_BASE_URL if provider == "gemini" else "https://api.openai.com/v1"
            base_url = _prompt_value("API base URL", default_url)
            validate("base-url", base_url)
            api_key = getpass.getpass("API key (leave blank to keep current): ")
            updated = {**current, "provider": provider, "model": model, "base_url": base_url}
            if api_key:
                updated["api_key"] = api_key
            elif not updated.get("api_key"):
                raise TrackerError("An API key is required. Run `codemate config set api-key` to add one.")
            path = save_config(updated)
        except (TrackerError, EOFError, KeyboardInterrupt) as exc:
            if isinstance(exc, KeyboardInterrupt):
                print("\nInitialization cancelled.", file=sys.stderr)
                return 130
            print(f"codemate: error: {exc}", file=sys.stderr)
            return 2
        print(f"AI configuration saved to {path}")
        print("Use `codemate config set FIELD` to change a setting.")
        return 0
    if args.command == "config":
        action = args.config_action or "list"
        try:
            values = load_config()
            if action == "list":
                print(f"Configuration: {config_path()}")
                print("\n".join(display_config(values)))
                return 0
            if action == "set":
                if args.field == "api-key":
                    if args.value is not None:
                        raise TrackerError("Omit the API key value to enter it at the hidden prompt.")
                    value = getpass.getpass("API key: ")
                elif args.value is not None:
                    value = args.value
                else:
                    raise TrackerError(f"Provide a value for {args.field}.")
                validate(args.field, value)
                values[storage_key(args.field)] = value
            elif action == "unset":
                values.pop(storage_key(args.field), None)
            path = save_config(values)
        except (TrackerError, EOFError, KeyboardInterrupt) as exc:
            if isinstance(exc, KeyboardInterrupt):
                print("\nConfiguration change cancelled.", file=sys.stderr)
                return 130
            print(f"codemate: error: {exc}", file=sys.stderr)
            return 2
        print(f"Updated {args.field} in {path}")
        return 0
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
        print(render_markdown(answer))
        print(f"\nSources (retrieved locally with {embedding_model}):")
        for hit in hits:
            detail = f"; {hit.language}" if hit.language else ""
            if hit.symbol:
                detail += f"; {hit.symbol}"
            print(
                f"[{hit.label}] {hit.path}:{hit.start_line}-{hit.end_line} "
                f"(score {hit.score:.3f}{detail})"
            )
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
