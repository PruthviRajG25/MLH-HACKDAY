"""User-level AI configuration for Codemate."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from .tracker import TrackerError


DEFAULT_PROVIDER = "gemini"
DEFAULT_MODEL = "gemma-4-26b-a4b-it"
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_API_KEY = "ollama"
PROVIDERS = ("gemini", "openai-compatible", "ollama")
FIELDS = ("provider", "model", "base-url", "api-key")


def config_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME")
    return (Path(base).expanduser() if base else Path.home() / ".config") / "codemate" / "config.json"


def load_config() -> dict[str, str]:
    path = config_path()
    try:
        data: Any = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        raise TrackerError(f"Could not read Codemate configuration at {path}: {exc}") from exc
    if not isinstance(data, dict) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in data.items()
    ):
        raise TrackerError(f"Codemate configuration at {path} must be a JSON object of string values.")
    return data


def save_config(values: dict[str, str]) -> Path:
    path = config_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd, temporary_name = tempfile.mkstemp(prefix="config.", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(values, handle, indent=2)
                handle.write("\n")
            os.replace(temporary, path)
            os.chmod(path, 0o600)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    except OSError as exc:
        raise TrackerError(f"Could not save Codemate configuration at {path}: {exc}") from exc
    return path


def storage_key(field: str) -> str:
    return field.replace("-", "_")


def validate(field: str, value: str) -> None:
    if field == "provider" and value not in PROVIDERS:
        raise TrackerError("Provider must be 'gemini', 'openai-compatible', or 'ollama'.")
    if not value.strip():
        raise TrackerError(f"{field} cannot be empty.")
    if field == "base-url" and not re.match(r"^https?://", value):
        raise TrackerError("Base URL must start with http:// or https://.")


def display_config(values: dict[str, str]) -> list[str]:
    lines = []
    for field in FIELDS:
        value = values.get(storage_key(field))
        if field == "api-key":
            shown = "(configured)" if value else "(not set)"
        else:
            shown = value or "(not set)"
        lines.append(f"{field}: {shown}")
    return lines
